#!/usr/bin/env bun
/**
 * Local Weight Stream Processor (TypeScript)
 *
 * Processes weight measurements from CSV data using direct method calls instead of API.
 * Uses in-memory storage for state management.
 * Outputs a filtered CSV with only accepted (non-rejected) measurements.
 */

import { parseArgs } from "node:util";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { parse } from "csv-parse/sync";
import { stringify } from "csv-stringify/sync";
import { join } from "node:path";

import {
  WeightProcessorService,
  type MeasurementInput,
  type BatchProcessResult
} from "./src/services/weight_processor_service";
import { ProcessorStateDB } from "./src/core/database/database";
import { ConfigManager } from "./src/config/config_manager";
import { SUPPORTED_WEIGHT_UNITS } from "./src/constants";
import type {
  ProcessResponseData,
  ProcessResult,
} from "./src/models";

/**
 * Load configuration from config.toml, overriding database backend for local processing.
 */
function getDefaultConfig(): any {
  // Load from unified config file (default: ./config.toml)
  const config = ConfigManager.loadConfig();

  // Override database backend for local in-memory processing
  if (config.database) {
    config.database.backend = "memory";
  }

  return config;
}

/**
 * Parse various timestamp formats and return Date in UTC.
 */
function parseTimestamp(dateStr: string): Date {
  if (!dateStr) {
    return new Date();
  }

  try {
    if (dateStr.includes("T")) {
      // Parse ISO format
      const normalized = dateStr.replace("Z", "+00:00");
      return new Date(normalized);
    } else if (dateStr.includes(" ")) {
      // Parse space-separated format
      // Try with milliseconds first, then without
      const withMs = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d+)$/;
      const withoutMs = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})$/;

      if (withMs.test(dateStr) || withoutMs.test(dateStr)) {
        return new Date(dateStr.replace(" ", "T") + "Z");
      }
      throw new Error(`Cannot parse space-separated date: ${dateStr}`);
    } else {
      // Parse date only
      return new Date(dateStr + "T00:00:00Z");
    }
  } catch {
    // Fallback to current time if parsing fails
    return new Date();
  }
}

/**
 * CSV row type (flexible to handle both old and new column names)
 */
interface CsvRow {
  id?: string;
  measurement_id?: string;
  user_id: string;
  value_quantity?: string;
  weight?: string;
  unit: string;
  timestamp?: string;
  effective_date_time?: string;
  effectiveDateTime?: string;
  source_type: string;
  [key: string]: any;
}

/**
 * Load options for CSV data
 */
interface LoadOptions {
  maxUsers: number;
  maxRows: number;
  minReadings: number;
}

/**
 * Data quality statistics
 */
interface DataQualityStats {
  totalRows: number;
  invalidWeight: number;
  parseErrors: number;
  unitRejected: number;
  rejectedUnits: Map<string, number>;
  bsaMeasurements: number;
  missingData: number;
}

/**
 * Load CSV data and group measurements by user_id.
 */
function loadCsvData(
  csvPath: string,
  options: LoadOptions
): {
  userMeasurements: Map<string, MeasurementInput[]>;
  originalRows: CsvRow[];
} {
  const userMeasurements = new Map<string, MeasurementInput[]>();
  const originalRows: CsvRow[] = [];

  // Statistics for rejected data
  const stats: DataQualityStats = {
    totalRows: 0,
    invalidWeight: 0,
    parseErrors: 0,
    unitRejected: 0,
    rejectedUnits: new Map(),
    bsaMeasurements: 0,
    missingData: 0,
  };

  console.log(`Loading data from ${csvPath}...`);

  // Read and parse CSV file
  const content = readFileSync(csvPath, "utf-8");
  const records = parse(content, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
  }) as CsvRow[];

  let rowCount = 0;

  for (const row of records) {
    rowCount++;
    stats.totalRows++;

    if (options.maxRows > 0 && rowCount > options.maxRows) {
      break;
    }

    // Handle both old and new column names for ID
    const measurementId = row.id || row.measurement_id;
    const userId = row.user_id;
    if (!userId || !measurementId) {
      stats.missingData++;
      continue;
    }

    // Parse and validate weight - handle both old and new column names
    const weightStr = (row.value_quantity || row.weight || "").trim();
    if (!weightStr || weightStr.toUpperCase() === "NULL") {
      stats.missingData++;
      continue;
    }

    let weight: number;
    try {
      weight = parseFloat(weightStr);
      // Validate weight is reasonable (basic sanity check)
      if (weight <= 0 || weight > 1000) {
        stats.invalidWeight++;
        continue;
      }
      // Check for NaN and Inf
      if (isNaN(weight) || !isFinite(weight)) {
        stats.invalidWeight++;
        continue;
      }
    } catch {
      stats.parseErrors++;
      continue;
    }

    // Parse other fields - handle both old and new column names
    const dateStr = row.effective_date_time || row.effectiveDateTime || row.timestamp || "";
    const source = row.source_type || "unknown";
    const unit = (row.unit || "").trim(); // NO DEFAULT - must be explicit

    // Skip BSA measurements (Body Surface Area)
    if (
      source.toUpperCase().includes("BSA") ||
      unit === "m2" ||
      unit === "m²"
    ) {
      stats.bsaMeasurements++;
      continue;
    }

    // Early unit validation - check against whitelist
    if (!unit) {
      stats.unitRejected++;
      stats.rejectedUnits.set(
        "<missing>",
        (stats.rejectedUnits.get("<missing>") || 0) + 1
      );
      continue;
    }

    const unitLower = unit.toLowerCase().trim();
    if (!SUPPORTED_WEIGHT_UNITS.has(unitLower)) {
      stats.unitRejected++;
      stats.rejectedUnits.set(unit, (stats.rejectedUnits.get(unit) || 0) + 1);
      continue;
    }

    // Store original row with unique identifier for tracking
    const originalRow = { ...row };
    (originalRow as any)._row_index = rowCount;
    (originalRow as any)._accepted = false; // Will be updated during processing
    originalRows.push(originalRow);

    // Parse timestamp with error handling
    let timestamp: Date;
    try {
      timestamp = dateStr ? parseTimestamp(dateStr) : new Date();
    } catch {
      // Fallback to current time if parsing fails
      timestamp = new Date();
    }

    // Convert to MeasurementInput for service
    try {
      const measurement: MeasurementInput = {
        measurement_id: measurementId,
        weight: weight,
        unit: unit,
        timestamp: timestamp,
        source: source,
      };

      if (!userMeasurements.has(userId)) {
        userMeasurements.set(userId, []);
      }
      userMeasurements.get(userId)!.push(measurement);

      // Progress update
      if (rowCount % 10000 === 0) {
        console.log(
          `  Loaded ${rowCount.toLocaleString()} rows, ${userMeasurements.size.toLocaleString()} users...`
        );
      }
    } catch (e) {
      stats.parseErrors++;
      continue;
    }
  }

  // Calculate initial totals
  const initialUserCount = userMeasurements.size;
  const initialMeasurementCount = Array.from(userMeasurements.values()).reduce(
    (sum, m) => sum + m.length,
    0
  );

  // Filter users by minimum readings BEFORE applying max_users limit
  if (options.minReadings > 0) {
    const usersBeforeFilter = userMeasurements.size;
    const measurementsBeforeFilter = Array.from(
      userMeasurements.values()
    ).reduce((sum, m) => sum + m.length, 0);

    // Filter out users with fewer than min_readings
    const filteredUserMeasurements = new Map<string, Measurement[]>();
    for (const [uid, measurements] of userMeasurements.entries()) {
      if (measurements.length >= options.minReadings) {
        filteredUserMeasurements.set(uid, measurements);
      }
    }
    userMeasurements.clear();
    for (const [uid, measurements] of filteredUserMeasurements.entries()) {
      userMeasurements.set(uid, measurements);
    }

    // Filter original_rows to match remaining users
    const remainingUserSet = new Set(userMeasurements.keys());
    const filteredRows = originalRows.filter((row) =>
      remainingUserSet.has(row.user_id)
    );
    originalRows.length = 0;
    originalRows.push(...filteredRows);

    const usersFiltered = usersBeforeFilter - userMeasurements.size;
    const measurementsFiltered =
      measurementsBeforeFilter -
      Array.from(userMeasurements.values()).reduce(
        (sum, m) => sum + m.length,
        0
      );

    if (usersFiltered > 0) {
      console.log(
        `\nFiltered out ${usersFiltered.toLocaleString()} users with < ${options.minReadings} readings (${measurementsFiltered.toLocaleString()} measurements)`
      );
      console.log(
        `Remaining: ${userMeasurements.size.toLocaleString()} users with ${Array.from(userMeasurements.values())
          .reduce((sum, m) => sum + m.length, 0)
          .toLocaleString()} measurements`
      );
    }
  }

  // Apply user limit AFTER min_readings filter
  if (options.maxUsers > 0 && userMeasurements.size > options.maxUsers) {
    // Take first N users by sorted order for consistency
    const sortedUsers = Array.from(userMeasurements.keys())
      .sort()
      .slice(0, options.maxUsers);
    const limitedUserMeasurements = new Map<string, Measurement[]>();
    for (const uid of sortedUsers) {
      limitedUserMeasurements.set(uid, userMeasurements.get(uid)!);
    }
    userMeasurements.clear();
    for (const [uid, measurements] of limitedUserMeasurements.entries()) {
      userMeasurements.set(uid, measurements);
    }

    // Filter original_rows to match selected users
    const selectedUserSet = new Set(sortedUsers);
    const filteredRows = originalRows.filter((row) =>
      selectedUserSet.has(row.user_id)
    );
    originalRows.length = 0;
    originalRows.push(...filteredRows);
  }

  // Calculate valid measurements loaded
  const totalMeasurements = Array.from(userMeasurements.values()).reduce(
    (sum, m) => sum + m.length,
    0
  );

  console.log(
    `Loaded ${userMeasurements.size.toLocaleString()} users with ${totalMeasurements.toLocaleString()} total measurements`
  );

  // Report data quality statistics
  if (stats.totalRows > 0) {
    console.log(`\nData Quality Statistics:`);
    console.log(`  Total rows read: ${stats.totalRows.toLocaleString()}`);
    console.log(
      `  Valid measurements: ${totalMeasurements.toLocaleString()}`
    );

    const rejectedTotal =
      stats.invalidWeight +
      stats.parseErrors +
      stats.unitRejected +
      stats.bsaMeasurements +
      stats.missingData;
    console.log(`  Rejected measurements: ${rejectedTotal.toLocaleString()}`);

    if (stats.invalidWeight > 0) {
      console.log(
        `    Invalid weight values: ${stats.invalidWeight.toLocaleString()}`
      );
    }
    if (stats.parseErrors > 0) {
      console.log(`    Parse errors: ${stats.parseErrors.toLocaleString()}`);
    }
    if (stats.unitRejected > 0) {
      console.log(
        `    Invalid/unsupported units: ${stats.unitRejected.toLocaleString()}`
      );
    }
    if (stats.bsaMeasurements > 0) {
      console.log(
        `    BSA measurements (filtered): ${stats.bsaMeasurements.toLocaleString()}`
      );
    }
    if (stats.missingData > 0) {
      console.log(
        `    Missing required data: ${stats.missingData.toLocaleString()}`
      );
    }

    // Report rejected units breakdown
    if (stats.rejectedUnits.size > 0) {
      console.log(`\n  Top rejected units:`);
      const sortedUnits = Array.from(stats.rejectedUnits.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);
      for (const [unit, count] of sortedUnits) {
        console.log(`    '${unit}': ${count.toLocaleString()} measurements`);
      }
    }
  }

  return { userMeasurements, originalRows };
}

/**
 * Tracks which measurements were accepted during processing.
 */
class AcceptanceTracker {
  private acceptedMeasurements = new Set<string>(); // Track by "user_id|timestamp"
  private userAcceptanceDetails = new Map<string, any[]>();
  private userDetailedResults = new Map<string, any[]>();

  clear(): void {
    this.acceptedMeasurements.clear();
    this.userAcceptanceDetails.clear();
    this.userDetailedResults.clear();
  }

  markMeasurementAccepted(
    userId: string,
    timestamp: string,
    additionalInfo?: Record<string, any>
  ): void {
    this.acceptedMeasurements.add(`${userId}|${timestamp}`);
    if (!this.userAcceptanceDetails.has(userId)) {
      this.userAcceptanceDetails.set(userId, []);
    }

    const info: any = { timestamp, accepted: true };
    if (additionalInfo) {
      Object.assign(info, additionalInfo);
    }
    this.userAcceptanceDetails.get(userId)!.push(info);
  }

  markBatchResults(
    userId: string,
    measurements: MeasurementInput[],
    responseData: ProcessResponseData
  ): void {
    // Extract results from response
    for (let i = 0; i < responseData.results.length; i++) {
      const result = responseData.results[i];
      if (result.accepted && i < measurements.length) {
        const timestamp = typeof measurements[i].timestamp === 'string'
          ? new Date(measurements[i].timestamp as string).toISOString()
          : (measurements[i].timestamp as Date).toISOString();
        this.markMeasurementAccepted(userId, timestamp, {
          qualityScore: result.qualityScore,
          kalmanEstimate: result.kalmanEstimate,
          processingResult: result,
        });
      }
    }
  }

  storeDetailedResult(
    userId: string,
    measurement: MeasurementInput,
    result: ProcessResult,
    wasReset: boolean = false,
    resetInfo?: Record<string, any>
  ): void {
    if (!this.userDetailedResults.has(userId)) {
      this.userDetailedResults.set(userId, []);
    }

    // Get kalman estimate with fallback to raw weight
    const kalmanEst = result.kalmanEstimate || measurement.weight;
    const kalmanUnc = result.kalmanUncertainty || 1.0;

    // Create detailed result dict
    const timestamp_str = typeof measurement.timestamp === 'string'
      ? new Date(measurement.timestamp).toISOString()
      : measurement.timestamp.toISOString();

    const detail: any = {
      timestamp: timestamp_str,
      rawWeight: measurement.weight,
      source: measurement.source,
      accepted: result.accepted,
      filteredWeight: result.accepted ? kalmanEst : measurement.weight,
      qualityScore: result.qualityScore || 0.0,
      kalmanEstimate: kalmanEst,
      kalmanVariance: kalmanUnc,
      innovation: result.innovation || 0.0,
      normalizedInnovation: result.normalizedInnovation || 0.0,
      confidence: result.confidence || 0.95,
      trend: result.trend || 0.0,
      trendWeekly: result.trendWeekly || 0.0,
      kalmanConfidenceUpper: kalmanEst + 2 * kalmanUnc,
      kalmanConfidenceLower: kalmanEst - 2 * kalmanUnc,
      qualityComponents: result.qualityComponents || {},
      wasReset: wasReset,
    };

    if (!result.accepted) {
      detail.reason = result.rejectionReason || "Unknown";
    }

    if (resetInfo) {
      Object.assign(detail, resetInfo);
    }

    this.userDetailedResults.get(userId)!.push(detail);
  }

  isAccepted(userId: string, timestamp: string): boolean {
    return this.acceptedMeasurements.has(`${userId}|${timestamp}`);
  }

  getDetailedResults(userId: string): any[] {
    return this.userDetailedResults.get(userId) || [];
  }
}

/**
 * Write filtered CSV with only accepted measurements.
 */
function writeFilteredCsv(
  originalRows: CsvRow[],
  acceptanceTracker: AcceptanceTracker,
  outputPath: string
): number {
  if (originalRows.length === 0) {
    console.log("No original rows to filter");
    return 0;
  }

  console.log(`\nWriting filtered CSV to ${outputPath}...`);

  // Get fieldnames from first row (excluding internal tracking fields)
  const fieldnames = Object.keys(originalRows[0]).filter(
    (k) => !k.startsWith("_")
  );

  let acceptedCount = 0;
  const totalCount = originalRows.length;

  const acceptedRows: any[] = [];

  for (const row of originalRows) {
    const userId = row.user_id;
    // Handle both old and new column names for timestamp
    const timestamp = row.effective_date_time || row.effectiveDateTime || row.timestamp;

    // Convert timestamp to ISO format to match what's stored in AcceptanceTracker
    if (timestamp) {
      const normalizedTimestamp = parseTimestamp(timestamp).toISOString();

      if (
        userId &&
        normalizedTimestamp &&
        acceptanceTracker.isAccepted(userId, normalizedTimestamp)
      ) {
        // Write only the original CSV fields (exclude tracking fields)
        const filteredRow: any = {};
        for (const key of fieldnames) {
          filteredRow[key] = row[key];
        }
        acceptedRows.push(filteredRow);
        acceptedCount++;
      }
    }
  }

  // Write CSV
  const csvContent = stringify(acceptedRows, {
    header: true,
    columns: fieldnames,
  });

  writeFileSync(outputPath, csvContent, "utf-8");

  console.log(
    `Filtered CSV written: ${acceptedCount.toLocaleString()}/${totalCount.toLocaleString()} measurements accepted (${((acceptedCount / totalCount) * 100).toFixed(1)}%)`
  );

  return acceptedCount;
}

/**
 * Main CLI function
 */
async function main(): Promise<number> {
  // Parse command-line arguments
  const { values } = parseArgs({
    options: {
      "csv-file": {
        type: "string",
        default: "data/2025-10-22_weights_all.csv",
      },
      "max-users": {
        type: "string",
        default: "0",
      },
      "max-rows": {
        type: "string",
        default: "0",
      },
      "min-readings": {
        type: "string",
        default: "20",
      },
      "user-ids": {
        type: "string",
      },
      "output-dir": {
        type: "string",
        default: "output_local",
      },
      "filtered-csv": {
        type: "string",
      },
      config: {
        type: "string",
      },
    },
  });

  const csvFile = values["csv-file"]!;
  const maxUsers = parseInt(values["max-users"]!, 10);
  const maxRows = parseInt(values["max-rows"]!, 10);
  const minReadings = parseInt(values["min-readings"]!, 10);
  const userIds = values["user-ids"];
  const outputDir = values["output-dir"]!;
  const filteredCsv = values["filtered-csv"];
  const configPath = values.config;

  // Validate inputs
  if (!existsSync(csvFile)) {
    console.error(`Error: CSV file not found: ${csvFile}`);
    return 1;
  }

  // Create output directory
  if (!existsSync(outputDir)) {
    mkdirSync(outputDir, { recursive: true });
  }

  // Initialize in-memory storage
  console.log("Initializing in-memory storage...");
  const stateStore = new ProcessorStateDB();

  // Load configuration
  console.log("Loading configuration...");
  let config: any;
  if (configPath) {
    config = ConfigManager.loadConfig(configPath);
    // Override database backend for local in-memory processing
    if (config.database) {
      config.database.backend = "memory";
    }
    console.log(`  Using config from: ${configPath}`);
  } else {
    // Use default config
    config = getDefaultConfig();
    console.log("  Using default configuration");
  }

  // Initialize service with in-memory storage
  console.log("Initializing weight processor service...");
  const service = new WeightProcessorService(stateStore, config);

  // Load CSV data
  let { userMeasurements, originalRows } = loadCsvData(csvFile, {
    maxUsers,
    maxRows,
    minReadings,
  });

  if (userMeasurements.size === 0) {
    console.log("No valid measurements found in CSV file");
    return 1;
  }

  // Filter by specific user IDs if provided
  if (userIds) {
    const requestedUserIds = userIds.split(",").map((uid) => uid.trim());
    console.log(
      `\nFiltering to ${requestedUserIds.length} specific user ID(s)...`
    );

    // Track which users were found and not found
    const foundUsers: string[] = [];
    const notFoundUsers: string[] = [];

    for (const userId of requestedUserIds) {
      if (userMeasurements.has(userId)) {
        foundUsers.push(userId);
      } else {
        notFoundUsers.push(userId);
      }
    }

    // Report not found users
    if (notFoundUsers.length > 0) {
      console.log(
        `\n⚠️  Warning: ${notFoundUsers.length} requested user ID(s) not found in CSV:`
      );
      for (const userId of notFoundUsers) {
        console.log(`  - ${userId}`);
      }
    }

    // Filter to only found users
    if (foundUsers.length > 0) {
      const filteredUserMeasurements = new Map<string, Measurement[]>();
      for (const uid of foundUsers) {
        filteredUserMeasurements.set(uid, userMeasurements.get(uid)!);
      }
      userMeasurements = filteredUserMeasurements;

      // Filter original_rows to match selected users
      const selectedUserSet = new Set(foundUsers);
      originalRows = originalRows.filter((row) =>
        selectedUserSet.has(row.user_id)
      );

      console.log(
        `\n✓ Processing ${foundUsers.length} user(s) with ${Array.from(userMeasurements.values())
          .reduce((sum, m) => sum + m.length, 0)
          .toLocaleString()} measurements`
      );
    } else {
      console.log(
        "\nError: None of the requested user IDs were found in the CSV file"
      );
      return 1;
    }
  }

  // Initialize acceptance tracker
  const acceptanceTracker = new AcceptanceTracker();

  // Track overall results
  const startTime = new Date();
  const overallResults: any = {
    startTime: startTime.toISOString(),
    csvFile: csvFile,
    storageType: "in-memory",
    usersLoaded: userMeasurements.size,
    totalMeasurements: Array.from(userMeasurements.values()).reduce(
      (sum, m) => sum + m.length,
      0
    ),
    processingResults: null,
    replayMode: "manual",
  };

  // Process measurements (manual replay only)
  console.log("\n=== Processing Measurements (Manual Replay) ===");
  console.log(
    "Note: Use manual replay endpoints for historical conflict resolution"
  );

  // Process all measurements first
  const processingResults: Record<string, any> = {};
  const totalUsers = userMeasurements.size;
  const totalMeasurements = Array.from(userMeasurements.values()).reduce(
    (sum, m) => sum + m.length,
    0
  );

  console.log(`\nProcessing ${totalUsers.toLocaleString()} users...`);
  console.log(`Total measurements: ${totalMeasurements.toLocaleString()}`);

  let processedMeasurements = 0;
  let successfulUsers = 0;
  let failedUsers = 0;

  let i = 1;
  for (const [userId, measurements] of userMeasurements.entries()) {
    console.log(
      `[${i}/${totalUsers}] Processing user ${userId.substring(0, 12)}... (${measurements.length} measurements)`
    );

    const userResults: any = {
      measurementsProcessed: 0,
      measurementsAccepted: 0,
      measurementsRejected: 0,
      errors: [],
      results: [],
    };

    try {
      // Sort measurements chronologically
      const sortedMeasurements = [...measurements].sort((a, b) => {
        const ts_a = typeof a.timestamp === 'string' ? new Date(a.timestamp) : a.timestamp;
        const ts_b = typeof b.timestamp === 'string' ? new Date(b.timestamp) : b.timestamp;
        return ts_a.getTime() - ts_b.getTime();
      });

      // Process batch
      const responseData = await service.processBatch(userId, sortedMeasurements);

      userResults.measurementsProcessed = responseData.measurements_processed;
      userResults.measurementsAccepted = responseData.measurements_accepted;
      userResults.measurementsRejected = responseData.measurements_rejected;
      userResults.results = responseData.results;

      // Track acceptance - convert BatchProcessResult to ProcessResponseData format
      const processResponseData: ProcessResponseData = {
        userId: userId,
        measurementsProcessed: responseData.measurements_processed,
        measurementsAccepted: responseData.measurements_accepted,
        measurementsRejected: responseData.measurements_rejected,
        results: responseData.results
      };
      acceptanceTracker.markBatchResults(
        userId,
        sortedMeasurements,
        processResponseData
      );

      processedMeasurements += responseData.measurements_processed;
      successfulUsers++;

      console.log(
        `  ✓ Processed: ${responseData.measurements_processed}, Accepted: ${responseData.measurements_accepted}, Rejected: ${responseData.measurements_rejected}`
      );
    } catch (error: any) {
      failedUsers++;
      userResults.errors.push(error.message);
      console.error(`  ✗ Error processing user: ${error.message}`);
    }

    processingResults[userId] = userResults;
    i++;
  }

  // Summary
  console.log("\n=== Processing Summary ===");
  console.log(`Total users processed: ${successfulUsers}/${totalUsers}`);
  console.log(`Failed users: ${failedUsers}`);
  console.log(
    `Total measurements processed: ${processedMeasurements.toLocaleString()}`
  );

  const totalAccepted = Object.values(processingResults).reduce(
    (sum: number, r: any) => sum + r.measurementsAccepted,
    0
  );
  const totalRejected = Object.values(processingResults).reduce(
    (sum: number, r: any) => sum + r.measurementsRejected,
    0
  );

  console.log(`Total accepted: ${totalAccepted.toLocaleString()}`);
  console.log(`Total rejected: ${totalRejected.toLocaleString()}`);
  console.log(
    `Acceptance rate: ${((totalAccepted / processedMeasurements) * 100).toFixed(1)}%`
  );

  // Write filtered CSV
  const timestamp = new Date()
    .toISOString()
    .replace(/[:.]/g, "-")
    .substring(0, 19);
  const outputCsvPath =
    filteredCsv || join(outputDir, `filtered_${timestamp}.csv`);
  const acceptedCount = writeFilteredCsv(
    originalRows,
    acceptanceTracker,
    outputCsvPath
  );

  // Write results JSON
  const resultsPath = join(outputDir, `results_${timestamp}.json`);
  overallResults.processingResults = processingResults;
  overallResults.endTime = new Date().toISOString();
  overallResults.filteredCsvPath = outputCsvPath;
  overallResults.acceptedCount = acceptedCount;

  writeFileSync(resultsPath, JSON.stringify(overallResults, null, 2), "utf-8");
  console.log(`\nResults saved to: ${resultsPath}`);

  console.log("\n=== Processing Complete ===");
  return 0;
}

// Run main function
main()
  .then((exitCode) => {
    process.exit(exitCode);
  })
  .catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
  });

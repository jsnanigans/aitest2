#!/usr/bin/env bun
/**
 * Local Weight Stream Processor (TypeScript)
 *
 * Processes weight measurements from CSV data using typescript_lib core library.
 * Uses in-memory storage for state management.
 * Outputs a filtered CSV with only accepted (non-rejected) measurements.
 *
 * This version uses ONLY typescript_lib (core library), not weight-processor-ts.
 */

import { parseArgs } from "node:util";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { parse } from "csv-parse/sync";
import { stringify } from "csv-stringify/sync";
import { join } from "node:path";

// Import from typescript_lib (core library)
import {
  InMemoryStore,
  SUPPORTED_WEIGHT_UNITS,
  type ProcessingResult,
} from "./typescript_lib/src/index";

// Import service layer
import {
  WeightProcessorService,
  type ProcessResponseData,
  type MeasurementInput,
} from "./services/weight_processor_service";

/**
 * Default configuration matching Python implementation
 */
function getDefaultConfig(): any {
  return {
    database: {
      backend: "memory",
    },
    kalman: {
      initial_variance: 0.364,
      transition_covariance_weight: 0.018,
      transition_covariance_trend: 0.00015,
      observation_covariance: 3.49,
    },
    quality_scoring: {
      threshold: 0.5,
      components: {
        kalman_fit: { weight: 0.3, enabled: true },
        temporal_consistency: { weight: 0.25, enabled: true },
        anomaly_detection: { weight: 0.25, enabled: true },
        source_reliability: { weight: 0.1, enabled: true },
        trend_alignment: { weight: 0.1, enabled: true },
      },
    },
    processing: {
      enable_validation: true,
      enable_quality_scoring: true,
    },
    reset: {
      time_gap_days: 30,
      weight_change_threshold_kg: 10,
    },
    snapshot: {
      interval_hours: 24,
      periodic_enabled: true,
    },
    adaptive_noise: {
      enabled: true,
    },
    replay: {
      buffered_replay_enabled: true,
      buffer_hours: 24,
      max_buffer_measurements: 100,
    },
  };
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
      return new Date(dateStr.replace(" ", "T") + "Z");
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

    // Convert to MeasurementInput for processing
    try {
      const measurement: MeasurementInput = {
        measurementId,
        weight,
        unit,
        timestamp,
        source,
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

  // Filter users by minimum readings BEFORE applying max_users limit
  if (options.minReadings > 0) {
    const usersBeforeFilter = userMeasurements.size;
    const measurementsBeforeFilter = Array.from(
      userMeasurements.values()
    ).reduce((sum, m) => sum + m.length, 0);

    // Filter out users with fewer than min_readings
    const filteredUserMeasurements = new Map<string, MeasurementInput[]>();
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
    for (const row of filteredRows) {
      originalRows.push(row);
    }

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
    const limitedUserMeasurements = new Map<string, MeasurementInput[]>();
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

  clear(): void {
    this.acceptedMeasurements.clear();
  }

  markMeasurementAccepted(userId: string, timestamp: string): void {
    this.acceptedMeasurements.add(`${userId}|${timestamp}`);
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
        const timestamp = measurements[i].timestamp.toISOString();
        this.markMeasurementAccepted(userId, timestamp);
      }
    }
  }

  isAccepted(userId: string, timestamp: string): boolean {
    return this.acceptedMeasurements.has(`${userId}|${timestamp}`);
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
    },
  });

  const csvFile = values["csv-file"]!;
  const maxUsers = parseInt(values["max-users"]!, 10);
  const maxRows = parseInt(values["max-rows"]!, 10);
  const minReadings = parseInt(values["min-readings"]!, 10);
  const userIds = values["user-ids"];
  const outputDir = values["output-dir"]!;
  const filteredCsv = values["filtered-csv"];

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
  const stateStore = new InMemoryStore();

  // Load configuration
  console.log("Loading configuration...");
  const config = getDefaultConfig();
  console.log("  Using default configuration");

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
      const filteredUserMeasurements = new Map<string, MeasurementInput[]>();
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

  // Initialize service
  console.log("Initializing weight processor service...");
  const service = new WeightProcessorService(stateStore, config);

  // Initialize acceptance tracker
  const acceptanceTracker = new AcceptanceTracker();

  // Track overall results
  const startTime = new Date();

  // Process measurements with automatic buffered replay
  console.log("\n=== Processing Measurements (Automatic Buffered Replay) ===");
  console.log(
    "Note: Replay triggers automatically at end of batch or when time window/buffer exceeded"
  );

  const totalUsers = userMeasurements.size;
  const totalMeasurements = Array.from(userMeasurements.values()).reduce(
    (sum, m) => sum + m.length,
    0
  );

  console.log(`\nProcessing ${totalUsers.toLocaleString()} users...`);
  console.log(`Total measurements: ${totalMeasurements.toLocaleString()}`);

  let processedMeasurements = 0;
  let acceptedMeasurements = 0;
  let rejectedMeasurements = 0;
  let successfulUsers = 0;
  let failedUsers = 0;
  const processingResults: Record<string, any> = {};

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
      replayMetadata: [],
    };

    try {
      // Sort measurements chronologically
      const sortedMeasurements = [...measurements].sort((a, b) => {
        return a.timestamp.getTime() - b.timestamp.getTime();
      });

      // Process batch using service (includes automatic buffered replay)
      const response: ProcessResponseData = await service.processBatch(
        userId,
        sortedMeasurements
      );

      userResults.measurementsProcessed = response.measurements_processed;
      userResults.measurementsAccepted = response.measurements_accepted;
      userResults.measurementsRejected = response.measurements_rejected;

      // Capture replay metadata if present
      if (response.replay_metadata && response.replay_metadata.length > 0) {
        userResults.replayMetadata = response.replay_metadata;
        console.log(
          `  🔄 Replay triggered ${response.replay_metadata.length} time(s)`
        );
        for (const replay of response.replay_metadata) {
          console.log(
            `    - Trigger: ${replay.trigger}, ` +
              `Buffer size: ${replay.buffer_size}, ` +
              `From: ${replay.replay_from} to ${replay.replay_to}`
          );
        }
      }

      processedMeasurements += response.measurements_processed;
      acceptedMeasurements += response.measurements_accepted;
      rejectedMeasurements += response.measurements_rejected;

      // Track acceptance
      acceptanceTracker.markBatchResults(userId, sortedMeasurements, response);

      successfulUsers++;
      console.log(
        `  ✓ Processed: ${response.measurements_processed}, Accepted: ${response.measurements_accepted}, Rejected: ${response.measurements_rejected}`
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
  console.log(`Total accepted: ${acceptedMeasurements.toLocaleString()}`);
  console.log(`Total rejected: ${rejectedMeasurements.toLocaleString()}`);
  if (processedMeasurements > 0) {
    console.log(
      `Acceptance rate: ${((acceptedMeasurements / processedMeasurements) * 100).toFixed(1)}%`
    );
  }

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
  const endTime = new Date();
  const overallResults = {
    startTime: startTime.toISOString(),
    endTime: endTime.toISOString(),
    durationSeconds: (endTime.getTime() - startTime.getTime()) / 1000,
    csvFile,
    storageType: "in-memory",
    usersLoaded: totalUsers,
    totalMeasurements: processedMeasurements,
    acceptedCount,
    rejectedCount: rejectedMeasurements,
    filteredCsvPath: outputCsvPath,
    replayMode: "automatic",
    processingResults,
  };

  writeFileSync(resultsPath, JSON.stringify(overallResults, null, 2), "utf-8");
  console.log(`\nResults saved to: ${resultsPath}`);
  console.log(
    `Duration: ${overallResults.durationSeconds.toFixed(1)} seconds`
  );

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

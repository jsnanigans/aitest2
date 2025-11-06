# Specifications: Matching Detailed Logging for Python and TypeScript Weight Processors

**Date:** 2025-11-05
**Feature:** Add detailed matching logging to both Python and TypeScript weight processor implementations to verify 1:1 behavioral equivalence

---

## 1. Overview

Add comprehensive, matching logging to both the Python (`weight_values/`) and TypeScript (`weight-processor-ts/`) weight processor implementations. The logging must be detailed enough to:
- Compare implementations step-by-step to verify 1:1 behavioral match
- Identify specific points where implementations diverge
- Debug processing logic issues
- Verify output CSV files match exactly

## 2. Goals and Objectives

### Primary Goals
1. Add detailed logging to every processing step in both implementations
2. Ensure logs from both implementations contain the same information
3. Run both implementations with `test_user.csv` and verify:
   - Log content matches (same information, not necessarily identical format)
   - Output CSV files match exactly (byte-for-byte)

### Success Criteria
- ✅ Both implementations log every major processing step with consistent markers
- ✅ Numeric values logged to 6 decimal places
- ✅ Logs include enough detail to identify where divergence occurs
- ✅ Output CSVs from both implementations are identical
- ✅ Logging is configurable (can be enabled/disabled)

## 3. Functional Requirements

### 3.1 Logging Scope

Log the following processing steps in both implementations:

#### Input Processing
- User ID (first 12 characters for brevity)
- Raw weight value and unit
- Timestamp (ISO format)
- Source

#### Step 1: Data Cleaning and Preprocessing
- Cleaned weight value
- Any preprocessing flags or warnings
- Rejection reason if preprocessing fails

#### Step 2: State Management
- Whether state exists or is being created
- Last weight, timestamp, and source from existing state
- Whether Kalman filter is already initialized

#### Step 3: Reset Detection
- Whether reset check is performed
- Reset type if reset is triggered
- Reset reason and gap days
- Confirmation of reset completion

#### Step 4: Kalman Initialization (if needed)
- Adaptive Kalman configuration parameters
- Noise multiplier for source
- Observation covariance
- Initial state vector values

#### Step 5: Quality Scoring
- Kalman prediction value
- Innovation covariance
- Previous weight and time difference
- Quality score (overall)
- Quality score components (breakdown)
- Rejection reason if quality check fails

#### Step 6: Kalman Update (if not already done)
- Adaptive parameter usage
- Noise multiplier
- Observation covariance
- Updated state vector values
- Trend limiting (before and after)

#### Final Result
- Acceptance status
- Kalman estimate
- Kalman uncertainty
- Quality score
- Processing stage
- Any reset event information

### 3.2 Logging Format

#### Text Format Requirements
- Human-readable text output
- Consistent step markers: `Step 1:`, `Step 2:`, etc.
- Implementation prefix: `[PY]` for Python, `[TS]` for TypeScript
- Numeric precision: 6 decimal places for floats
- Visual separators: `===` lines between measurements

#### Example Log Entry Format
```
================================================================================
[PY] Processing measurement for user ADC64C0B-CB46...
[PY]   Weight: 104.326160 kg
[PY]   Timestamp: 2025-01-14T23:33:34.522Z
[PY]   Source: https://api.iglucose.com
[PY] Step 1: Data cleaning and preprocessing
[PY]   Cleaned weight: 104.326160
[PY] Step 2: Load or create user state
[PY]   Creating initial state (first measurement)
[PY] Step 3: Check for reset
[PY]   No reset needed
[PY] Step 4: Initialize Kalman if needed
[PY]   Initializing Kalman filter
[PY]   Observation covariance: 5.235000
[PY]   Initial state: [104.326160, 0.000000]
[PY] Step 5: Quality scoring
[PY]   Quality score: 0.850000
[PY]   Kalman prediction: None
[PY] Result: ACCEPTED
[PY]   Kalman estimate: 104.326160
[PY]   Kalman uncertainty: 1.805000
================================================================================
```

### 3.3 Output CSV Requirements

- Output CSV must match exactly between implementations
- Same number of rows
- Same columns in same order
- Same values (no floating-point differences)
- Same measurement IDs for accepted measurements

### 3.4 Test Requirements

#### Test Input
- Primary test file: `test_user.csv`
- Contains: 122 rows (121 measurements + 1 header)
- Single user: `ADC64C0B-CB46-41F9-BDA0-CC11A35942D7`
- Date range: January 2025 - September 2025
- Mix of normal and outlier measurements

#### Test Commands
Python:
```bash
uv run python local_main.py \
  --csv-file test_user.csv \
  --user-ids ADC64C0B-CB46-41F9-BDA0-CC11A35942D7 \
  --min-readings 0 \
  --output-dir output_local \
  --filtered-csv filtered_weights_py.csv
```

TypeScript:
```bash
bun run local_main.ts \
  --csv-file test_user.csv \
  --user-ids ADC64C0B-CB46-41F9-BDA0-CC11A35942D7 \
  --min-readings 0 \
  --output-dir output_local \
  --filtered-csv filtered_weights_ts.csv
```

#### Verification
1. Capture stdout logs from both runs
2. Compare log content (same steps, values within precision)
3. Compare output CSV files byte-by-byte
4. Report any differences with specific line/step numbers

## 4. Non-Functional Requirements

### 4.1 Configurability
- Logging should be configurable via command-line flag or environment variable
- Default: logging enabled for development/testing
- Can be disabled for production performance

### 4.2 Performance
- Logging should not significantly impact processing time (<10% overhead)
- Acceptable for test/debug purposes, not production

### 4.3 Maintainability
- Logging code should be isolated in dedicated functions/methods
- Easy to update log formats without changing core logic
- Same log structure in both implementations for easy comparison

## 5. Constraints and Assumptions

### Constraints
- Must not modify core processing logic
- Must not change output CSV format
- Must work with existing configuration and command-line arguments
- Logs go to stdout (not files) for easy capture and comparison

### Assumptions
- Both implementations are functionally equivalent (this logging will verify)
- Test data (test_user.csv) is valid and complete
- Both implementations use same configuration (config.toml)
- Floating-point calculations may have minor differences (<0.000001)

## 6. Out of Scope

The following are explicitly OUT OF SCOPE for this feature:

- ❌ Automated test harness (will be manual comparison)
- ❌ Performance optimization of logging
- ❌ Logging to files or structured log formats (JSON, etc.)
- ❌ Log rotation or management
- ❌ Testing with datasets other than test_user.csv (in this phase)
- ❌ Fixing any discovered implementation differences (identify only)

## 7. Dependencies

### Technical Dependencies
- Python: `local_main.py` and `weight_values/src/core/processing/processor.py`
- TypeScript: `local_main.ts` and `weight-processor-ts/src/core/processing/processor.ts`
- Test data: `test_user.csv`
- Configuration: `config.toml` (shared or separate per implementation)

### Process Dependencies
- Both implementations must be runnable in current state
- Test CSV must be accessible to both implementations
- Output directory must be writable

## 8. Risk Assessment

### High Risk
- **Logging changes break processing logic**: Mitigate by isolating logging code
- **Implementations already diverge**: Logging will help identify this

### Medium Risk
- **Floating-point precision differences**: Accept minor differences within tolerance
- **Timestamp formatting differences**: Normalize to ISO format in logs

### Low Risk
- **Performance impact**: Acceptable for test/debug purposes
- **Log verbosity**: Can be adjusted based on findings

## 9. Acceptance Criteria

This feature is complete when:

1. ✅ Python implementation has detailed logging at every processing step
2. ✅ TypeScript implementation has matching detailed logging
3. ✅ Logs from both implementations contain the same information
4. ✅ Both implementations can be run with test_user.csv
5. ✅ Output CSVs from both implementations are identical
6. ✅ Logging can be enabled/disabled via flag or config
7. ✅ Documentation updated with logging examples

## 10. Future Considerations

After initial implementation and verification:
- Automated comparison scripts
- Testing with additional CSV files
- Fixing identified implementation differences
- Performance profiling with logging enabled
- Structured logging for automated analysis

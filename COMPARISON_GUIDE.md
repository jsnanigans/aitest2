# TypeScript vs Python Comparison Guide

## Quick Start

Run the comprehensive comparison:

```bash
./compare_ts_py.sh
```

This will:
1. Run both TypeScript and Python implementations with `test_user.csv`
2. Compare outputs in detail
3. Report any differences

## What Gets Compared

### 1. Execution Time
- Duration for each implementation
- Performance comparison

### 2. Output Files
- **Filtered CSV**: Contains only accepted measurements
- **Results JSON**: Processing statistics and metadata

### 3. Processing Statistics
- Total measurements processed
- Measurements accepted
- Measurements rejected
- Acceptance rate

### 4. Replay Metadata
- Number of replay triggers
- Trigger types (time_gap, batch_end, buffer_overflow)
- Buffer sizes
- Time ranges

### 5. Measurement-Level Comparison
- Exact measurement IDs that were accepted
- Checks if both implementations accept/reject the same measurements
- Validates that the filtered CSVs are identical

### 6. Sample Values
- Compares actual weight values for accepted measurements
- Ensures numerical consistency

## Output Structure

```
output_comparison_ts/
├── filtered_TIMESTAMP.csv    # Accepted measurements (TS)
├── results_TIMESTAMP.json     # Processing stats (TS)
└── console.log                # Console output (TS)

output_comparison_py/
├── filtered_TIMESTAMP.csv    # Accepted measurements (PY)
├── local_processing_results_TIMESTAMP.json  # Processing stats (PY)
└── console.log                # Console output (PY)
```

## Understanding Results

### ✓ PASS (Green)
Both implementations produce identical results:
- Same number of accepted/rejected measurements
- Same measurement IDs in filtered CSVs
- Identical processing statistics

### ✗ FAIL (Red)
Differences detected:
- Review console logs to see which measurements differ
- Check replay metadata to understand triggering differences
- Compare JSON files for detailed statistics

## Common Differences (Expected)

Some differences are expected and don't indicate bugs:

1. **Replay Timing**: The exact number of replay triggers may vary slightly due to:
   - Buffer timing edge cases
   - Snapshot creation timing

2. **Execution Time**: Performance may vary based on:
   - System load
   - JIT compilation (Bun vs Python)
   - Memory allocation patterns

## Manual Comparison

If you need to compare specific aspects manually:

### Compare CSV Files
```bash
diff output_comparison_ts/filtered_*.csv output_comparison_py/filtered_*.csv
```

### Compare JSON Statistics
```bash
# View TypeScript results
jq . output_comparison_ts/results_*.json

# View Python results
jq . output_comparison_py/local_processing_results_*.json
```

### Count Accepted Measurements
```bash
# TypeScript
tail -n +2 output_comparison_ts/filtered_*.csv | wc -l

# Python
tail -n +2 output_comparison_py/filtered_*.csv | wc -l
```

### View Console Outputs
```bash
# TypeScript console output
cat output_comparison_ts/console.log

# Python console output
cat output_comparison_py/console.log
```

## Test Data

The comparison uses `test_user.csv` which contains:
- Single user: `ADC64C0B-CB46-41F9-BDA0-CC11A35942D7`
- 121 measurements
- Date range: January 2025 - September 2025
- Various weight values testing edge cases:
  - Normal ranges (58-60 kg)
  - Outliers (43 kg, 115 kg)
  - Rapid changes (for replay testing)
  - Unit conversions

## Troubleshooting

### Script Fails to Run

1. **Check dependencies**:
   ```bash
   # Ensure Bun is installed
   bun --version

   # Ensure Python is installed
   python --version

   # Ensure jq is installed (for JSON parsing)
   jq --version
   ```

2. **Check file paths**:
   ```bash
   # Verify test data exists
   ls -lh test_user.csv

   # Verify main scripts exist
   ls -lh local_main.ts local_main.py
   ```

### Unexpected Differences

1. **Check replay configuration**: Ensure both have identical config:
   ```typescript
   // TypeScript (local_main.ts)
   replay: {
     buffered_replay_enabled: true,
     buffer_hours: 24,
     max_buffer_measurements: 100,
   }
   ```

   ```python
   # Python (local_main.py - via config.toml or defaults)
   [replay]
   buffered_replay_enabled = true
   buffer_hours = 24
   max_buffer_measurements = 100
   ```

2. **Check Kalman parameters**: Ensure identical:
   ```typescript
   kalman: {
     initial_variance: 0.364,
     transition_covariance_weight: 0.018,
     transition_covariance_trend: 0.00015,
     observation_covariance: 3.49,
   }
   ```

3. **Verify core library versions**: Make sure both are using the latest code

## Advanced Comparison

For deeper analysis, you can run both versions with verbose logging:

```bash
# TypeScript with verbose output
VERBOSE_LOGGING=true bun run local_main.ts \
  --csv-file test_user.csv \
  --min-readings 0 \
  --output-dir output_debug_ts

# Python with verbose output
VERBOSE_LOGGING=true python local_main.py \
  --csv-file test_user.csv \
  --min-readings 0 \
  --output-dir output_debug_py
```

## CI/CD Integration

To integrate this comparison into CI/CD:

```bash
#!/bin/bash
set -e

# Run comparison
./compare_ts_py.sh

# Exit code 0 = PASS, non-zero = FAIL
if [ $? -eq 0 ]; then
    echo "✓ Implementations match"
    exit 0
else
    echo "✗ Implementations differ"
    exit 1
fi
```

## Expected Output Example

```
╔════════════════════════════════════════════════════════════╗
║  TypeScript vs Python Implementation Comparison           ║
╚════════════════════════════════════════════════════════════╝

Test Configuration:
  CSV File: test_user.csv
  User ID: ADC64C0B-CB46-41F9-BDA0-CC11A35942D7

═══════════════════════════════════════════════════════════
Running TypeScript Implementation
═══════════════════════════════════════════════════════════

[Processing output...]
✓ TypeScript version completed successfully

═══════════════════════════════════════════════════════════
Running Python Implementation
═══════════════════════════════════════════════════════════

[Processing output...]
✓ Python version completed successfully

═══════════════════════════════════════════════════════════
Comparison Results
═══════════════════════════════════════════════════════════

1. Execution Time:
   TypeScript: 2s
   Python:     3s
   TypeScript is 1.50x faster

2. CSV Output:
   TypeScript rows: 85
   Python rows:     85
   ✓ Row counts match

3. Processing Statistics:
   Measurements Processed:
     TypeScript: 121
     Python:     121
     ✓ Match

   Measurements Accepted:
     TypeScript: 85
     Python:     85
     ✓ Match

   Measurements Rejected:
     TypeScript: 36
     Python:     36
     ✓ Match

4. Replay Metadata:
   TypeScript replay triggers: 1
   Python replay triggers:     1
   ✓ Replay counts match

5. Detailed Measurement Comparison:
   Extracting measurement IDs from CSV files...
   ✓ All accepted measurements match exactly

6. Sample Weight Values:
   Comparing first 5 accepted measurements...
   [Weight values shown...]

═══════════════════════════════════════════════════════════
Summary
═══════════════════════════════════════════════════════════

✓ PASS: Both implementations produce identical results!
```

## Contact

For issues or questions about the comparison:
- Check console logs in `output_comparison_*/console.log`
- Review JSON results for detailed statistics
- Compare individual measurements in filtered CSV files

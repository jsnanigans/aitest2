# Run Both Processors with Logging

This script runs both Python and TypeScript weight processors with verbose logging enabled and provides a comparison summary.

## Usage

```bash
./run_both_with_logs.sh [CSV_FILE]
```

**Default:** If no CSV file is specified, it uses `test_user.csv`

**Examples:**
```bash
# Use default test_user.csv
./run_both_with_logs.sh

# Use a specific CSV file
./run_both_with_logs.sh test_small.csv
```

## What It Does

1. Runs Python processor with `VERBOSE_LOGGING=true`
2. Runs TypeScript processor with `VERBOSE_LOGGING=true`
3. Captures logs to `logs_py.txt` and `logs_ts.txt`
4. Generates filtered CSVs: `filtered_weights_py.csv` and `filtered_weights_ts.csv`
5. Shows a summary comparing:
   - Number of log lines
   - Number of accepted measurements
   - Divergence (if any)

## Output Files

- `logs_py.txt` - Complete Python processing logs with [PY] prefix
- `logs_ts.txt` - Complete TypeScript processing logs with [TS] prefix
- `filtered_weights_py.csv` - Python output (accepted measurements only)
- `filtered_weights_ts.csv` - TypeScript output (accepted measurements only)

## Example Output

```
=========================================
Running Weight Processors with Logging
=========================================
Input: test_user.csv
User ID: ADC64C0B-CB46-41F9-BDA0-CC11A35942D7

🐍 Running Python processor...
   ✓ Python complete - logs saved to logs_py.txt
📘 Running TypeScript processor...
   ✓ TypeScript complete - logs saved to logs_ts.txt

=========================================
Summary
=========================================
Python logs:     2290 lines
TypeScript logs: 2880 lines

Python accepted:     41 measurements
TypeScript accepted: 116 measurements

❌ DIVERGENCE: Difference of 75 measurements
   (TypeScript accepted 75 more)

Files generated:
  - logs_py.txt (Python logs)
  - logs_ts.txt (TypeScript logs)
  - filtered_weights_py.csv (Python output)
  - filtered_weights_ts.csv (TypeScript output)
```

## Analyzing the Logs

The logs contain detailed step-by-step processing information:

```bash
# View Python logs
cat logs_py.txt | grep "\[PY\]"

# View TypeScript logs
cat logs_ts.txt | grep "\[TS\]"

# Compare specific measurement processing
grep "Processing measurement" logs_py.txt | head -10
grep "Processing measurement" logs_ts.txt | head -10
```

## Configuration

The script is configured for:
- User ID: `ADC64C0B-CB46-41F9-BDA0-CC11A35942D7`
- Output directory: `output_local/`
- Minimum readings: 0

To modify these, edit the script's configuration section.

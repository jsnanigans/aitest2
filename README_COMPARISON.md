# TypeScript vs Python Comparison Tools

## Quick Start

Run the comparison with one command:

```bash
./run_comparison.sh
```

This script:
1. Runs both TypeScript (`local_main.ts`) and Python (`local_main.py`) with `test_user.csv`
2. Compares the filtered CSV outputs
3. Reports PASS/FAIL based on exact measurement matching

## Files

### Scripts
- **`run_comparison.sh`** - Quick comparison script (recommended)
- **`compare_ts_py.sh`** - Comprehensive comparison with detailed statistics
- **`compare_quick.sh`** - Simplified version

### Documentation
- **`TS_PY_COMPARISON.md`** - Detailed comparison of implementations
- **`COMPARISON_GUIDE.md`** - Complete usage guide

### Test Data
- **`test_user.csv`** - Single user test data (121 measurements)

## Expected Output

```
=== TypeScript vs Python Comparison ===

Cleaning up old outputs...

Running TypeScript...
[Processing output...]

Running Python...
[Processing output...]

=== Comparison ===
CSV Rows:
  TypeScript: 85
  Python:     85
  ✓ MATCH
✓ PASS: All accepted measurements match exactly!
```

## What Gets Tested

1. **CSV Row Counts**: Both implementations accept the same number of measurements
2. **Measurement IDs**: The exact same measurements are accepted by both
3. **Processing Logic**: Kalman filtering, quality scoring, and replay behave identically

## Output Directories

After running, you'll find:

```
output_comparison_ts/
├── filtered_*.csv         # Accepted measurements (TypeScript)
├── results_*.json         # Processing statistics
└── console.log            # Full console output

output_comparison_py/
├── filtered_*.csv         # Accepted measurements (Python)
├── local_processing_results_*.json  # Processing statistics
└── console.log            # Full console output
```

## Troubleshooting

### Test file not found
```bash
# Check if test_user.csv exists
ls -lh test_user.csv
```

### Dependencies missing
```bash
# Check Bun
bun --version

# Check Python
python --version
```

### Scripts won't run
```bash
# Make executable
chmod +x run_comparison.sh

# Check syntax
bash -n run_comparison.sh
```

## Manual Comparison

If you need to compare specific aspects:

```bash
# View TypeScript output
cat output_comparison_ts/console.log

# View Python output
cat output_comparison_py/console.log

# Compare CSV files
diff output_comparison_ts/filtered_*.csv output_comparison_py/filtered_*.csv

# Compare JSON results
jq . output_comparison_ts/results_*.json
jq . output_comparison_py/local_processing_results_*.json
```

## Test Data Details

The `test_user.csv` contains measurements designed to test:
- Normal weight ranges (58-60 kg)
- Outliers that should be rejected (e.g., 31.2 kg, 117 kg)
- Rapid weight changes (for replay triggering)
- Time gaps (30+ day gaps)
- BMI-based validation
- Kalman filter adaptation

## CI/CD Integration

```yaml
# Example GitHub Actions workflow
- name: Run Comparison
  run: |
    ./run_comparison.sh
    if [ $? -ne 0 ]; then
      echo "❌ Implementation mismatch detected"
      exit 1
    fi
```

## Success Criteria

✅ **PASS** - Both implementations produce identical results:
- Same row count in filtered CSV
- Identical measurement IDs accepted
- Consistent processing statistics

❌ **FAIL** - Differences detected:
- Review console logs for details
- Check replay metadata
- Compare individual measurement results

## Performance

Typical execution time:
- TypeScript: 1-2 seconds
- Python: 2-3 seconds

Total comparison time: ~5-10 seconds

## Next Steps

1. Run `./run_comparison.sh` to verify implementations match
2. Review `COMPARISON_GUIDE.md` for detailed analysis
3. Check `TS_PY_COMPARISON.md` for implementation details
4. Examine output files for specific differences if tests fail

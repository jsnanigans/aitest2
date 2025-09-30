# Local Weight Processor (`local_main.py`)

Direct method-based weight measurement processor with in-memory storage. Alternative to `api_main.py` that bypasses the API layer for faster local processing.

## Features

- **Direct Service Calls**: Uses `WeightProcessorService` directly instead of HTTP API
- **In-Memory Storage**: Fast in-memory state management via `ProcessorStateDB`
- **Safety Parsing**: Comprehensive input validation (units, weights, BSA filtering)
- **Replay Support**: Optional full replay from beginning to recalculate with complete context
- **Data Quality Reports**: Detailed statistics on rejected measurements

## Processing Modes

### Mode 1: Individual + Replay with Outlier Detection (Default)

Processes measurements individually first, then uses windowed replay with outlier detection to correct Kalman state. This is the **default production mode** that matches `local_old.py` behavior.

```bash
./local_main.py
./local_main.py --max-users 100 --batch-size 50
```

**Use this mode to:**
- Detect and remove outlier measurements that distort Kalman state
- Correct processing history when bad measurements slip through
- Get production-quality processing with outlier correction
- Analyze data quality issues

### Mode 2: Individual Processing Only (Reference Dataset Mode)

Processes measurements one at a time WITHOUT replay, building up Kalman state incrementally. This matches how the reference dataset `2025-09-29_all_filtered_e.csv` was created.

```bash
./local_main.py --disable-replay
./local_main.py --max-users 100 --disable-replay
```

**Use this mode to:**
- Match reference dataset creation
- Get baseline acceptance without replay correction
- Benchmark acceptance rates
- Test without outlier detection

## Command-Line Options

```bash
./local_main.py [OPTIONS]

Data Options:
  --csv-file PATH         Input CSV file (default: data/2025-09-29_weights_all.csv)
  --max-users N           Maximum users to process (0 = unlimited)
  --max-rows N            Maximum CSV rows to read (0 = unlimited)
  --output-dir PATH       Output directory (default: output_local)
  --filtered-csv PATH     Output path for filtered CSV

Processing Options:
  --batch-size N          Measurements per service call (default: 1)
  --enable-replay         Enable full replay from beginning after individual processing
  --skip-replay           Explicitly skip replay (same as default)
  --config PATH           Custom config file (optional, uses lambda.env.template defaults)
```

## Configuration

Default configuration is based on `config/lambda.env.template`:

- **Kalman Filter**: Adaptive filtering with 0.1 process noise, 1.0 observation noise
- **Quality Scoring**: Multi-component quality assessment (kalman=0.25, temporal=0.20, source=0.20)
- **Quality Thresholds**: high=0.8, medium=0.5, outlier_override=0.85
- **Replay**: 72-hour buffer window, 10 minimum measurements

## Data Quality Validation

Automatically filters out:
- Invalid weights (≤0, >1000, NaN, Inf)
- Unsupported units (validates against whitelist)
- BSA measurements (Body Surface Area)
- Missing required data
- Parse errors

Reports detailed statistics:
```
Data Quality Statistics:
  Total rows read: 926,557
  Valid measurements: 37
  Rejected measurements: 3,838
    Parse errors: 6
    Invalid/unsupported units: 1,061
    BSA measurements (filtered): 2,771

  Top rejected units:
    '[lb_ap]': 987 measurements
    '{number}': 38 measurements
```

## Output Files

Generated in `output_local/` directory:

1. **`filtered_TIMESTAMP.csv`** - Accepted measurements only
   - Without replay: Contains Phase 1 (individual) results
   - With replay: Contains Phase 2 (replay) results

2. **`local_processing_results_TIMESTAMP.json`** - Complete processing statistics
   ```json
   {
     "start_time": "2025-09-30T15:40:51+00:00",
     "users_loaded": 10,
     "total_measurements": 46,
     "individual_processing": {...},
     "replay_processing": {...},
     "accepted_measurements": 33,
     "duration_seconds": 0.05
   }
   ```

## Examples

### Process 100 users with defaults
```bash
./local_main.py --max-users 100
```

### Process with replay enabled (E2E test mode)
```bash
./local_main.py --max-users 100 --enable-replay
```

### Fast processing with large batches
```bash
./local_main.py --batch-size 100 --output-dir output_fast
```

### Custom output location
```bash
./local_main.py --filtered-csv my_filtered_data.csv
```

### Use custom configuration
```bash
./local_main.py --config my_config.toml --enable-replay
```

## Comparison with `api_main.py`

| Feature | `local_main.py` | `api_main.py` |
|---------|----------------|---------------|
| Processing | Direct service calls | HTTP API calls |
| Storage | In-memory | DynamoDB |
| Speed | Faster (no network) | Slower (HTTP overhead) |
| State Persistence | No (session only) | Yes (DynamoDB) |
| Use Case | Local testing, batch processing | Production API testing |
| Replay | Full from beginning | Middle-point window |

## Performance

Typical throughput:
- Individual processing: ~10,000 measurements/sec
- With replay enabled: ~5,000 measurements/sec (due to reset and reprocessing)

## Notes

1. **In-Memory Only**: State is not persisted between runs
2. **Replay Eligibility**: Only users with ≥10 measurements are eligible for replay
3. **State Reset**: Replay mode clears user states before reprocessing
4. **Acceptance Tracking**: Tracker is cleared and repopulated during replay
5. **CSV Output**: Contains only measurements from eligible users when replay is enabled

## Troubleshooting

### No replay happening
- Check if users have ≥10 measurements
- Verify `--enable-replay` flag is set
- Check console output for "Processing full replay from beginning"

### Acceptance rate too high/low
- Verify configuration matches expected settings
- Check data quality statistics for rejected measurements
- Compare with/without replay to see effect of full context

### Memory issues with large datasets
- Use `--max-users` to limit processing
- Use `--max-rows` to limit CSV reading
- Process in multiple batches

## Understanding the Replay Mechanism

The replay feature solves a specific problem with streaming weight measurement processing:

### The Problem

When measurements arrive in real-time:
1. First measurement in a time window: **82 kg** (slightly off, but passes quality threshold)
   - Gets accepted, updates Kalman state to 82 kg
2. Second measurement moments later: **75 kg** (actually fits trend better)
   - Now rejected because Kalman state is at 82 kg, so 75 kg looks too different

Result: The "worse" measurement was accepted and the "better" one rejected, just due to timing!

### The Solution (Replay with Outlier Detection)

**Conceptual Flow:**

```
Measurements arrive: [... previous data ...] | [82kg, 75kg, 83kg, 76kg, 81kg] <- 72-hour window
                                                ^
                                          Buffer start (save state snapshot)
```

**Step 1: Save State Snapshot**
- Snapshot Kalman state before the buffer window started
- This represents "what we knew before these measurements"

**Step 2: Detect Outliers**
- Compare EACH measurement in buffer against the snapshot state
- Ask: "Does this measurement fit with what we knew before?"
- Use statistical methods: IQR, MAD, Kalman prediction deviation
- Example: 82kg might be detected as outlier if trend was 75-76kg

**Step 3: Selective Replay**
- Restore state to snapshot (forget all buffer measurements)
- Replay ONLY clean measurements: [75kg, 83kg, 76kg, 81kg]
- Process them chronologically as if 82kg never happened

**Result:** Kalman state is now based on the measurements that actually fit the trend!

### What Replay Does and Doesn't Do

**Replay DOES:**
- ✅ Detect outliers in measurement windows
- ✅ Correct Kalman filter state by removing bad data from history
- ✅ Improve prediction accuracy for future measurements
- ✅ Fix "order-dependent" acceptance problems

**Replay DOES NOT:**
- ❌ Change which measurements appear in filtered CSV
- ❌ Re-decide acceptance for past measurements
- ❌ Modify acceptance tracker
- ❌ Update measurement metadata or quality scores

**Why?** The filtered CSV represents the "live processing decision" - what was accepted when the measurement arrived. Replay is an internal correction mechanism for Kalman state, not a retrospective acceptance re-evaluation.

### Configuration

Replay behavior is controlled by several config parameters:

```python
"replay": {
    "buffer_hours": 72,              # Window size for analysis
    "min_measurements": 10,          # Minimum data for meaningful replay
    "outlier_methods": ["iqr", "mad"], # Detection methods
    "iqr_multiplier": 1.5,          # IQR sensitivity
    "mad_threshold": 3.0             # MAD sensitivity
}
```

### When to Use Replay

**Enable replay if:**
- Processing real user data with potential outliers
- Want to improve Kalman state accuracy
- Testing outlier detection capabilities
- Analyzing data quality issues

**Skip replay if:**
- Creating reference datasets for testing
- Benchmarking acceptance rates
- Data is already clean/validated
- Want to match exact historical processing
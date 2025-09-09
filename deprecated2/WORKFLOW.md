# CLI Workflow and Progress Indication

## Overview

The weight stream processor now features comprehensive progress indication for better user experience when running from the command line.

## Features

### 1. Progress Bar
- Shows percentage completion
- Real-time ETA calculation
- Color-coded progress (red → yellow → green)
- Current user being processed
- Live statistics update

Example:
```
████████████████████░░░░░░░░░░░░░░░░░░░ 50.0% | 50,000/100,000 | ETA: 1m 23s | User: abc123
```

### 2. Spinner Animation
- Used when total count is unknown
- Smooth animation frames
- Real-time statistics
- Row and user counts

Example:
```
⠼ Rows: 45,231 | Users: 523 | Time: 2m 15s | Current: user_xyz789
```

### 3. Colored Status Messages
- ℹ️ Info (blue)
- ✓ Success (green)
- ⚠️ Warning (yellow)
- ✗ Error (red)
- ⟳ Processing (cyan)

### 4. Section Headers
- Clear visual separation
- Bold formatting
- Automatic width adjustment

## Workflow

### Starting the Processor

```bash
python main.py
```

### What You'll See

#### 1. Initialization Phase
```
============================================================
WEIGHT STREAM PROCESSING
============================================================
ℹ  Processing file: ./2025-09-05_optimized.csv
⟳  Counting rows...
✓  Found 704,225 rows to process
```

#### 2. Processing Phase
```
Processing weight data...
████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░ 30.2% | 212,543/704,225 | ETA: 3m 45s | User: abc123
```

Live updates show:
- Progress bar with color transitions
- Exact row count
- Estimated time remaining
- Current user being processed

#### 3. Visualization Phase
```
============================================================
CREATING VISUALIZATIONS
============================================================
ℹ  Creating up to 10 dashboards in output/visualizations

Generating dashboards...
██████████████████████░░░░░░░░░░░░░░░░░ 5/10 | ETA: 5s | User: user_005
```

#### 4. Completion Summary
```
============================================================
PROCESSING COMPLETE
============================================================
✓  Total rows processed: 704,225
✓  Total users: 3,142
✓  Initialized users: 2,987
ℹ  Processing time: 245.3 seconds
ℹ  Processing rate: 2,871 rows/sec
✓  Acceptance rate: 87.3%

📁 Output Files Created:
  • Results: output/results
    - Summary: summary_20250907_174523.json
    - All results: all_results_20250907_174523.json
    - User files: 2,987 files
  • Visualizations: output/visualizations
    - Dashboards: 10 files
  • Logs: output/logs/app.log

============================================================
DONE
============================================================
```

## Configuration

### Adjusting Verbosity

In `config.toml`:

```toml
[logging]
stdout_level = "WARNING"  # Minimal console output (cleanest progress)
# or
stdout_level = "INFO"     # More detailed console output
# or  
stdout_level = "DEBUG"    # Full debug output (progress may be cluttered)
```

### Progress Settings

The progress system automatically:
- Detects terminal width
- Adjusts update frequency (100ms minimum)
- Handles terminal resize
- Falls back gracefully if terminal detection fails

## Performance Impact

The progress indication system has minimal overhead:
- < 0.1% CPU usage
- Updates throttled to 10Hz maximum
- No impact on processing speed
- Automatic cleanup on interruption

## Error Handling

If processing encounters errors:
```
✗ Encountered an error at row 45,231
⚠  Attempting to recover...
✓  Recovery successful, continuing...
```

The progress bar continues after recovery, with error count tracked.

## Interrupting Processing

Press `Ctrl+C` to stop:
- Progress bar clears cleanly
- Partial results are saved
- Summary shows what was completed

## Benefits

1. **Transparency**: See exactly what's happening
2. **Time Estimation**: Know how long processing will take
3. **Early Problem Detection**: Spot issues immediately
4. **Professional Appearance**: Polished CLI experience
5. **Reduced Anxiety**: No more wondering if it's stuck

## Terminal Compatibility

Works best with:
- Modern terminals with ANSI color support
- Width of 80+ characters
- UTF-8 encoding for icons

Fallback for basic terminals:
- Disables colors automatically
- Uses ASCII characters
- Maintains functionality

## Tips

1. **For fastest processing**: Set `stdout_level = "WARNING"`
2. **For debugging**: Set `stdout_level = "DEBUG"` 
3. **For normal use**: Default settings are optimal
4. **Large files**: Progress bar shows accurate ETA after ~5% completion
5. **Multiple runs**: Each run gets timestamped outputs

## Implementation Details

The progress system uses:
- ANSI escape codes for colors
- Unicode characters for progress bar
- Real-time terminal width detection
- Efficient string buffering
- Rate-limited updates

All progress code is in `src/core/progress.py` for easy customization.
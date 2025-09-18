# Daily Weight Analysis Module

## Overview

The `generate_daily_analysis.py` module creates a detailed day-by-day comparison of raw vs filtered weight data for each user, using the same "closest value" logic as the existing 90-day analysis.

## Features

- **Comprehensive Daily Tracking**: Generates records for each user for each day (0-180 days from start)
- **Same Logic as 90-Day Analysis**: Uses identical `get_weight_at_date` function with 20-day window
- **Performance Optimized**: Processes 1000+ users efficiently with batch processing
- **Detailed Metrics**: 15 columns of data per record including cumulative loss and divergence

## Output Format

### Main Output: `daily_weight_analysis.csv`

| Column | Description |
|--------|-------------|
| `user_id` | User identifier |
| `day_number` | Days since start (0, 1, 2, ..., 180) |
| `date` | Actual calendar date (YYYY-MM-DD) |
| `raw_weight` | Closest raw weight value (or NULL) |
| `raw_days_offset` | Days between measurement and target date |
| `filtered_weight` | Closest filtered weight value (or NULL) |
| `filtered_days_offset` | Days between measurement and target date |
| `raw_cumulative_loss_kg` | Weight lost since start (raw) |
| `raw_cumulative_loss_pct` | Percentage lost since start (raw) |
| `filtered_cumulative_loss_kg` | Weight lost since start (filtered) |
| `filtered_cumulative_loss_pct` | Percentage lost since start (filtered) |
| `divergence_kg` | Difference between raw and filtered (kg) |
| `divergence_pct` | Difference between raw and filtered (%) |
| `has_raw_measurement` | Boolean: measurement found within window |
| `has_filtered_measurement` | Boolean: measurement found within window |

### Summary Output: `daily_analysis_summary.json`

Contains aggregate statistics including:
- Total records generated
- Data availability percentages
- Average/max divergence between raw and filtered
- Processing performance metrics

## Usage

### Standalone Execution

```bash
# Run with default settings (180 days)
python generate_daily_analysis.py

# Customize days and batch size
python generate_daily_analysis.py --max-days 90 --batch-size 100

# Specify output directory
python generate_daily_analysis.py --output ./results/
```

### Integration with Main Pipeline

The module is automatically called by `run_analysis.py`:

```bash
python run_analysis.py

# With employer filter
python run_analysis.py --employer AMAZON_EMPLOYER

# With user limit (for testing)
python run_analysis.py --limit 100
```

### Programmatic Usage

```python
from generate_daily_analysis import generate_daily_report
from analyze_90_day import load_eligible_users

# Load eligible users
user_start_dates = load_eligible_users()

# Generate report
summary = generate_daily_report(
    user_start_dates=user_start_dates,
    output_path=Path("./results"),
    max_days=180,
    batch_size=50
)

print(f"Generated {summary['total_records']} records")
print(f"Processing took {summary['processing_time_seconds']:.1f} seconds")
```

## Performance Optimizations

The module includes several performance optimizations:

1. **Batch Processing**: Users processed in configurable batches (default 50)
2. **Pre-processing**: Data indexed by user_id for O(1) lookups
3. **Memory Management**: Original DataFrames freed after pre-processing
4. **Incremental Writing**: CSV written batch-by-batch to avoid memory issues
5. **Efficient Column Selection**: Only loads necessary columns (user_id, effectiveDateTime, weight)
6. **Progress Tracking**: Real-time ETA calculation and progress reporting

## Performance Benchmarks

Expected performance (based on typical data):
- 100 users × 180 days: ~5 seconds
- 500 users × 180 days: ~20 seconds  
- 1000 users × 180 days: ~40 seconds
- 5000 users × 180 days: ~3 minutes

Memory usage remains constant regardless of dataset size due to batch processing.

## Key Implementation Details

### Closest Value Logic

The module uses the exact same `get_weight_at_date` function from `analyze_90_day.py`:
- Searches within a 20-day window (before and after target date)
- Returns the measurement closest to the target date
- Returns None if no measurement found within window

### No Interpolation

The module does NOT interpolate missing values. If no measurement exists within the 20-day window, the weight fields are NULL. This preserves data integrity and makes gaps visible.

### Divergence Calculation

Divergence is calculated two ways:
- `divergence_kg`: Simple difference in weight (filtered - raw)
- `divergence_pct`: Difference in cumulative loss percentage

## Troubleshooting

### "No module named 'pandas'" Error

Install required dependencies:
```bash
pip install pandas numpy
```

### Memory Issues

Reduce batch size:
```python
generate_daily_report(user_start_dates, output_path, batch_size=10)
```

### Slow Performance

1. Check data file sizes - ensure indices on user_id
2. Reduce max_days if full 180 days not needed
3. Process subset of users first for testing

## Example Analysis

To answer "How much weight has each user lost since start date, day by day?":

```python
import pandas as pd

# Load the generated CSV
df = pd.read_csv("daily_weight_analysis.csv")

# Filter to a specific user
user_df = df[df['user_id'] == 'user_123']

# See their journey
print(user_df[['day_number', 'date', 'raw_cumulative_loss_pct', 'filtered_cumulative_loss_pct']])

# Find maximum divergence day
max_div_day = user_df.loc[user_df['divergence_pct'].abs().idxmax()]
print(f"Maximum divergence on day {max_div_day['day_number']}: {max_div_day['divergence_pct']:.2f}%")

# Count days where raw vs filtered disagree on gain/loss
disagree = user_df[
    (user_df['raw_cumulative_loss_pct'] > 0) != 
    (user_df['filtered_cumulative_loss_pct'] > 0)
]
print(f"Raw and filtered disagree on {len(disagree)} days")
```

## Files Generated

- `daily_weight_analysis.csv` - Main output with all daily records
- `daily_analysis_summary.json` - Summary statistics and metadata

## Integration Status

✅ Module implemented and tested
✅ Integrated into main pipeline (`run_analysis.py`)
✅ Performance optimizations implemented
✅ Uses same logic as existing 90-day analysis
✅ Handles missing data appropriately
✅ Generates comprehensive output
# Weight Loss Analysis Report - Usage Guide

## Overview

The `report.py` tool analyzes weight loss progress at 30-day intervals, comparing raw measurements against Kalman-filtered data to quantify the impact of outlier rejection.

## Prerequisites

Ensure your `config.toml` is properly configured with:
- `profile` - Processing profile (e.g., "custom", "balanced", "aggressive")
- `[data]` section with:
  - `min_readings` - Minimum readings per user (e.g., 20)
  - `max_users` - Maximum users to process (e.g., 100)
  - `user_offset` - User pagination offset (usually 0)

## Commands

### 1. Generate Filtered Dataset

First, process raw data through the Kalman filtering pipeline:

```bash
uv run python report.py filter <raw_csv> --output <filtered_csv>
```

Example:
```bash
uv run python report.py filter data/2025-09-05_nocon.csv \
  --output reports/filtered_weights.csv
```

This command:
- Processes raw data through full Kalman filtering pipeline
- Applies quality scoring and outlier detection
- Respects config settings (min_readings, max_users)
- Outputs clean dataset with only accepted measurements

### 2. Generate Analysis Report

#### Option A: Full Pipeline (Automatic Filtering + Analysis)

```bash
uv run python report.py analyze <raw_csv> --output-dir reports
```

This automatically:
1. Generates filtered data through Kalman pipeline
2. Analyzes both raw and filtered datasets
3. Compares results to show filtering impact
4. Creates reports and visualizations

⚠️ **Note**: This can be slow for large datasets as it runs the full pipeline.

#### Option B: Two-Step Process (Recommended for Large Datasets)

Step 1: Generate filtered data (once)
```bash
uv run python report.py filter data/weights.csv \
  --output reports/filtered_weights.csv
```

Step 2: Analyze using pre-filtered data
```bash
uv run python report.py analyze data/weights.csv \
  --use-filtered reports/filtered_weights.csv \
  --output-dir reports
```

#### Option C: Quick Analysis (Skip Filtering)

For testing or when you only have filtered data:
```bash
uv run python report.py analyze data/weights.csv \
  --skip-filtering \
  --output-dir reports
```

This treats raw data as already filtered (no comparison possible).

## Parameters

### Filter Command

- `input_csv` - Raw CSV file to process
- `--output` - Output path for filtered CSV
- `--config` - Config file path (default: config.toml)
- `--output-dir` - Output directory (default: reports)

### Analyze Command

- `input_csv` - Raw CSV file to analyze
- `--output-dir` - Output directory for reports (default: reports)
- `--top-n` - Number of top divergent users to visualize (default: 200)
- `--interval-days` - Days between intervals (default: 30)
- `--window-days` - ±days tolerance for measurements (default: 7)
- `--skip-filtering` - Skip Kalman filtering (treat raw as filtered)
- `--use-filtered` - Path to pre-generated filtered CSV
- `--config` - Config file path (default: config.toml)

## Output Files

### CSV Files

1. **filtered_data.csv** - Clean dataset with outliers removed
2. **user_progress_raw.csv** - Weight at each interval (raw data)
3. **user_progress_filtered.csv** - Weight at each interval (filtered)
4. **interval_statistics.csv** - Statistics for box plots
5. **user_comparison.csv** - Users ranked by raw/filtered divergence

### Visualizations

- **weight_loss_distribution.png** - Box plots comparing intervals
- **outlier_impact_heatmap.png** - Impact across users/intervals
- **quality_correlation.png** - Data quality vs weight loss
- **population_summary.png** - Overall statistics
- **top_200_users/** - Individual charts for most affected users
- **top_200_summary.png** - Dashboard for top users

## Examples

### Small Dataset (Quick Test)

```bash
# Generate filtered data and full analysis
uv run python report.py analyze data/test_weights.csv \
  --output-dir test_reports \
  --top-n 10
```

### Large Dataset (Recommended Workflow)

```bash
# Step 1: Generate filtered data (do this once)
uv run python report.py filter data/2025-09-05_nocon.csv \
  --output data/2025-09-05_filtered.csv

# Step 2: Run analysis using filtered data
uv run python report.py analyze data/2025-09-05_nocon.csv \
  --use-filtered data/2025-09-05_filtered.csv \
  --output-dir reports/2025-09-05 \
  --top-n 50

# Step 3: View results
ls reports/2025-09-05/data/        # CSV outputs
ls reports/2025-09-05/visualizations/  # Charts
cat reports/2025-09-05/report_summary.json  # Summary
```

### Custom Configuration

```bash
# Use different config file
uv run python report.py filter data/weights.csv \
  --config config_aggressive.toml \
  --output data/filtered_aggressive.csv

# Analyze with custom intervals
uv run python report.py analyze data/weights.csv \
  --interval-days 14 \
  --window-days 3 \
  --output-dir reports/biweekly
```

## Configuration Impact

The analysis respects all config.toml settings:

- **Profile** - Determines filtering aggressiveness
- **min_readings** - Users with fewer readings are excluded
- **max_users** - Limits processing to top N users
- **user_offset** - Skips first N eligible users
- **quality_threshold** - Minimum quality score for acceptance
- **extreme_threshold** - Maximum allowed deviation
- **Kalman parameters** - Affect filtered weight calculations
- **Replay settings** - Additional outlier detection if enabled

## Performance Tips

1. **Pre-filter large datasets** - Run filter command separately
2. **Adjust max_users** - Start with smaller numbers for testing
3. **Use --skip-filtering** - For quick visualization tests
4. **Reduce --top-n** - Fewer user charts = faster generation
5. **Check config settings** - Ensure min_readings isn't too restrictive

## Troubleshooting

### "Missing required columns"
- Ensure CSV has: user_id, weight, timestamp/effectiveDateTime, source/source_type

### Processing takes too long
- Reduce max_users in config.toml
- Use pre-filtered data with --use-filtered
- Decrease --top-n for fewer visualizations

### No differences between raw/filtered
- Ensure you're comparing raw vs filtered (not filtered vs filtered)
- Check that filtering is actually removing outliers
- Verify config.toml has appropriate thresholds

### Empty results
- Check min_readings - may be filtering out all users
- Verify date range in config includes your data
- Ensure CSV has valid weight values
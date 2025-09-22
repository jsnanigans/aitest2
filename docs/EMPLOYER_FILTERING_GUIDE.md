# Employer Filtering Guide

## Overview
The filtering analysis script now supports analyzing data for specific employers, allowing you to generate employer-specific reports and insights.

### Key Behavior
- **When filtering by employer**: Analyzes ALL users from that employer (no user limit)
- **Visualizations**: Generated for the top 10 most impacted users (highest removal rates and variance reduction)
- **Metrics**: Calculated for the entire employer cohort

## Basic Usage

### Run analysis for a specific employer
```bash
uv run python scripts/run_filtering_analysis.py \
  --filter-employer "EMPLOYER_ID"
```

### With additional options
```bash
uv run python scripts/run_filtering_analysis.py \
  --filter-employer "EMPLOYER_ID" \
  --max-users 100 \
  --output-dir reports/employer_analysis
```

## Available Commands

### List all employers and their user counts
```bash
uv run python scripts/list_employers.py
```

### Show usage examples with top employers
```bash
uv run python scripts/show_employer_usage.py
```

## Examples

### Analyze the largest employer (7973 users)
```bash
uv run python scripts/run_filtering_analysis.py \
  --filter-employer "0a427a45-cebe-4cec-977b-f65a9b6534bc"
```

### Analyze second largest employer (2076 users) with custom settings
```bash
uv run python scripts/run_filtering_analysis.py \
  --filter-employer "287fcc30-03df-45f0-a00f-7f4b2814da0d" \
  --max-users 50 \
  --output-dir reports/employer_287fcc30
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--filter-employer` | Employer ID to filter by | None (all users) |
| `--max-users` | Maximum number of users to analyze<br>**Note**: When `--filter-employer` is used, ALL users from the employer are analyzed regardless of this setting | 10 |
| `--output-dir` | Directory for visualizations | reports/visualizations |
| `--employer` | Path to employer data file | data/2025-09-17-user-employers.csv |
| `--filtered` | Path to filtered data file | data/2025-09-05_all_filtered.csv |
| `--verbose` | Enable verbose logging | False |

## Output

The analysis generates:
- **Report**: `reports/filtering_analysis_TIMESTAMP.md` - Comprehensive markdown report
- **Metrics**: `reports/filtering_metrics_TIMESTAMP.json` - Detailed JSON metrics
- **Visualizations**: Saved to specified output directory
  - `cohort_distribution_overlay.png` - Raw vs filtered weight distributions
  - `outlier_clustering_map.png` - Outlier patterns visualization
  - `trajectory_fans.png` - Individual user trajectories

## Notes

1. The employer filtering requires the employer data file to exist and contain valid mappings
2. Only users present in both raw and filtered datasets will be analyzed
3. If no users are found for the specified employer, the script will show available employers
4. The analysis shows removal rates and quality improvements specific to that employer's users

## Impact Scoring

When visualizing individual users, the script selects the top 10 most impacted users based on:
- **Removal Rate**: Percentage of measurements filtered out
- **Variance Reduction**: Improvement in weight measurement consistency
- **Combined Impact Score**: Sum of removal rate and variance reduction

This ensures visualizations focus on users where filtering made the biggest difference.
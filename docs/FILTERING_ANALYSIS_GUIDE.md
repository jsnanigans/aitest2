# Filtering Effectiveness Analysis Guide

## Overview

The filtering effectiveness analysis system provides comprehensive metrics and visualizations to quantify the impact of data filtering on weight measurements. It compares raw and filtered data to demonstrate improvements in data quality, medical decision accuracy, and reporting reliability.

## Components

### 1. Core Analysis Module (`src/analysis/filtering_effectiveness.py`)

The `FilteringAnalyzer` class provides comprehensive metrics across multiple dimensions:

#### Distribution Metrics
- **Central Tendency**: Mean, median, mode shifts
- **Dispersion**: Standard deviation, IQR, MAD improvements
- **Shape**: Skewness and kurtosis corrections
- **Statistical Tests**: Normality, variance equality

#### Outlier Detection Metrics
- **Detection Rates**: Overall and by method (IQR, Z-score, MAD, temporal)
- **Outlier Characteristics**: Magnitude distribution, temporal clustering
- **Source Analysis**: Outlier rates by data source

#### Temporal Consistency Metrics
- **Daily Changes**: Maximum changes, impossible changes, volatility
- **Trend Analysis**: Correlation, smoothness, inflection points
- **Gap Handling**: Gap detection and interpolation quality

#### Medical Impact Metrics
- **Weight Change Accuracy**: Start/end point variance
- **Clinical Thresholds**: Misclassification rates
- **Confidence Intervals**: Width reduction in filtered data

#### Reporting Metrics
- **Cohort Statistics**: Mean weight loss comparisons
- **Success Rates**: Percentage achieving 5%, 10% weight loss
- **User Inclusion**: Valid baseline/endpoint counts
- **Statistical Power**: Variance reduction, effect size

### 2. Visualization Generator (`src/analysis/visualization_generator.py`)

The `FilteringVisualizationGenerator` creates comprehensive visualizations:

#### Individual User Visualizations
- **Dual-axis Time Series**: Raw vs filtered with outliers highlighted
- **Residual Plots**: Deviations from filtered trend
- **Daily Change Histograms**: Distribution comparisons
- **Quality Score Heatmaps**: Temporal quality patterns
- **Comprehensive Dashboard**: Multi-panel summary

#### Population-Level Visualizations
- **Distribution Overlays**: Kernel density plots
- **Outlier Clustering Maps**: 2D density in time-magnitude space
- **Source Reliability Matrix**: Heatmap of outlier rates
- **Trajectory Fans**: Individual and average weight trajectories
- **Impact Dashboard**: Clinical and reporting impact summary

### 3. Analysis Runner (`scripts/run_filtering_analysis.py`)

Orchestrates the complete analysis workflow:
- Loads raw data from CSV
- Processes through filtering pipeline
- Calculates all metrics
- Generates visualizations
- Produces comprehensive reports

## Usage

### Basic Analysis

```bash
# Run analysis on CSV data
uv run python scripts/run_filtering_analysis.py data/weights.csv

# With custom configuration
uv run python scripts/run_filtering_analysis.py data/weights.csv --config custom_config.toml

# Limit number of users analyzed
uv run python scripts/run_filtering_analysis.py data/weights.csv --max-users 50

# Specify output directory
uv run python scripts/run_filtering_analysis.py data/weights.csv --output-dir analysis_output

# Enable verbose logging
uv run python scripts/run_filtering_analysis.py data/weights.csv --verbose
```

### Test Run

```bash
# Run with synthetic test data
uv run python scripts/test_filtering_analysis.py
```

## Configuration

Add these settings to `config.toml`:

```toml
[analysis]
# Output settings
output_dir = "reports/visualizations"

# Data selection
max_users = 10                # Maximum users to analyze
min_measurements = 20          # Minimum measurements per user
parallel_processing = true     # Enable parallel processing

# Visualization settings
chart_dpi = 150
generate_individual_charts = true
```

## Output Files

The analysis generates several output files:

### Reports
- `reports/filtering_analysis_YYYYMMDD_HHMMSS.md`: Comprehensive markdown report
- `reports/filtering_metrics_YYYYMMDD_HHMMSS.json`: Detailed metrics in JSON format

### Visualizations
- `reports/visualizations/`: Main visualization directory
- `reports/visualizations/user_*/`: Individual user visualizations
- Population-level plots in main visualization directory

## Metrics Interpretation

### Key Success Indicators

#### Data Quality
- **Std Reduction > 20%**: Significant noise reduction
- **Outlier Rate < 5%**: Effective outlier detection
- **Daily Volatility < 1kg**: Improved temporal consistency

#### Medical Impact
- **CI Reduction > 15%**: Better measurement confidence
- **Direction Errors = 0**: No weight change misclassification
- **Misclassification Rate < 5%**: Minimal threshold crossing errors

#### Reporting Impact
- **Variance Reduction > 30%**: Improved cohort statistics
- **Effect Size > 0.5**: Meaningful statistical improvement
- **User Inclusion ≥ Raw**: No loss of valid users

### Warning Signs

- **Removal Rate > 20%**: Possible over-filtering
- **Trend Correlation < 0.9**: Potential signal loss
- **User Inclusion < Raw**: Loss of valid data points

## Example Analysis Workflow

1. **Prepare Data**
   - Ensure CSV has required columns: user_id, effectiveDateTime, weight
   - Optional columns: source, unit, quality_score

2. **Configure Analysis**
   - Edit config.toml for thresholds and parameters
   - Set output directory and user limits

3. **Run Analysis**
   ```bash
   uv run python scripts/run_filtering_analysis.py data/weights.csv
   ```

4. **Review Results**
   - Check markdown report for summary
   - Examine visualizations for patterns
   - Review JSON metrics for detailed analysis

5. **Iterate and Refine**
   - Adjust quality thresholds if needed
   - Re-run with different parameters
   - Compare results across configurations

## Troubleshooting

### Common Issues

**Memory Issues with Large Datasets**
- Reduce max_users in configuration
- Process in batches
- Disable parallel processing

**Missing Visualizations**
- Check matplotlib backend settings
- Ensure output directory has write permissions
- Review logs for specific errors

**Poor Filtering Performance**
- Review quality threshold settings
- Check source reliability mappings
- Examine outlier detection parameters

## API Reference

### FilteringAnalyzer

```python
from src.analysis.filtering_effectiveness import FilteringAnalyzer

analyzer = FilteringAnalyzer(config)

# Analyze single user
user_metrics = analyzer.analyze_user_data(
    user_id="user001",
    raw_data=raw_df,
    filtered_data=filtered_df
)

# Analyze cohort
cohort_metrics = analyzer.analyze_cohort_data(
    cohort_raw=raw_dict,
    cohort_filtered=filtered_dict
)
```

### FilteringVisualizationGenerator

```python
from src.analysis.visualization_generator import FilteringVisualizationGenerator

visualizer = FilteringVisualizationGenerator(output_dir="reports/viz")

# Generate user visualizations
files = visualizer.generate_user_visualization_suite(
    user_id="user001",
    raw_df=raw_df,
    filtered_df=filtered_df,
    metrics=user_metrics
)

# Generate cohort visualizations
files = visualizer.generate_cohort_visualization_suite(
    cohort_raw=raw_dict,
    cohort_filtered=filtered_dict,
    cohort_metrics=cohort_metrics
)
```

## Performance Considerations

- **Processing Time**: ~1-2 seconds per user with full visualization
- **Memory Usage**: ~50MB per 1000 users
- **Disk Space**: ~5MB of visualizations per user

## Future Enhancements

Planned improvements include:
- Interactive HTML dashboards
- Real-time streaming analysis
- Machine learning-based outlier detection
- Automated parameter optimization
- A/B testing framework for filter configurations
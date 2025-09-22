# Quarterly Reporting Guide

## Overview

The quarterly reporting module provides comprehensive analysis of weight loss outcomes for users who have been in the program for 90+ days. This addresses the key business question for quarterly reports: **"What is the average weight loss for users in the program for 90+ days?"**

## Key Features

### 1. Primary Metrics Answered

The system automatically calculates and reports:
- **Average weight loss** (mean and median) for all 90+ day users
- **Success rates** at 5%, 10%, and 15% weight loss thresholds
- **Data quality impact** - how filtering affects these metrics
- **Progression analysis** - weight loss at 90, 105, 120... up to 210 days

### 2. Raw vs Filtered Comparison

Every metric is calculated for both:
- **Raw data**: All measurements as collected
- **Filtered data**: After outlier removal and quality filtering
- **Improvement metrics**: Shows exactly how filtering impacts reporting accuracy

## Data Requirements

### Start Date Information
- Loaded from `data/2025-09-17-user-employers.csv`
- Contains `user_id` and `start_date` columns
- Used to determine how long each user has been in the program

### Reference Date
- Set to `2025-09-05` (when data was exported)
- Users must have `start_date` at least 90 days before this date to be included

## Analysis Components

### 1. Overall 90+ Day User Analysis

Answers: **"What's the average weight loss for all users in the program for 90+ days?"**

- Takes all users with 90+ days in program
- Calculates weight loss from start weight to last recorded weight
- Provides distribution statistics (mean, median, std, quartiles)
- Shows success rates at clinical thresholds

### 2. Cohort Progression Analysis

Shows weight loss progression at specific time intervals:
- Analyzes users at 90, 105, 120, 135, 150, 165, 180, 195, and 210 days
- For each checkpoint, takes the closest weight on or before that date
- Simulates "as-of" reporting to show how metrics evolve over time

### 3. Distribution Analysis

Provides detailed weight loss distribution comparisons:
- Box plots showing quartiles and outliers
- Violin plots showing distribution density
- Histograms with clinical threshold markers
- Success rate comparisons

## Generated Visualizations

The system automatically generates 4 comprehensive visualizations:

### 1. `quarterly_weight_loss_distribution.png`
- Box plots comparing raw vs filtered distributions
- Violin plots showing density
- Histogram overlays
- Success rate bar charts

### 2. `quarterly_cohort_progression.png`
- Mean weight loss over time (90-210 days)
- Success rate progression
- Data availability at each checkpoint
- Standard deviation comparison

### 3. `quarterly_detailed_metrics.png`
- Key statistical metrics comparison
- Distribution quartiles
- Success counts by threshold
- Data completeness metrics

### 4. `quarterly_impact_summary.png`
- Executive dashboard showing filtering impact
- Average weight loss improvement
- Success rate changes
- Variability reduction

## Report Section

The quarterly analysis adds a dedicated section to the main report:

```markdown
## 📊 QUARTERLY REPORTING ANALYSIS

### Key Business Question Answered

**"What is the average weight loss for users in the program for 90+ days?"**

| Metric | Raw Data | Filtered Data | Improvement |
|--------|----------|---------------|-------------|
| **Average Weight Loss** | X.XX% | Y.YY% | ±Z.ZZ% |
| Median Weight Loss | X.XX% | Y.YY% | ±Z.ZZ% |
| Standard Deviation | X.XX% | Y.YY% | Z.ZZ% reduction |
```

## Usage

### Basic Analysis
```bash
# Run with quarterly reporting (included by default)
uv run python scripts/run_filtering_analysis.py
```

### With Employer Filtering
```bash
# Analyze specific employer's 90+ day users
uv run python scripts/run_filtering_analysis.py \
  --filter-employer "EMPLOYER_ID"
```

### Test Quarterly Functionality
```bash
# Run dedicated quarterly test
uv run python scripts/test_quarterly_reporting.py
```

## Key Insights Provided

1. **Reporting Accuracy**: Shows exactly how filtering affects quarterly metrics
2. **Data Quality Impact**: Quantifies how many users have reliable data
3. **Clinical Outcomes**: Tracks success rates at medical thresholds
4. **Time-based Trends**: Shows how weight loss progresses over program duration
5. **Statistical Confidence**: Reduces variance and improves reliability of averages

## Technical Implementation

### Modules
- `src/analysis/quarterly_reporting.py` - Core analysis logic
- `src/analysis/quarterly_visualizations.py` - Visualization generation
- Integrated into `scripts/run_filtering_analysis.py`

### Key Classes
- `QuarterlyReportingAnalyzer` - Performs the analysis
- `QuarterlyVisualizationGenerator` - Creates visualizations
- `QuarterlyMetrics` - Data structure for metrics
- `CohortAnalysis` - Data structure for time-based analysis

## Interpretation Guide

### Positive Improvements
- **Higher average weight loss** in filtered data = better accuracy
- **Lower standard deviation** = more consistent results
- **Higher data availability** = more complete analysis

### Warning Signs
- Large differences between raw and filtered (>2%) may indicate data quality issues
- Low data availability (<80%) suggests missing measurements
- High standard deviation (>10%) indicates inconsistent outcomes

## Business Value

This quarterly reporting provides:

1. **Accurate program effectiveness metrics** for stakeholder reports
2. **Evidence-based success rates** for clinical validation
3. **Data quality assurance** for regulatory compliance
4. **Trend analysis** for program optimization
5. **Comparative insights** showing the value of data filtering
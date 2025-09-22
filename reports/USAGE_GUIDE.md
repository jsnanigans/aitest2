# Comprehensive Filtering Analysis - Usage Guide

## Quick Start

### 1. Run Basic Analysis
```bash
# Analyze all users
uv run python scripts/run_comprehensive_analysis.py

# Analyze specific employer
uv run python scripts/run_comprehensive_analysis.py --employer "AMAZON_EMPLOYER"

# Generate detailed report
uv run python scripts/run_comprehensive_analysis.py --detailed
```

### 2. Run Full Comprehensive Analysis
```bash
# With raw and filtered data files
uv run python scripts/comprehensive_filtering_analysis.py \
    --raw-data data/weights.csv \
    --filtered-data data/weight_filtered.csv \
    --output-dir reports \
    --visualize
```

## Analysis Components Created

### 1. **Comprehensive Analysis Plan** (`reports/comprehensive_filtering_analysis_plan.md`)
- Complete framework for analyzing filtered vs raw data
- Quantitative metrics definitions
- Success criteria and evaluation methods
- Implementation roadmap

### 2. **Comprehensive Filtering Analyzer** (`scripts/comprehensive_filtering_analysis.py`)
Full implementation including:
- **Statistical Metrics**: Mean, median, std dev, IQR, CV analysis
- **Outlier Detection**: IQR, Z-score, MAD, Isolation Forest methods
- **Temporal Consistency**: Daily change analysis, volatility indexing
- **Medical Impact**: Weight change accuracy, threshold crossing analysis
- **Quarterly Reporting**: Cohort statistics, success rates, statistical power

### 3. **Integration Script** (`scripts/run_comprehensive_analysis.py`)
Enhanced analysis features:
- **Outlier Source Analysis**: Categorizes removed outliers by type
- **Multi-User Detection**: Identifies bimodal distributions suggesting shared scales
- **Medical Decision Impacts**: Calculates specific clinical decision changes
- **Executive Summary Generation**: Automated report with key findings

## Output Reports

### Executive Summary
Provides high-level insights:
- Data quality improvements
- Multi-user detection results
- Medical decision safety assessment
- Quarterly reporting accuracy impact
- Specific recommendations

### Detailed Analysis Report
Comprehensive metrics including:
- Distribution comparisons (raw vs filtered)
- Outlier removal statistics by method
- Temporal consistency improvements
- Clinical threshold analysis
- Statistical test results
- Success criteria evaluation

### Analysis Data (JSON)
Machine-readable output containing:
- Outlier analysis details
- Multi-user patterns
- Medical impact measurements
- Source reliability metrics

## Key Metrics Explained

### 1. **Variance Reduction**
- Target: >20% reduction in standard deviation
- Impact: Cleaner data for analysis
- Formula: `(σ_raw - σ_filtered) / σ_raw * 100`

### 2. **Outlier Detection Rate**
- Measures: % of points removed
- Expected: 5-15% for normal datasets
- Too low (<5%): May miss errors
- Too high (>20%): May remove valid data

### 3. **Medical Safety Metrics**
- **Direction Agreement**: % of cases where weight change direction (gain/loss) matches
- **Threshold Crossing**: Users reclassified at 5% or 10% weight loss thresholds
- **Magnitude Error**: Average kg difference in weight change calculations

### 4. **Reporting Impact**
- **Effect Size**: Cohen's d improvement for cohort analysis
- **Confidence Intervals**: Width reduction in weight loss estimates
- **User Inclusion**: Change in eligible users for reporting

## Interpretation Guidelines

### ✅ Good Filtering Performance
- Variance reduction >20%
- Direction agreement >95%
- <1kg average weight change difference
- Effect size improvement >0.2

### ⚠️ Areas for Review
- Removal rate <5% or >20%
- Direction agreement <90%
- >2kg weight change differences
- Multiple users with bimodal distributions

### ❌ Filtering Issues
- Direction agreement <85%
- >3kg weight change differences
- Negative effect size improvement
- >30% removal rate

## Advanced Usage

### Custom Time Windows
```python
# Analyze specific date range
analyzer = ComprehensiveFilteringAnalyzer()
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
reporting_metrics = analyzer.analyze_quarterly_reporting_impact(
    raw_df, filtered_df, start_date, end_date
)
```

### Source-Specific Analysis
```python
# Analyze by data source
for source in ['care-team-upload', 'patient-device', 'iglucose.com']:
    source_raw = raw_df[raw_df['source'] == source]
    source_filtered = filtered_df[filtered_df['source'] == source]
    metrics = analyzer.analyze_user_data(source_raw, source_filtered, f"source_{source}")
```

### Cohort Comparison
```python
# Compare multiple cohorts
cohorts = ['AMAZON_EMPLOYER', 'GOOGLE_EMPLOYER', 'FACEBOOK_EMPLOYER']
for cohort in cohorts:
    # Run analysis for each cohort
    # Compare results
```

## Troubleshooting

### Issue: Memory errors with large datasets
**Solution**: Use sampling
```bash
uv run python scripts/comprehensive_filtering_analysis.py \
    --raw-data data/weights.csv \
    --filtered-data data/weight_filtered.csv \
    --user-sample 1000
```

### Issue: Missing visualization libraries
**Solution**: Install optional dependencies
```bash
uv pip install plotly matplotlib seaborn
```

### Issue: No filtered data available
**Solution**: Run filtering first
```bash
uv run python main.py data/weights.csv --config config.toml
```

## Next Steps

1. **Run initial analysis** to establish baseline metrics
2. **Review executive summary** for key findings
3. **Examine detailed report** for specific areas of concern
4. **Adjust filtering parameters** based on recommendations
5. **Re-run analysis** to verify improvements
6. **Generate visualizations** for presentations

## Contact & Support

For questions about the analysis or interpretation of results, consult:
- Data Science team for statistical questions
- Medical team for clinical threshold decisions
- Engineering team for implementation details
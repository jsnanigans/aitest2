
## Analysis Overview
Generated: 2025-09-22 15:40:02

### Methodology

- **Raw Data**: Original weight measurements from all sources without any filtering
- **Filtered Data**: Measurements processed through our quality pipeline including:
  - Adaptive Kalman filtering for noise reduction
  - Statistical outlier detection (IQR, MAD, temporal consistency)
  - Source-specific reliability weighting
  - Quality score-based acceptance thresholds

## Executive Summary

- **Total Users Analyzed**: 4133
- **Total Readings:**
	- Raw = `346497`
	- Filtered = `357921`
- **Average Removal Rate**: 3.0% `Percentage of measurements filtered out`
- **Average CI Improvement**: 2.6% `Confidence interval improvement`
	- >5% would be considered a "meaningful statistical change"

> impact: the data is still quite chaotic. we still have a few big jumps in the data which are hard/impossible to handle automatically.
> Example of cases that highly impact the consistency of the data:
> - long gaps in the data
> - manually entered values that "must be trusted" that are significantly different than the established baseline (>5%)
> 	- sources: care-team-uploads, questionnaires
> 	- manual entries from users do not trigger a "reset"


## Cohort-Level Impact

This section shows how filtering affects cohort-level reporting metrics that are critical for clinical trials and population health studies.

### Weight Change Statistics

*These metrics show the average weight change across all users in the cohort.*

| Metric | Raw | Filtered | Improvement |
|--------|-----|----------|-------------|
| Mean Weight Change | -3.73% | -4.88% | -1.15% |
![[weight_change_comparison.png]]

> impact: filtering does slightly change the outcomes for the reports, but overall the filtered data is still very consistent with the raw data

### Clinical Success Rates

*Percentage of users achieving clinically significant weight loss thresholds.*

| Threshold       | Raw   | Filtered | Delta |
| --------------- | ----- | -------- | ----- |
| 5% Weight Loss  | 47.3% | 48.3%    | +1.0% |
| 10% Weight Loss | 27.7% | 26.7%    | -1.0% |
| missing data    | 25%   | 25%      |       |
![[clinical_success_rates 1.png]]

> impact: outliers and unreasonable changes in data are filtered out, this leads to a reduction in outcomes for >5% weight loss, but an increase for outcomes that are <=5% weight loss.



### User Inclusion Impact

*How filtering affects the number of users with valid data for analysis.*

| Stage | Raw | Filtered | Change |
|-------|-----|----------|--------|
| Valid Baseline | 755 | 673 | -82 |
| Valid Endpoint | 596 | 528 | -68 |

![[user_inclusion_funnel 2.png]]

> impact: after filtering, less users have sufficient data for the full analysis which picks the closest value to the start/end date (future blind)
> - program-start -> 14d
> - 14d -> 90d

### Statistical Power Improvements

*How filtering improves the statistical reliability of analyses.*

- **Variance Reduction**: 42.9% - Lower variance means more consistent measurements
- **Effect Size Improvement**: 0.347 - Larger effect sizes are easier to detect statistically



> Variance Reduction is HIGH! commonly a 15-40% is considered a substantial improvement
>
> Effect Size Imp. is a bit low -- in statistics a <0.5 is considered to be a "small effect", but in our case I think thins is good because it means that it only has a small effect on the overall data.
> In medical fields a "Effect Size Imp." of 0.1 could already be highly significant for health (psychology: .07 (small), .16 (medium) and .32 (large)) [-source-](https://www.tandfonline.com/doi/full/10.1080/10503307.2025.2494270#:~:text=Different%20thresholds%20for%20effect%20sizes,24%2C%20and%20.)

## 📊 QUARTERLY REPORTING ANALYSIS

This section analyzes users who have been in the program for 90+ days, which is the standard timeframe for quarterly business reporting and clinical outcome assessment.

In contrast to the "Clinical Success Rates", this does not take the value at 90 days, but instead it takes the latest value (based on 2025-09-11 data), thats why the outcome differs.

### Key Business Question Answered

**"What is the average weight loss for users in the program for 90+ days?"**

| Metric                  | Raw Data | Filtered Data | Improvement     |
| ----------------------- | -------- | ------------- | --------------- |
| **Average Weight Loss** | 6.03%    | 6.08%         | +0.05%          |
| Median Weight Loss      | 5.12%    | 5.13%         | +0.01%          |
| Standard Deviation      | 8.04%    | 7.15%         | 0.89% reduction |
|                         |          |               |                 |
> impact: better numbers in reports

### Data Quality Impact

*How many users have usable data for quarterly reporting.*

- **Eligible Users**: 3633 users with 90+ days in program
- **Valid Data (Raw)**: 3633 users (100.0%)
- **Valid Data (Filtered)**: 3585 users (98.7%)



### Clinical Success Rates (90+ Day Users)

| Threshold | Raw Success Rate | Filtered Success Rate | Difference |
|-----------|-----------------|----------------------|------------|
| 5% Loss | 50.5% (1834 users) | 50.6% (1814 users) | +0.1% |
| 10% Loss | 26.6% (966 users) | 26.8% (959 users) | +0.2% |
| 15% Loss | 10.9% (396 users) | 10.9% (389 users) | -0.0% |

![[quarterly_success_rates.png]]

### Weight Loss Progression by Program Duration

Average weight loss at different time checkpoints:

| Days in Program | Raw Avg Loss | Filtered Avg Loss | Improvement |
| --------------- | ------------ | ----------------- | ----------- |
| 90 days         | 2.51%        | 2.61%             | +0.10% 📈   |
| 105 days        | 2.93%        | 2.99%             | +0.06% ➡️   |
| 120 days        | 3.30%        | 3.42%             | +0.12% 📈   |
| 135 days        | 3.73%        | 3.88%             | +0.15% 📈   |
| 150 days        | 4.18%        | 4.33%             | +0.15% 📈   |
| 165 days        | 4.75%        | 4.83%             | +0.08% ➡️   |
| 180 days        | 5.23%        | 5.27%             | +0.04% ➡️   |
| 195 days        | 5.64%        | 5.74%             | +0.10% ➡️   |
| 210 days        | 6.10%        | 6.22%             | +0.12% 📈   |

**Average Improvement Across All Checkpoints:** +0.10%
**Maximum Improvement:** +0.15% at 135 days

### Quarterly Reporting Visualizations

#### Weight loss progression
![[weight_loss_progression_chart.png]]
![[quarterly_cohort_progression.png]]

#### Weight loss distributions
![[quarterly_weight_loss_distribution.png]]

#### Weight loss impact summary
![[quarterly_impact_summary.png]]

#### Quarterly reporting metrics details
![[quarterly_detailed_metrics.png]]


## Key Findings & Interpretation

### Data Quality Improvements

*These metrics show how filtering improves the reliability of weight measurements.*

1. **Outlier Detection**: Successfully identified and removed 9904 outliers across all users
2. **Temporal Consistency**: Reduced daily weight volatility by an average of 0.59kg
3. **Impossible Changes**: Eliminated 5434 physiologically impossible weight changes

### Clinical Impact

*How filtering prevents medical misinterpretations and improves clinical decision-making.*

1. **Direction Errors**: Prevented 32 cases where weight change direction would be misclassified (e.g., showing gain instead of loss)
2. **Confidence Intervals**: Improved measurement confidence by 2.6% on average (tighter confidence bands mean more reliable measurements)

## notes

### impact
1. The filtering process significantly improves data quality without compromising clinical validity

### future todos/improvements:
1. **Source Monitoring**: Pay special attention to data sources with high outlier rates
2. **Threshold Tuning**: improve the weights for the quality scoring of readings
3. **High uncertainty alerts**: when there is high variance or lots of uncertainty in the kalman filter, alert care team to look into the data
4. **Graph in PMP for weight**: show the accepted and rejected values in PMP for manual reviews and easier "fixing" of the data by adding historic values -- in cases where the algo takes a wrong path or to prevent a reset
5. **Reset alerts**: when data is reset because of a large gap in data, or because a care team entry or user manual entry has caused a reset in the algo.
6. **Regular Validation**: Regularly review impact and accuracy


## Technical Details

### Configuration Used

```toml
quality_threshold = 0.46
initial_variance = 0.364
```

### Analysis Parameters

- Analysis timestamp: 2025-09-22T13:51:03.803635
- Cohort size: 4199 users
- Minimum measurements per user: 20

---

*End of Report*

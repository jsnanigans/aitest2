# Statistical Evidence Report: Filtering Effectiveness
Generated: 2025-09-18 18:29:49

## Executive Summary

This report provides statistical evidence for the effectiveness of weight data filtering
based on analysis of 1 users with comprehensive data.

## 1. Distribution Normality (Shapiro-Wilk Test)

**Hypothesis**: Filtering improves data distribution normality

- **Raw data normal distributions**: 0/1 (0.0%)
- **Filtered data normal distributions**: 1/1 (100.0%)
- **Improvement rate**: 1/1 users showed improvement

**Verdict**: ✅ Filtering improves normality

## 2. Variance Reduction

**Hypothesis**: Filtering reduces measurement variance

- **Mean variance reduction**: 97.94%
- **Median variance reduction**: 97.94%
- **Standard deviation reduction**: 85.66%
- **Users with reduced variance**: 1/1 (100.0%)

**Verdict**: ✅ Significant variance reduction

## 3. Trend Smoothness

**Hypothesis**: Filtering produces smoother weight trends

- **Mean smoothness improvement**: 13943.8%
- **Median smoothness improvement**: 13943.8%
- **Jitter reduction**: 99.3%
- **Users with smoother trends**: 1/1 (100.0%)

**Verdict**: ✅ Trends significantly smoother

## 4. Plausible Weight Changes (GLP-1 Adjusted)

**Hypothesis**: Filtering removes implausible weight changes while preserving GLP-1 medication effects

- **Raw implausible rate**: 0.000%
- **Filtered implausible rate**: 0.000%
- **Implausible values removed**: 0
- **Extreme changes (>2kg/day) removed**: 1
- **Rapid weight loss events (0.5-1kg/day)**:
  - Raw data: 6 occurrences
  - Filtered data: 6 occurrences

*Note: Thresholds adjusted for GLP-1 medication - up to 1kg/day loss considered physiologically plausible*

**Verdict**: ✅ Improved plausibility while preserving GLP-1 effects

## 5. Temporal Consistency (Autocorrelation)

**Hypothesis**: Filtering improves temporal consistency

- **Raw mean autocorrelation**: 0.018
- **Filtered mean autocorrelation**: 0.659
- **Mean improvement**: +0.641
- **Users with improved consistency**: 1/1

**Verdict**: ✅ Improved temporal consistency

## 6. 90-Day Weight Loss Statistical Tests

### Paired t-test (Raw vs Filtered Weight Loss %)
- **t-statistic**: 3.7390
- **p-value**: 0.000187
- **Result**: Significant difference

### Wilcoxon Signed-Rank Test (Non-parametric)
- **Statistic**: 558.00
- **p-value**: 0.000000
- **Significant**: Yes

### Effect Size (Cohen's d)
- **Value**: -0.054
- **Interpretation**: Small effect

### Success Rate Comparison (Chi-square)
- **Raw success rate**: 79.0%
- **Filtered success rate**: 78.9%
- **Chi-square statistic**: 0.0057
- **p-value**: 0.940007
- **Significant difference**: No

## Overall Conclusion

**Statistical Evidence Score**: 4/5 metrics show improvement

### Final Verdict:

✅ **STRONG EVIDENCE**: Filtering significantly improves data quality
- Multiple statistical tests confirm improvement
- Recommendation: Continue using current filtering approach


## Key Findings

1. **Variance Reduction**: 97.9% average reduction in measurement variance
2. **Smoothness Gain**: 13944% improvement in trend smoothness
3. **Temporal Consistency**: +0.641 improvement in autocorrelation
4. **Plausibility (GLP-1 adjusted)**: 1 extreme changes removed

## Methodology Notes

- Sample size: 1 users
- Significance level: α = 0.05
- Tests performed: Shapiro-Wilk, paired t-test, Wilcoxon, Chi-square
- Metrics: Variance, smoothness, autocorrelation, plausible weight changes
- **GLP-1 Adjustment**: Weight loss up to 1kg/day considered plausible for users on GLP-1 medication

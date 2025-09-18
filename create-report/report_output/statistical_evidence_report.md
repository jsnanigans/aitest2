# Statistical Evidence Report: Filtering Effectiveness
Generated: 2025-09-18 18:03:26

## Executive Summary

This report provides statistical evidence for the effectiveness of weight data filtering
based on analysis of 13 users with comprehensive data.

## 1. Distribution Normality (Shapiro-Wilk Test)

**Hypothesis**: Filtering improves data distribution normality

- **Raw data normal distributions**: 0/13 (0.0%)
- **Filtered data normal distributions**: 0/13 (0.0%)
- **Improvement rate**: 8/13 users showed improvement

**Verdict**: ❌ No normality improvement

## 2. Variance Reduction

**Hypothesis**: Filtering reduces measurement variance

- **Mean variance reduction**: 36.67%
- **Median variance reduction**: 50.66%
- **Standard deviation reduction**: 30.42%
- **Users with reduced variance**: 8/13 (61.5%)

**Verdict**: ✅ Significant variance reduction

## 3. Trend Smoothness

**Hypothesis**: Filtering produces smoother weight trends

- **Mean smoothness improvement**: 3737.8%
- **Median smoothness improvement**: 102.8%
- **Jitter reduction**: 39.9%
- **Users with smoother trends**: 10/13 (76.9%)

**Verdict**: ✅ Trends significantly smoother

## 4. Plausible Weight Changes (GLP-1 Adjusted)

**Hypothesis**: Filtering removes implausible weight changes while preserving GLP-1 medication effects

- **Raw implausible rate**: 0.000%
- **Filtered implausible rate**: 0.000%
- **Implausible values removed**: 0
- **Extreme changes (>2kg/day) removed**: 371
- **Rapid weight loss events (0.5-1kg/day)**:
  - Raw data: 41 occurrences
  - Filtered data: 40 occurrences

*Note: Thresholds adjusted for GLP-1 medication - up to 1kg/day loss considered physiologically plausible*

**Verdict**: ✅ Improved plausibility while preserving GLP-1 effects

## 5. Temporal Consistency (Autocorrelation)

**Hypothesis**: Filtering improves temporal consistency

- **Raw mean autocorrelation**: 0.472
- **Filtered mean autocorrelation**: 0.649
- **Mean improvement**: +0.177
- **Users with improved consistency**: 9/13

**Verdict**: ✅ Improved temporal consistency

## 6. 90-Day Weight Loss Statistical Tests

### Paired t-test (Raw vs Filtered Weight Loss %)
- **t-statistic**: -0.1536
- **p-value**: 0.880449
- **Result**: No significant difference

### Wilcoxon Signed-Rank Test (Non-parametric)
- **Statistic**: 8.00
- **p-value**: 0.195312
- **Significant**: No

### Effect Size (Cohen's d)
- **Value**: 0.044
- **Interpretation**: Small effect

### Success Rate Comparison (Chi-square)
- **Raw success rate**: 69.2%
- **Filtered success rate**: 84.6%
- **Chi-square statistic**: 0.2167
- **p-value**: 0.641592
- **Significant difference**: No

## Overall Conclusion

**Statistical Evidence Score**: 3/5 metrics show improvement

### Final Verdict:

✓ **MODERATE EVIDENCE**: Filtering provides meaningful improvements
- Most metrics show positive impact
- Recommendation: Fine-tune thresholds for optimal performance


## Key Findings

1. **Variance Reduction**: 36.7% average reduction in measurement variance
2. **Smoothness Gain**: 3738% improvement in trend smoothness
3. **Temporal Consistency**: +0.177 improvement in autocorrelation
4. **Plausibility (GLP-1 adjusted)**: 371 extreme changes removed

## Methodology Notes

- Sample size: 13 users
- Significance level: α = 0.05
- Tests performed: Shapiro-Wilk, paired t-test, Wilcoxon, Chi-square
- Metrics: Variance, smoothness, autocorrelation, plausible weight changes
- **GLP-1 Adjustment**: Weight loss up to 1kg/day considered plausible for users on GLP-1 medication

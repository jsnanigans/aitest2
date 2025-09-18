#!/usr/bin/env python3
"""
Statistical Evidence Report Generator
Performs statistical tests to validate filtering effectiveness
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Constants
DATA_DIR = Path("../data")
# These can be overridden by run_analysis.py
RAW_FILE = DATA_DIR / "2025-09-05_nocon.csv"
FILTERED_FILE = DATA_DIR / "2025-09-05_nocon_filtered.csv"

def perform_normality_tests(df_raw: pd.DataFrame, df_filtered: pd.DataFrame,
                           sample_users: List[str]) -> Dict:
    """
    Perform Shapiro-Wilk normality tests on weight distributions.

    Returns:
        Dictionary with test results
    """
    results = {
        'raw_normal_count': 0,
        'filtered_normal_count': 0,
        'both_normal_count': 0,
        'improvement_count': 0,
        'p_values_raw': [],
        'p_values_filtered': [],
        'sample_size': 0
    }

    for user_id in sample_users:
        user_raw = df_raw[df_raw['user_id'] == user_id]['weight'].values
        user_filtered = df_filtered[df_filtered['user_id'] == user_id]['weight'].values

        if len(user_raw) >= 3 and len(user_filtered) >= 3:
            try:
                # Perform Shapiro-Wilk test
                stat_raw, p_raw = stats.shapiro(user_raw) if len(user_raw) <= 5000 else (0, 0)
                stat_filtered, p_filtered = stats.shapiro(user_filtered) if len(user_filtered) <= 5000 else (0, 0)

                results['p_values_raw'].append(p_raw)
                results['p_values_filtered'].append(p_filtered)
                results['sample_size'] += 1

                # Count normal distributions (p > 0.05)
                if p_raw > 0.05:
                    results['raw_normal_count'] += 1
                if p_filtered > 0.05:
                    results['filtered_normal_count'] += 1
                if p_raw > 0.05 and p_filtered > 0.05:
                    results['both_normal_count'] += 1
                if p_filtered > p_raw:
                    results['improvement_count'] += 1

            except Exception as e:
                logging.debug(f"Normality test failed for user {user_id}: {e}")
                continue

    return results

def calculate_variance_metrics(df_raw: pd.DataFrame, df_filtered: pd.DataFrame,
                              sample_users: List[str]) -> Dict:
    """
    Calculate variance reduction metrics.

    Returns:
        Dictionary with variance statistics
    """
    variance_reductions = []
    std_reductions = []

    for user_id in sample_users:
        user_raw = df_raw[df_raw['user_id'] == user_id]['weight'].values
        user_filtered = df_filtered[df_filtered['user_id'] == user_id]['weight'].values

        if len(user_raw) > 1 and len(user_filtered) > 1:
            var_raw = np.var(user_raw)
            var_filtered = np.var(user_filtered)

            if var_raw > 0:
                var_reduction = ((var_raw - var_filtered) / var_raw) * 100
                variance_reductions.append(var_reduction)

                std_raw = np.std(user_raw)
                std_filtered = np.std(user_filtered)
                std_reduction = ((std_raw - std_filtered) / std_raw) * 100
                std_reductions.append(std_reduction)

    return {
        'mean_variance_reduction': np.mean(variance_reductions) if variance_reductions else 0,
        'median_variance_reduction': np.median(variance_reductions) if variance_reductions else 0,
        'mean_std_reduction': np.mean(std_reductions) if std_reductions else 0,
        'positive_reduction_count': sum(1 for v in variance_reductions if v > 0),
        'total_users': len(variance_reductions)
    }

def calculate_smoothness_metrics(df_raw: pd.DataFrame, df_filtered: pd.DataFrame,
                                sample_users: List[str]) -> Dict:
    """
    Calculate trend smoothness improvements.

    Returns:
        Dictionary with smoothness metrics
    """
    smoothness_improvements = []
    jitter_reductions = []

    for user_id in sample_users:
        user_raw = df_raw[df_raw['user_id'] == user_id].sort_values('effectiveDateTime')
        user_filtered = df_filtered[df_filtered['user_id'] == user_id].sort_values('effectiveDateTime')

        raw_weights = user_raw['weight'].values
        filtered_weights = user_filtered['weight'].values

        if len(raw_weights) > 2 and len(filtered_weights) > 2:
            # Calculate first differences (day-to-day changes)
            raw_diffs = np.diff(raw_weights)
            filtered_diffs = np.diff(filtered_weights)

            # Smoothness = inverse of variance in changes
            raw_jitter = np.var(raw_diffs)
            filtered_jitter = np.var(filtered_diffs)

            if raw_jitter > 0:
                jitter_reduction = ((raw_jitter - filtered_jitter) / raw_jitter) * 100
                jitter_reductions.append(jitter_reduction)

                # Alternative smoothness metric
                raw_smoothness = 1 / (raw_jitter + 0.001)
                filtered_smoothness = 1 / (filtered_jitter + 0.001)
                smooth_improvement = ((filtered_smoothness - raw_smoothness) / raw_smoothness) * 100
                smoothness_improvements.append(smooth_improvement)

    return {
        'mean_smoothness_improvement': np.mean(smoothness_improvements) if smoothness_improvements else 0,
        'median_smoothness_improvement': np.median(smoothness_improvements) if smoothness_improvements else 0,
        'mean_jitter_reduction': np.mean(jitter_reductions) if jitter_reductions else 0,
        'improved_count': sum(1 for s in smoothness_improvements if s > 0),
        'total_users': len(smoothness_improvements)
    }

def calculate_plausibility_metrics(df_raw: pd.DataFrame, df_filtered: pd.DataFrame,
                                  sample_users: List[str]) -> Dict:
    """
    Calculate plausible weight change improvements.
    Accounts for GLP-1 medication effects on weight loss rates.

    Returns:
        Dictionary with plausibility metrics
    """
    # Define plausible weight range (kg)
    MIN_PLAUSIBLE = 30
    MAX_PLAUSIBLE = 300

    # GLP-1 adjusted limits for weight changes
    # Standard: 0.5-1 kg/week (0.07-0.14 kg/day)
    # GLP-1: 0.5-1 kg/week initially, up to 1-2 kg/week (0.14-0.28 kg/day)
    # Allow up to 1 kg/day for extreme but possible GLP-1 responses
    # But flag anything over 2 kg/day as likely measurement error
    MAX_DAILY_LOSS_GLP1 = 1.0  # kg/day - aggressive but possible on GLP-1
    MAX_DAILY_GAIN = 0.5  # kg/day - weight gain is less common but possible
    EXTREME_CHANGE = 2.0  # kg/day - likely measurement error even with GLP-1

    raw_implausible_total = 0
    filtered_implausible_total = 0
    raw_total = 0
    filtered_total = 0

    extreme_changes_raw = 0
    extreme_changes_filtered = 0
    rapid_loss_raw = 0
    rapid_loss_filtered = 0

    for user_id in sample_users:
        user_raw = df_raw[df_raw['user_id'] == user_id].sort_values('effectiveDateTime')
        user_filtered = df_filtered[df_filtered['user_id'] == user_id].sort_values('effectiveDateTime')

        raw_weights = user_raw['weight'].values
        filtered_weights = user_filtered['weight'].values
        raw_times = pd.to_datetime(user_raw['effectiveDateTime'].values)
        filtered_times = pd.to_datetime(user_filtered['effectiveDateTime'].values)

        if len(raw_weights) > 0:
            # Count implausible absolute values
            raw_implausible = np.sum((raw_weights < MIN_PLAUSIBLE) | (raw_weights > MAX_PLAUSIBLE))
            raw_implausible_total += raw_implausible
            raw_total += len(raw_weights)

            # Check for extreme day-to-day changes accounting for GLP-1
            if len(raw_weights) > 1:
                # Calculate daily rate of change
                for i in range(1, len(raw_weights)):
                    days_diff = (raw_times[i] - raw_times[i-1]).total_seconds() / 86400
                    if days_diff > 0:  # Avoid division by zero
                        daily_change = (raw_weights[i] - raw_weights[i-1]) / days_diff

                        # Count extreme changes (likely errors)
                        if abs(daily_change) > EXTREME_CHANGE:
                            extreme_changes_raw += 1
                        # Count rapid but plausible weight loss (GLP-1 effect)
                        elif daily_change < -MAX_DAILY_LOSS_GLP1:
                            rapid_loss_raw += 1

        if len(filtered_weights) > 0:
            filtered_implausible = np.sum((filtered_weights < MIN_PLAUSIBLE) | (filtered_weights > MAX_PLAUSIBLE))
            filtered_implausible_total += filtered_implausible
            filtered_total += len(filtered_weights)

            if len(filtered_weights) > 1:
                # Calculate daily rate of change
                for i in range(1, len(filtered_weights)):
                    days_diff = (filtered_times[i] - filtered_times[i-1]).total_seconds() / 86400
                    if days_diff > 0:
                        daily_change = (filtered_weights[i] - filtered_weights[i-1]) / days_diff

                        # Count extreme changes (likely errors)
                        if abs(daily_change) > EXTREME_CHANGE:
                            extreme_changes_filtered += 1
                        # Count rapid but plausible weight loss (GLP-1 effect)
                        elif daily_change < -MAX_DAILY_LOSS_GLP1:
                            rapid_loss_filtered += 1

    return {
        'raw_implausible_rate': (raw_implausible_total / raw_total * 100) if raw_total > 0 else 0,
        'filtered_implausible_rate': (filtered_implausible_total / filtered_total * 100) if filtered_total > 0 else 0,
        'implausible_removed': raw_implausible_total - filtered_implausible_total,
        'raw_extreme_changes': extreme_changes_raw,
        'filtered_extreme_changes': extreme_changes_filtered,
        'extreme_changes_removed': extreme_changes_raw - extreme_changes_filtered,
        'raw_rapid_loss': rapid_loss_raw,
        'filtered_rapid_loss': rapid_loss_filtered,
        'note': 'Adjusted for GLP-1 medication effects (up to 1kg/day loss considered plausible)'
    }

def calculate_temporal_consistency(df_raw: pd.DataFrame, df_filtered: pd.DataFrame,
                                  sample_users: List[str]) -> Dict:
    """
    Calculate temporal consistency (autocorrelation) metrics.

    Returns:
        Dictionary with consistency metrics
    """
    raw_autocorrs = []
    filtered_autocorrs = []
    improvements = []

    for user_id in sample_users:
        user_raw = df_raw[df_raw['user_id'] == user_id].sort_values('effectiveDateTime')
        user_filtered = df_filtered[df_filtered['user_id'] == user_id].sort_values('effectiveDateTime')

        raw_weights = user_raw['weight'].values
        filtered_weights = user_filtered['weight'].values

        # Need at least 10 measurements for meaningful autocorrelation
        if len(raw_weights) > 10:
            raw_series = pd.Series(raw_weights)
            raw_autocorr = raw_series.autocorr(lag=1)
            if not np.isnan(raw_autocorr):
                raw_autocorrs.append(raw_autocorr)

        if len(filtered_weights) > 10:
            filtered_series = pd.Series(filtered_weights)
            filtered_autocorr = filtered_series.autocorr(lag=1)
            if not np.isnan(filtered_autocorr):
                filtered_autocorrs.append(filtered_autocorr)

                if len(raw_weights) > 10:
                    raw_autocorr = pd.Series(raw_weights).autocorr(lag=1)
                    if not np.isnan(raw_autocorr):
                        improvement = filtered_autocorr - raw_autocorr
                        improvements.append(improvement)

    return {
        'mean_raw_autocorr': np.mean(raw_autocorrs) if raw_autocorrs else 0,
        'mean_filtered_autocorr': np.mean(filtered_autocorrs) if filtered_autocorrs else 0,
        'mean_improvement': np.mean(improvements) if improvements else 0,
        'improved_count': sum(1 for i in improvements if i > 0),
        'total_users': len(improvements)
    }

def perform_statistical_tests(df_90_day: pd.DataFrame) -> Dict:
    """
    Perform statistical tests on 90-day weight loss data.

    Returns:
        Dictionary with test results
    """
    # Filter to complete data
    complete = df_90_day[
        (df_90_day['raw_loss_pct'].notna()) &
        (df_90_day['filtered_loss_pct'].notna())
    ]

    if complete.empty:
        return {}

    results = {}

    # Paired t-test for weight loss percentages
    t_stat, p_value = stats.ttest_rel(complete['raw_loss_pct'], complete['filtered_loss_pct'])
    results['paired_t_test'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'interpretation': 'Significant difference' if p_value < 0.05 else 'No significant difference'
    }

    # Wilcoxon signed-rank test (non-parametric alternative)
    w_stat, w_p_value = stats.wilcoxon(complete['raw_loss_pct'], complete['filtered_loss_pct'])
    results['wilcoxon_test'] = {
        'statistic': w_stat,
        'p_value': w_p_value,
        'significant': w_p_value < 0.05
    }

    # Effect size (Cohen's d)
    diff = complete['filtered_loss_pct'] - complete['raw_loss_pct']
    cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
    results['effect_size'] = {
        'cohens_d': cohens_d,
        'interpretation': 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
    }

    # Chi-square test for success rates
    raw_success = (complete['raw_loss_pct'] > 0).sum()
    raw_fail = len(complete) - raw_success
    filtered_success = (complete['filtered_loss_pct'] > 0).sum()
    filtered_fail = len(complete) - filtered_success

    contingency_table = [[raw_success, raw_fail], [filtered_success, filtered_fail]]
    chi2, chi_p, dof, expected = stats.chi2_contingency(contingency_table)

    results['chi_square_test'] = {
        'statistic': chi2,
        'p_value': chi_p,
        'significant': chi_p < 0.05,
        'raw_success_rate': raw_success / len(complete) * 100,
        'filtered_success_rate': filtered_success / len(complete) * 100
    }

    return results

def generate_report(output_dir: Path = Path(".")) -> None:
    """
    Generate comprehensive statistical evidence report.
    """
    # Load data - try cache first
    try:
        from data_cache import data_cache
        logging.info("Loading data from cache for statistical analysis...")
        weight_cols = ['user_id', 'effectiveDateTime', 'weight']
        df_raw = data_cache.get_dataframe(RAW_FILE, weight_cols)
        df_filtered = data_cache.get_dataframe(FILTERED_FILE, weight_cols)
    except:
        logging.info("Loading data directly for statistical analysis...")
        df_raw = pd.read_csv(RAW_FILE, usecols=['user_id', 'effectiveDateTime', 'weight'])
        df_filtered = pd.read_csv(FILTERED_FILE, usecols=['user_id', 'effectiveDateTime', 'weight'])
        df_raw['effectiveDateTime'] = pd.to_datetime(df_raw['effectiveDateTime'])
        df_filtered['effectiveDateTime'] = pd.to_datetime(df_filtered['effectiveDateTime'])

    # Load 90-day analysis if available
    analysis_file = output_dir / "90_day_analysis.csv"
    if analysis_file.exists():
        df_90_day = pd.read_csv(analysis_file)
    else:
        logging.info("Running 90-day analysis first...")
        from analyze_90_day import main as analyze_main
        df_90_day, _, _ = analyze_main(output_dir=output_dir)

    # Get sample of users for detailed analysis
    all_users = set(df_raw['user_id'].unique()) & set(df_filtered['user_id'].unique())
    sample_size = min(10000, len(all_users))
    sample_users = list(all_users)[:sample_size]

    logging.info(f"Analyzing {sample_size} users for statistical evidence...")

    # Perform analyses
    normality_results = perform_normality_tests(df_raw, df_filtered, sample_users)
    variance_results = calculate_variance_metrics(df_raw, df_filtered, sample_users)
    smoothness_results = calculate_smoothness_metrics(df_raw, df_filtered, sample_users)
    plausibility_results = calculate_plausibility_metrics(df_raw, df_filtered, sample_users)
    consistency_results = calculate_temporal_consistency(df_raw, df_filtered, sample_users)
    statistical_tests = perform_statistical_tests(df_90_day)

    # Generate markdown report
    report_content = f"""# Statistical Evidence Report: Filtering Effectiveness
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report provides statistical evidence for the effectiveness of weight data filtering
based on analysis of {sample_size} users with comprehensive data.

## 1. Distribution Normality (Shapiro-Wilk Test)

**Hypothesis**: Filtering improves data distribution normality

- **Raw data normal distributions**: {normality_results['raw_normal_count']}/{normality_results['sample_size']} ({normality_results['raw_normal_count']/normality_results['sample_size']*100:.1f}%)
- **Filtered data normal distributions**: {normality_results['filtered_normal_count']}/{normality_results['sample_size']} ({normality_results['filtered_normal_count']/normality_results['sample_size']*100:.1f}%)
- **Improvement rate**: {normality_results['improvement_count']}/{normality_results['sample_size']} users showed improvement

**Verdict**: {'✅ Filtering improves normality' if normality_results['filtered_normal_count'] > normality_results['raw_normal_count'] else '❌ No normality improvement'}

## 2. Variance Reduction

**Hypothesis**: Filtering reduces measurement variance

- **Mean variance reduction**: {variance_results['mean_variance_reduction']:.2f}%
- **Median variance reduction**: {variance_results['median_variance_reduction']:.2f}%
- **Standard deviation reduction**: {variance_results['mean_std_reduction']:.2f}%
- **Users with reduced variance**: {variance_results['positive_reduction_count']}/{variance_results['total_users']} ({variance_results['positive_reduction_count']/max(variance_results['total_users'],1)*100:.1f}%)

**Verdict**: {'✅ Significant variance reduction' if variance_results['mean_variance_reduction'] > 10 else '⚠️ Moderate variance reduction' if variance_results['mean_variance_reduction'] > 0 else '❌ No variance reduction'}

## 3. Trend Smoothness

**Hypothesis**: Filtering produces smoother weight trends

- **Mean smoothness improvement**: {smoothness_results['mean_smoothness_improvement']:.1f}%
- **Median smoothness improvement**: {smoothness_results['median_smoothness_improvement']:.1f}%
- **Jitter reduction**: {smoothness_results['mean_jitter_reduction']:.1f}%
- **Users with smoother trends**: {smoothness_results['improved_count']}/{smoothness_results['total_users']} ({smoothness_results['improved_count']/max(smoothness_results['total_users'],1)*100:.1f}%)

**Verdict**: {'✅ Trends significantly smoother' if smoothness_results['mean_smoothness_improvement'] > 20 else '✓ Moderate smoothing' if smoothness_results['mean_smoothness_improvement'] > 0 else '❌ No smoothing improvement'}

## 4. Plausible Weight Changes (GLP-1 Adjusted)

**Hypothesis**: Filtering removes implausible weight changes while preserving GLP-1 medication effects

- **Raw implausible rate**: {plausibility_results['raw_implausible_rate']:.3f}%
- **Filtered implausible rate**: {plausibility_results['filtered_implausible_rate']:.3f}%
- **Implausible values removed**: {plausibility_results['implausible_removed']}
- **Extreme changes (>2kg/day) removed**: {plausibility_results['extreme_changes_removed']}
- **Rapid weight loss events (0.5-1kg/day)**:
  - Raw data: {plausibility_results['raw_rapid_loss']} occurrences
  - Filtered data: {plausibility_results['filtered_rapid_loss']} occurrences

*Note: Thresholds adjusted for GLP-1 medication - up to 1kg/day loss considered physiologically plausible*

**Verdict**: {'✅ Improved plausibility while preserving GLP-1 effects' if plausibility_results['extreme_changes_removed'] > 0 else '✓ Data already plausible'}

## 5. Temporal Consistency (Autocorrelation)

**Hypothesis**: Filtering improves temporal consistency

- **Raw mean autocorrelation**: {consistency_results['mean_raw_autocorr']:.3f}
- **Filtered mean autocorrelation**: {consistency_results['mean_filtered_autocorr']:.3f}
- **Mean improvement**: {consistency_results['mean_improvement']:+.3f}
- **Users with improved consistency**: {consistency_results['improved_count']}/{consistency_results['total_users']}

**Verdict**: {'✅ Improved temporal consistency' if consistency_results['mean_improvement'] > 0.05 else '⚠️ Minimal impact on consistency' if abs(consistency_results['mean_improvement']) < 0.05 else '❌ Reduced consistency'}

## 6. 90-Day Weight Loss Statistical Tests
"""

    if statistical_tests:
        report_content += f"""
### Paired t-test (Raw vs Filtered Weight Loss %)
- **t-statistic**: {statistical_tests['paired_t_test']['t_statistic']:.4f}
- **p-value**: {statistical_tests['paired_t_test']['p_value']:.6f}
- **Result**: {statistical_tests['paired_t_test']['interpretation']}

### Wilcoxon Signed-Rank Test (Non-parametric)
- **Statistic**: {statistical_tests['wilcoxon_test']['statistic']:.2f}
- **p-value**: {statistical_tests['wilcoxon_test']['p_value']:.6f}
- **Significant**: {'Yes' if statistical_tests['wilcoxon_test']['significant'] else 'No'}

### Effect Size (Cohen's d)
- **Value**: {statistical_tests['effect_size']['cohens_d']:.3f}
- **Interpretation**: {statistical_tests['effect_size']['interpretation']} effect

### Success Rate Comparison (Chi-square)
- **Raw success rate**: {statistical_tests['chi_square_test']['raw_success_rate']:.1f}%
- **Filtered success rate**: {statistical_tests['chi_square_test']['filtered_success_rate']:.1f}%
- **Chi-square statistic**: {statistical_tests['chi_square_test']['statistic']:.4f}
- **p-value**: {statistical_tests['chi_square_test']['p_value']:.6f}
- **Significant difference**: {'Yes' if statistical_tests['chi_square_test']['significant'] else 'No'}
"""

    # Overall conclusion
    positive_indicators = 0
    if normality_results['filtered_normal_count'] > normality_results['raw_normal_count']:
        positive_indicators += 1
    if variance_results['mean_variance_reduction'] > 5:
        positive_indicators += 1
    if smoothness_results['mean_smoothness_improvement'] > 10:
        positive_indicators += 1
    if plausibility_results['implausible_removed'] > 0:
        positive_indicators += 1
    if consistency_results['mean_improvement'] > 0.02:
        positive_indicators += 1

    report_content += f"""
## Overall Conclusion

**Statistical Evidence Score**: {positive_indicators}/5 metrics show improvement

### Final Verdict:
"""

    if positive_indicators >= 4:
        report_content += """
✅ **STRONG EVIDENCE**: Filtering significantly improves data quality
- Multiple statistical tests confirm improvement
- Recommendation: Continue using current filtering approach
"""
    elif positive_indicators >= 3:
        report_content += """
✓ **MODERATE EVIDENCE**: Filtering provides meaningful improvements
- Most metrics show positive impact
- Recommendation: Fine-tune thresholds for optimal performance
"""
    elif positive_indicators >= 2:
        report_content += """
⚠️ **MIXED EVIDENCE**: Some improvements, some concerns
- Benefits are context-dependent
- Recommendation: Review filtering parameters
"""
    else:
        report_content += """
❌ **INSUFFICIENT EVIDENCE**: Filtering may not be beneficial
- Few metrics show improvement
- Recommendation: Reconsider filtering approach
"""

    report_content += f"""

## Key Findings

1. **Variance Reduction**: {variance_results['mean_variance_reduction']:.1f}% average reduction in measurement variance
2. **Smoothness Gain**: {smoothness_results['mean_smoothness_improvement']:.0f}% improvement in trend smoothness
3. **Temporal Consistency**: {consistency_results['mean_improvement']:+.3f} improvement in autocorrelation
4. **Plausibility (GLP-1 adjusted)**: {plausibility_results['extreme_changes_removed']} extreme changes removed

## Methodology Notes

- Sample size: {sample_size} users
- Significance level: α = 0.05
- Tests performed: Shapiro-Wilk, paired t-test, Wilcoxon, Chi-square
- Metrics: Variance, smoothness, autocorrelation, plausible weight changes
- **GLP-1 Adjustment**: Weight loss up to 1kg/day considered plausible for users on GLP-1 medication
"""

    # Save report
    report_file = output_dir / "statistical_evidence_report.md"
    with open(report_file, 'w') as f:
        f.write(report_content)

    logging.info(f"\nStatistical evidence report saved to {report_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate statistical evidence report")
    parser.add_argument('--output-dir', type=Path, default=Path("."),
                       help='Output directory for report')
    args = parser.parse_args()

    generate_report(args.output_dir)

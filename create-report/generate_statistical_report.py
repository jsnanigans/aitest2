#!/usr/bin/env python3
"""
Statistical Evidence Report Generator
Performs statistical tests to validate filtering effectiveness
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

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

def _process_normality_test_batch(user_batch: List[str], df_raw: pd.DataFrame,
                                  df_filtered: pd.DataFrame) -> Dict:
    """Process a batch of users for normality testing."""
    batch_results = {
        'raw_normal_count': 0,
        'filtered_normal_count': 0,
        'both_normal_count': 0,
        'improvement_count': 0,
        'p_values_raw': [],
        'p_values_filtered': [],
        'sample_size': 0
    }

    for user_id in user_batch:
        user_raw = df_raw[df_raw['user_id'] == user_id]['weight'].values
        user_filtered = df_filtered[df_filtered['user_id'] == user_id]['weight'].values

        if len(user_raw) >= 3 and len(user_filtered) >= 3:
            try:
                # Perform Shapiro-Wilk test
                stat_raw, p_raw = stats.shapiro(user_raw) if len(user_raw) <= 5000 else (0, 0)
                stat_filtered, p_filtered = stats.shapiro(user_filtered) if len(user_filtered) <= 5000 else (0, 0)

                batch_results['p_values_raw'].append(p_raw)
                batch_results['p_values_filtered'].append(p_filtered)
                batch_results['sample_size'] += 1

                # Count normal distributions (p > 0.05)
                if p_raw > 0.05:
                    batch_results['raw_normal_count'] += 1
                if p_filtered > 0.05:
                    batch_results['filtered_normal_count'] += 1
                if p_raw > 0.05 and p_filtered > 0.05:
                    batch_results['both_normal_count'] += 1
                if p_filtered > p_raw:
                    batch_results['improvement_count'] += 1

            except Exception as e:
                logging.debug(f"Normality test failed for user {user_id}: {e}")
                continue

    return batch_results


def perform_normality_tests(df_raw: pd.DataFrame, df_filtered: pd.DataFrame,
                           sample_users: List[str]) -> Dict:
    """
    Perform Shapiro-Wilk normality tests on weight distributions with parallel processing.

    Returns:
        Dictionary with test results
    """
    start_time = time.time()

    results = {
        'raw_normal_count': 0,
        'filtered_normal_count': 0,
        'both_normal_count': 0,
        'improvement_count': 0,
        'p_values_raw': [],
        'p_values_filtered': [],
        'sample_size': 0
    }

    # Determine optimal number of workers
    n_workers = min(os.cpu_count() or 4, 8)
    batch_size = max(10, len(sample_users) // (n_workers * 4))

    # Split users into batches
    user_batches = [sample_users[i:i + batch_size]
                    for i in range(0, len(sample_users), batch_size)]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit batch processing tasks
        futures = {
            executor.submit(_process_normality_test_batch, batch, df_raw, df_filtered): batch
            for batch in user_batches
        }

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                # Aggregate results
                results['raw_normal_count'] += batch_results['raw_normal_count']
                results['filtered_normal_count'] += batch_results['filtered_normal_count']
                results['both_normal_count'] += batch_results['both_normal_count']
                results['improvement_count'] += batch_results['improvement_count']
                results['p_values_raw'].extend(batch_results['p_values_raw'])
                results['p_values_filtered'].extend(batch_results['p_values_filtered'])
                results['sample_size'] += batch_results['sample_size']
            except Exception as e:
                logging.error(f"Batch processing failed: {e}")

    elapsed = time.time() - start_time
    logging.debug(f"Normality tests completed in {elapsed:.2f}s (parallel)")

    return results

def _process_variance_batch(user_batch: List[str], df_raw: pd.DataFrame,
                            df_filtered: pd.DataFrame) -> Dict:
    """Process a batch of users for variance metrics."""
    variance_reductions = []
    std_reductions = []

    for user_id in user_batch:
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
        'variance_reductions': variance_reductions,
        'std_reductions': std_reductions
    }


def calculate_variance_metrics(df_raw: pd.DataFrame, df_filtered: pd.DataFrame,
                              sample_users: List[str]) -> Dict:
    """
    Calculate variance reduction metrics with parallel processing.

    Returns:
        Dictionary with variance statistics
    """
    start_time = time.time()

    all_variance_reductions = []
    all_std_reductions = []

    # Determine optimal number of workers
    n_workers = min(os.cpu_count() or 4, 8)
    batch_size = max(10, len(sample_users) // (n_workers * 4))

    # Split users into batches
    user_batches = [sample_users[i:i + batch_size]
                    for i in range(0, len(sample_users), batch_size)]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit batch processing tasks
        futures = {
            executor.submit(_process_variance_batch, batch, df_raw, df_filtered): batch
            for batch in user_batches
        }

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                all_variance_reductions.extend(batch_results['variance_reductions'])
                all_std_reductions.extend(batch_results['std_reductions'])
            except Exception as e:
                logging.error(f"Variance batch processing failed: {e}")

    elapsed = time.time() - start_time
    logging.debug(f"Variance metrics completed in {elapsed:.2f}s (parallel)")

    return {
        'mean_variance_reduction': np.mean(all_variance_reductions) if all_variance_reductions else 0,
        'median_variance_reduction': np.median(all_variance_reductions) if all_variance_reductions else 0,
        'mean_std_reduction': np.mean(all_std_reductions) if all_std_reductions else 0,
        'positive_reduction_count': sum(1 for v in all_variance_reductions if v > 0),
        'total_users': len(all_variance_reductions)
    }

def _process_smoothness_batch(user_batch: List[str], df_raw: pd.DataFrame,
                              df_filtered: pd.DataFrame) -> Dict:
    """Process a batch of users for smoothness metrics."""
    smoothness_improvements = []
    jitter_reductions = []

    for user_id in user_batch:
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
        'smoothness_improvements': smoothness_improvements,
        'jitter_reductions': jitter_reductions
    }


def calculate_smoothness_metrics(df_raw: pd.DataFrame, df_filtered: pd.DataFrame,
                                sample_users: List[str]) -> Dict:
    """
    Calculate trend smoothness improvements with parallel processing.

    Returns:
        Dictionary with smoothness metrics
    """
    start_time = time.time()

    all_smoothness_improvements = []
    all_jitter_reductions = []

    # Determine optimal number of workers
    n_workers = min(os.cpu_count() or 4, 8)
    batch_size = max(10, len(sample_users) // (n_workers * 4))

    # Split users into batches
    user_batches = [sample_users[i:i + batch_size]
                    for i in range(0, len(sample_users), batch_size)]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit batch processing tasks
        futures = {
            executor.submit(_process_smoothness_batch, batch, df_raw, df_filtered): batch
            for batch in user_batches
        }

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                all_smoothness_improvements.extend(batch_results['smoothness_improvements'])
                all_jitter_reductions.extend(batch_results['jitter_reductions'])
            except Exception as e:
                logging.error(f"Smoothness batch processing failed: {e}")

    elapsed = time.time() - start_time
    logging.debug(f"Smoothness metrics completed in {elapsed:.2f}s (parallel)")

    return {
        'mean_smoothness_improvement': np.mean(all_smoothness_improvements) if all_smoothness_improvements else 0,
        'median_smoothness_improvement': np.median(all_smoothness_improvements) if all_smoothness_improvements else 0,
        'mean_jitter_reduction': np.mean(all_jitter_reductions) if all_jitter_reductions else 0,
        'improved_count': sum(1 for s in all_smoothness_improvements if s > 0),
        'total_users': len(all_smoothness_improvements)
    }

def _process_plausibility_batch(user_batch: List[str], df_raw: pd.DataFrame,
                                df_filtered: pd.DataFrame) -> Dict:
    """Process a batch of users for plausibility metrics."""
    # Define plausible weight range (kg)
    MIN_PLAUSIBLE = 30
    MAX_PLAUSIBLE = 300

    # GLP-1 adjusted limits for weight changes
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

    for user_id in user_batch:
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
        'raw_implausible_total': raw_implausible_total,
        'filtered_implausible_total': filtered_implausible_total,
        'raw_total': raw_total,
        'filtered_total': filtered_total,
        'extreme_changes_raw': extreme_changes_raw,
        'extreme_changes_filtered': extreme_changes_filtered,
        'rapid_loss_raw': rapid_loss_raw,
        'rapid_loss_filtered': rapid_loss_filtered
    }


def calculate_plausibility_metrics(df_raw: pd.DataFrame, df_filtered: pd.DataFrame,
                                  sample_users: List[str]) -> Dict:
    """
    Calculate plausible weight change improvements with parallel processing.
    Accounts for GLP-1 medication effects on weight loss rates.

    Returns:
        Dictionary with plausibility metrics
    """
    start_time = time.time()

    # Aggregate results
    total_raw_implausible = 0
    total_filtered_implausible = 0
    total_raw = 0
    total_filtered = 0
    total_extreme_changes_raw = 0
    total_extreme_changes_filtered = 0
    total_rapid_loss_raw = 0
    total_rapid_loss_filtered = 0

    # Determine optimal number of workers
    n_workers = min(os.cpu_count() or 4, 8)
    batch_size = max(10, len(sample_users) // (n_workers * 4))

    # Split users into batches
    user_batches = [sample_users[i:i + batch_size]
                    for i in range(0, len(sample_users), batch_size)]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit batch processing tasks
        futures = {
            executor.submit(_process_plausibility_batch, batch, df_raw, df_filtered): batch
            for batch in user_batches
        }

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                total_raw_implausible += batch_results['raw_implausible_total']
                total_filtered_implausible += batch_results['filtered_implausible_total']
                total_raw += batch_results['raw_total']
                total_filtered += batch_results['filtered_total']
                total_extreme_changes_raw += batch_results['extreme_changes_raw']
                total_extreme_changes_filtered += batch_results['extreme_changes_filtered']
                total_rapid_loss_raw += batch_results['rapid_loss_raw']
                total_rapid_loss_filtered += batch_results['rapid_loss_filtered']
            except Exception as e:
                logging.error(f"Plausibility batch processing failed: {e}")

    elapsed = time.time() - start_time
    logging.debug(f"Plausibility metrics completed in {elapsed:.2f}s (parallel)")

    return {
        'raw_implausible_rate': (total_raw_implausible / total_raw * 100) if total_raw > 0 else 0,
        'filtered_implausible_rate': (total_filtered_implausible / total_filtered * 100) if total_filtered > 0 else 0,
        'implausible_removed': total_raw_implausible - total_filtered_implausible,
        'raw_extreme_changes': total_extreme_changes_raw,
        'filtered_extreme_changes': total_extreme_changes_filtered,
        'extreme_changes_removed': total_extreme_changes_raw - total_extreme_changes_filtered,
        'raw_rapid_loss': total_rapid_loss_raw,
        'filtered_rapid_loss': total_rapid_loss_filtered,
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

    # Track overall performance
    overall_start = time.time()

    # Perform analyses with timing
    test_start = time.time()
    normality_results = perform_normality_tests(df_raw, df_filtered, sample_users)
    logging.info(f"  Normality tests: {time.time() - test_start:.2f}s")

    test_start = time.time()
    variance_results = calculate_variance_metrics(df_raw, df_filtered, sample_users)
    logging.info(f"  Variance metrics: {time.time() - test_start:.2f}s")

    test_start = time.time()
    smoothness_results = calculate_smoothness_metrics(df_raw, df_filtered, sample_users)
    logging.info(f"  Smoothness metrics: {time.time() - test_start:.2f}s")

    test_start = time.time()
    plausibility_results = calculate_plausibility_metrics(df_raw, df_filtered, sample_users)
    logging.info(f"  Plausibility metrics: {time.time() - test_start:.2f}s")

    test_start = time.time()
    consistency_results = calculate_temporal_consistency(df_raw, df_filtered, sample_users)
    logging.info(f"  Temporal consistency: {time.time() - test_start:.2f}s")

    test_start = time.time()
    statistical_tests = perform_statistical_tests(df_90_day)
    logging.info(f"  Statistical tests: {time.time() - test_start:.2f}s")

    logging.info(f"Total analysis time: {time.time() - overall_start:.2f}s")

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

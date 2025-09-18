#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import argparse
from typing import Optional, Set, Union, Dict, Tuple, List
from datetime import datetime
import time
import logging
import numpy as np
from scipy import stats

# ============= CONFIGURATION =============
# File paths
DATA_DIR = Path("data")
FILTERED_FILE = Path("filtered.csv")
RAW_FILE = DATA_DIR / "2025-09-05_nocon.csv"
PARTNERS_FILE = DATA_DIR / "partners.csv"
USER_EMPLOYERS_FILE = DATA_DIR / "2025-09-17-user-employers.csv"

# Performance settings
CHUNKSIZE = 10000  # For reading large files in chunks if needed
MAX_DAYS_WEIGHT_WINDOW = 10  # Days to search for closest weight measurement

# Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_employer_filter(employer_name: str) -> Tuple[Set[str], Dict[str, str]]:
    """Load user IDs and start dates for a specific employer (matching report.py logic)

    Returns:
        Tuple of (set of user IDs, dict of user_id -> start_date)
    """
    # Load partners CSV to get employer_id from name
    if not PARTNERS_FILE.exists():
        return set(), {}

    partners_df = pd.read_csv(PARTNERS_FILE, usecols=['id', 'name'])
    employer_rows = partners_df[partners_df['name'] == employer_name]

    if employer_rows.empty:
        available_employers = partners_df[partners_df['name'].str.contains('_EMPLOYER', na=False)]['name'].unique()
        logging.error(f"Employer '{employer_name}' not found. Available employers: {', '.join(sorted(available_employers))}")
        return set(), {}

    employer_id = employer_rows.iloc[0]['id']

    # Load user-employer mappings
    if not USER_EMPLOYERS_FILE.exists():
        return set(), {}

    user_employers_df = pd.read_csv(USER_EMPLOYERS_FILE, usecols=['user_id', 'employer_id', 'start_date'])
    employer_data = user_employers_df[user_employers_df['employer_id'] == employer_id]

    # Create user ID set and start date mapping
    user_ids = set(employer_data['user_id'].values)
    user_start_dates = dict(zip(employer_data['user_id'], employer_data['start_date']))

    return user_ids, user_start_dates

def get_closest_weight(df: pd.DataFrame, target_date: Union[str, datetime], max_days: int = MAX_DAYS_WEIGHT_WINDOW) -> Optional[float]:
    """
    Find the weight measurement closest to the target date within a time window.

    Args:
        df: DataFrame with 'effectiveDateTime' and 'weight' columns
        target_date: Target date as string or datetime object
        max_days: Maximum days before/after target date to search (default 10)

    Returns:
        The weight value closest to the target date within the window, or None if no data
    """
    if df.empty:
        return None

    # Convert target_date to datetime if string
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)

    # Make a copy to avoid modifying original
    df_copy = df.copy()

    # Ensure required columns exist
    if 'effectiveDateTime' not in df_copy.columns or 'weight' not in df_copy.columns:
        return None

    df_copy.loc[:, 'effectiveDateTime'] = pd.to_datetime(df_copy['effectiveDateTime'], errors='coerce')

    # Calculate time difference from target date
    df_copy.loc[:, 'time_diff'] = abs(df_copy['effectiveDateTime'] - target_date)

    # Filter to only include measurements within the time window (±max_days)
    time_window = pd.Timedelta(days=max_days)
    df_window = df_copy[df_copy['time_diff'] <= time_window]

    if df_window.empty:
        return None

    # Find the row with minimum time difference within the window
    closest_idx = df_window['time_diff'].idxmin()

    if pd.isna(closest_idx):
        return None

    return df_window.loc[closest_idx, 'weight']

def load_data_efficient(filtered_path: Path, raw_path: Path, user_ids: Optional[Set[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data efficiently with optional user filtering."""
    start_time = time.time()

    # Determine columns to load
    filtered_cols = ['user_id', 'effectiveDateTime', 'weight']
    raw_cols = ['user_id', 'effectiveDateTime', 'weight']

    # Load with user filtering if provided
    if user_ids:
        # For large datasets, could use chunked reading
        df_filtered = pd.read_csv(filtered_path, usecols=filtered_cols)
        df_filtered = df_filtered[df_filtered['user_id'].isin(user_ids)]

        df_raw = pd.read_csv(raw_path, usecols=raw_cols)
        df_raw = df_raw[df_raw['user_id'].isin(user_ids)]
    else:
        df_filtered = pd.read_csv(filtered_path, usecols=filtered_cols)
        df_raw = pd.read_csv(raw_path, usecols=raw_cols)

    # Convert datetime columns once
    df_filtered['effectiveDateTime'] = pd.to_datetime(df_filtered['effectiveDateTime'], errors='coerce')
    df_raw['effectiveDateTime'] = pd.to_datetime(df_raw['effectiveDateTime'], errors='coerce')

    return df_filtered, df_raw

def analyze_filtering_patterns(df_raw: pd.DataFrame, df_filtered: pd.DataFrame, user_ids: Set[str]) -> None:
    """
    Analyze patterns in what data gets filtered out.
    
    Args:
        df_raw: Raw data DataFrame
        df_filtered: Filtered data DataFrame
        user_ids: Set of user IDs to analyze
    """
    logging.info(f"\n=== FILTERING PATTERNS ===")
    
    # Analyze per-user filtering rates
    user_filtering_rates = []
    high_filter_users = []
    no_filter_users = []
    
    for user_id in user_ids:
        raw_count = len(df_raw[df_raw['user_id'] == user_id])
        filtered_count = len(df_filtered[df_filtered['user_id'] == user_id])
        
        if raw_count > 0:
            filter_rate = (raw_count - filtered_count) / raw_count * 100
            user_filtering_rates.append(filter_rate)
            
            if filter_rate > 50:
                high_filter_users.append((user_id, filter_rate))
            elif filter_rate == 0:
                no_filter_users.append(user_id)
    
    if user_filtering_rates:
        avg_filter_rate = sum(user_filtering_rates) / len(user_filtering_rates)
        median_filter_rate = sorted(user_filtering_rates)[len(user_filtering_rates)//2]
        
        logging.info(f"📊 Per-User Filtering:")
        logging.info(f"  • Average: {avg_filter_rate:.1f}% of measurements removed")
        logging.info(f"  • Median: {median_filter_rate:.1f}% removed")
        logging.info(f"  • Users with >50% filtered: {len(high_filter_users)}")
        logging.info(f"  • Users with no filtering: {len(no_filter_users)}")
        
        # Identify filtering distribution
        low_filter = sum(1 for r in user_filtering_rates if r < 10)
        moderate_filter = sum(1 for r in user_filtering_rates if 10 <= r < 30)
        high_filter = sum(1 for r in user_filtering_rates if r >= 30)
        
        logging.info(f"\n📈 Filtering Distribution:")
        logging.info(f"  • Light (<10%): {low_filter} users")
        logging.info(f"  • Moderate (10-30%): {moderate_filter} users")
        logging.info(f"  • Heavy (>30%): {high_filter} users")

def calculate_interval_analysis(user_data: Dict, start_weights: Dict, interval_days: int = 30, max_days: int = 360) -> pd.DataFrame:
    """
    Calculate weight loss analysis at regular intervals from start date.
    
    Args:
        user_data: Dict of user_id -> {'filtered': df, 'raw': df}
        start_weights: Dict of user_id -> {'start_date', 'raw_start_weight', 'filtered_start_weight'}
        interval_days: Days between each measurement point (default 30)
        max_days: Maximum days to analyze from start date (default 360)
    
    Returns:
        DataFrame with interval analysis results
    """
    results = []
    
    for user_id, weights_info in start_weights.items():
        if user_id not in user_data:
            continue
            
        start_date = weights_info.get('start_date')
        raw_start = weights_info.get('raw_start_weight')
        filtered_start = weights_info.get('filtered_start_weight')
        
        if not start_date:
            continue
            
        # Convert start_date to datetime
        start_dt = pd.to_datetime(start_date)
        
        # Calculate weight at each interval
        for days in range(interval_days, max_days + 1, interval_days):
            target_date = start_dt + pd.Timedelta(days=days)
            
            # Get weights at this interval
            raw_weight = get_closest_weight(user_data[user_id]['raw'], target_date)
            filtered_weight = get_closest_weight(user_data[user_id]['filtered'], target_date)
            
            # Calculate weight loss (negative means weight gain)
            raw_loss = None
            filtered_loss = None
            raw_loss_pct = None
            filtered_loss_pct = None
            
            if raw_start and raw_weight:
                raw_loss = raw_start - raw_weight
                raw_loss_pct = (raw_loss / raw_start) * 100
                
            if filtered_start and filtered_weight:
                filtered_loss = filtered_start - filtered_weight
                filtered_loss_pct = (filtered_loss / filtered_start) * 100
            
            results.append({
                'user_id': user_id,
                'interval_days': days,
                'raw_start_weight': raw_start,
                'raw_weight': raw_weight,
                'raw_weight_loss': raw_loss,
                'raw_loss_pct': raw_loss_pct,
                'filtered_start_weight': filtered_start,
                'filtered_weight': filtered_weight,
                'filtered_weight_loss': filtered_loss,
                'filtered_loss_pct': filtered_loss_pct
            })
    
    return pd.DataFrame(results)

def analyze_interval_results(df_intervals: pd.DataFrame) -> None:
    """
    Analyze and print summary statistics for interval weight loss data.
    
    Args:
        df_intervals: DataFrame from calculate_interval_analysis
    """
    if df_intervals.empty:
        logging.warning("No interval data to analyze")
        return
    
    # Create summary comparison table
    summary_data = []
    all_success_rates = []
    
    # For each interval, calculate statistics
    for interval in sorted(df_intervals['interval_days'].unique()):
        interval_data = df_intervals[df_intervals['interval_days'] == interval]
        
        # Count users with data at this interval
        both_count = (interval_data['raw_weight'].notna() & interval_data['filtered_weight'].notna()).sum()
        
        # Calculate weight loss statistics for users with data
        raw_loss_data = interval_data[interval_data['raw_loss_pct'].notna()]['raw_loss_pct']
        filtered_loss_data = interval_data[interval_data['filtered_loss_pct'].notna()]['filtered_loss_pct']
        
        # Calculate success rate (% who lost weight)
        if not filtered_loss_data.empty:
            success_rate = (filtered_loss_data > 0).sum() / len(filtered_loss_data) * 100
            all_success_rates.append((interval, success_rate))
        
        # Compare raw vs filtered for users with both
        both_data = interval_data[(interval_data['raw_loss_pct'].notna()) & (interval_data['filtered_loss_pct'].notna())]
        if not both_data.empty:
            diff = both_data['filtered_loss_pct'] - both_data['raw_loss_pct']
            
            # Add to summary
            summary_data.append({
                'Day': interval,
                'Raw Avg Loss %': raw_loss_data.mean() if not raw_loss_data.empty else None,
                'Filtered Avg Loss %': filtered_loss_data.mean() if not filtered_loss_data.empty else None,
                'Success Rate': success_rate if not filtered_loss_data.empty else None,
                'Difference %': diff.mean() if not both_data.empty else None,
                'Users w/ Data': both_count
            })
    
    # Print summary table with raw vs filtered comparison
    if summary_data:
        logging.info("\n=== WEIGHT LOSS PROGRESSION (RAW vs FILTERED) ===")
        logging.info(f"{'Day':<6} {'Raw Loss %':<12} {'Filtered Loss %':<16} {'Diff':<8} {'Success':<10} {'Users':<8}")
        logging.info("-" * 70)
        
        for row in summary_data:
            raw_loss = f"{row['Raw Avg Loss %']:.1f}" if row['Raw Avg Loss %'] is not None else "N/A"
            filtered_loss = f"{row['Filtered Avg Loss %']:.1f}" if row['Filtered Avg Loss %'] is not None else "N/A"
            diff = f"{row['Difference %']:+.1f}" if row['Difference %'] is not None else "N/A"
            success = f"{row['Success Rate']:.0f}%" if row['Success Rate'] is not None else "N/A"
            logging.info(f"{row['Day']:<6} {raw_loss:<12} {filtered_loss:<16} {diff:<8} {success:<10} {row['Users w/ Data']:<8}")
        
        # Calculate advanced insights
        logging.info("\n=== KEY INSIGHTS ===")
        
        # Weight loss velocity (rate of change)
        velocities = []
        for i in range(1, len(summary_data)):
            if summary_data[i]['Filtered Avg Loss %'] is not None and summary_data[i-1]['Filtered Avg Loss %'] is not None:
                days_diff = summary_data[i]['Day'] - summary_data[i-1]['Day']
                loss_diff = summary_data[i]['Filtered Avg Loss %'] - summary_data[i-1]['Filtered Avg Loss %']
                velocity = loss_diff / days_diff * 30  # Normalize to 30-day rate
                velocities.append((summary_data[i]['Day'], velocity))
        
        if velocities:
            max_velocity = max(velocities, key=lambda x: x[1])
            logging.info(f"📈 Fastest weight loss: {max_velocity[1]:.2f}% per month (day {max_velocity[0]})")
            
            # Detect plateau (velocity near zero)
            plateaus = [v for v in velocities if abs(v[1]) < 0.5]
            if plateaus:
                logging.info(f"📊 Weight loss plateau detected around day {plateaus[0][0]}")
        
        # Success rate analysis
        if all_success_rates:
            avg_success = sum(sr[1] for sr in all_success_rates) / len(all_success_rates)
            best_period = max(all_success_rates, key=lambda x: x[1])
            logging.info(f"✓ Average success rate: {avg_success:.0f}% of users losing weight")
            logging.info(f"🏆 Best period: Day {best_period[0]} ({best_period[1]:.0f}% success rate)")
        
        # Weight loss milestones
        milestones = []
        for row in summary_data:
            if row['Filtered Avg Loss %'] is not None:
                if row['Filtered Avg Loss %'] >= 5 and not any(m[1] == 5 for m in milestones):
                    milestones.append((row['Day'], 5))
                if row['Filtered Avg Loss %'] >= 10 and not any(m[1] == 10 for m in milestones):
                    milestones.append((row['Day'], 10))
        
        if milestones:
            for day, pct in milestones:
                logging.info(f"🎯 {pct}% average loss achieved by day {day}")
        
        # Retention analysis
        initial_users = summary_data[0]['Users w/ Data'] if summary_data else 0
        if initial_users > 0:
            retention_90 = next((row['Users w/ Data'] for row in summary_data if row['Day'] == 90), 0)
            retention_180 = next((row['Users w/ Data'] for row in summary_data if row['Day'] == 180), 0)
            retention_360 = summary_data[-1]['Users w/ Data'] if summary_data[-1]['Day'] == 360 else 0
            
            logging.info(f"\n📊 DATA RETENTION:")
            logging.info(f"  • 90 days: {retention_90}/{initial_users} ({retention_90/initial_users*100:.0f}%)")
            logging.info(f"  • 180 days: {retention_180}/{initial_users} ({retention_180/initial_users*100:.0f}%)")
            logging.info(f"  • 360 days: {retention_360}/{initial_users} ({retention_360/initial_users*100:.0f}%)")
        
        # Raw vs Filtered comparison insights
        logging.info(f"\n=== RAW vs FILTERED INSIGHTS ===")
        
        # Calculate consistency metrics
        diffs = [row['Difference %'] for row in summary_data if row['Difference %'] is not None]
        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            max_diff = max(abs(d) for d in diffs)
            
            # Check if filtering improves or worsens results
            if avg_diff > 0:
                logging.info(f"📈 Filtering shows BETTER weight loss (+{avg_diff:.1f}% on average)")
                logging.info(f"   → Outliers were likely masking true progress")
            elif avg_diff < -0.5:
                logging.info(f"📉 Filtering shows LESS weight loss ({avg_diff:.1f}% on average)")
                logging.info(f"   → Some legitimate measurements may be filtered")
            else:
                logging.info(f"✅ Raw and filtered data are highly consistent (±{abs(avg_diff):.1f}%)")
                logging.info(f"   → Filtering is removing noise without changing trends")
            
            # Identify periods where filtering matters most
            high_impact_periods = [(row['Day'], row['Difference %']) 
                                  for row in summary_data 
                                  if row['Difference %'] is not None and abs(row['Difference %']) > 1.0]
            
            if high_impact_periods:
                logging.info(f"\n🔍 High-impact filtering periods:")
                for day, diff in high_impact_periods[:3]:  # Show top 3
                    impact = "improves" if diff > 0 else "reduces"
                    logging.info(f"  • Day {day}: Filtering {impact} loss by {abs(diff):.1f}%")
            
            # Overall assessment
            if max_diff < 0.5:
                logging.info(f"\n✅ FILTERING IMPACT: Minimal")
                logging.info(f"  • Max deviation: {max_diff:.2f}%")
                logging.info(f"  • Data quality is high, minimal outliers")
            elif max_diff < 2.0:
                logging.info(f"\n📊 FILTERING IMPACT: Moderate")
                logging.info(f"  • Max deviation: {max_diff:.2f}%")
                logging.info(f"  • Filtering effectively removes noise")
            else:
                logging.info(f"\n⚠️  FILTERING IMPACT: Significant")
                logging.info(f"  • Max deviation: {max_diff:.2f}%")
                logging.info(f"  • Consider reviewing outlier thresholds")

def analyze_outlier_effectiveness(df_raw: pd.DataFrame, df_filtered: pd.DataFrame, user_ids: Set[str]) -> None:
    """
    Analyze the effectiveness of outlier filtering by comparing raw vs filtered data quality.
    
    Tests the hypothesis that filtering improves clinical reliability through:
    1. Distribution normality (Shapiro-Wilk test)
    2. Variance reduction analysis
    3. Trend smoothness measurement
    4. Clinical plausibility checks
    5. Temporal consistency metrics
    
    Args:
        df_raw: Raw weight data
        df_filtered: Filtered weight data  
        user_ids: Set of user IDs to analyze
    """
    logging.info("\n=== OUTLIER FILTERING EFFECTIVENESS ANALYSIS ===")
    
    # Initialize results tracking
    normality_improvements = []
    variance_reductions = []
    smoothness_gains = []
    plausibility_improvements = []
    consistency_scores = []
    
    # Sample users for detailed analysis (limit for performance)
    sample_size = min(100, len(user_ids))
    sample_users = list(user_ids)[:sample_size] if len(user_ids) > sample_size else list(user_ids)
    
    for user_id in sample_users:
        user_raw = df_raw[df_raw['user_id'] == user_id].copy()
        user_filtered = df_filtered[df_filtered['user_id'] == user_id].copy()
        
        # Skip users with insufficient data
        if len(user_raw) < 10 or len(user_filtered) < 5:
            continue
            
        # Sort by datetime for temporal analysis
        user_raw = user_raw.sort_values('effectiveDateTime')
        user_filtered = user_filtered.sort_values('effectiveDateTime')
        
        raw_weights = user_raw['weight'].values
        filtered_weights = user_filtered['weight'].values
        
        # 1. DISTRIBUTION NORMALITY TEST (Shapiro-Wilk)
        if len(raw_weights) >= 3 and len(filtered_weights) >= 3:
            try:
                # Test for normal distribution (p > 0.05 suggests normality)
                _, p_raw = stats.shapiro(raw_weights) if len(raw_weights) <= 5000 else (0, 0)
                _, p_filtered = stats.shapiro(filtered_weights) if len(filtered_weights) <= 5000 else (0, 0)
                
                # Score improvement (higher p-value = more normal)
                if p_raw > 0 and p_filtered > 0:
                    normality_improvements.append((p_filtered - p_raw) / max(p_raw, 0.001))
            except:
                pass  # Skip on numerical errors
        
        # 2. VARIANCE REDUCTION
        if len(raw_weights) > 1 and len(filtered_weights) > 1:
            var_raw = np.var(raw_weights)
            var_filtered = np.var(filtered_weights)
            
            if var_raw > 0:
                # Percentage reduction in variance (positive = improvement)
                variance_reduction = (var_raw - var_filtered) / var_raw * 100
                variance_reductions.append(variance_reduction)
        
        # 3. TREND SMOOTHNESS (using first differences)
        if len(raw_weights) > 2 and len(filtered_weights) > 2:
            # Calculate day-to-day changes
            raw_diffs = np.diff(raw_weights)
            filtered_diffs = np.diff(filtered_weights)
            
            # Smoothness = inverse of variance in changes (lower variance = smoother)
            if len(raw_diffs) > 0 and len(filtered_diffs) > 0:
                raw_smoothness = 1 / (np.var(raw_diffs) + 0.001)  # Add small constant to avoid division by zero
                filtered_smoothness = 1 / (np.var(filtered_diffs) + 0.001)
                
                # Relative improvement in smoothness
                smoothness_gain = (filtered_smoothness - raw_smoothness) / max(raw_smoothness, 0.001) * 100
                smoothness_gains.append(smoothness_gain)
        
        # 4. CLINICAL PLAUSIBILITY (extreme value detection)
        # Check for physiologically implausible values
        def count_implausible(weights: np.ndarray) -> int:
            """Count weights outside plausible human range"""
            return np.sum((weights < 30) | (weights > 300))  # kg ranges
        
        raw_implausible = count_implausible(raw_weights)
        filtered_implausible = count_implausible(filtered_weights)
        
        # Calculate improvement (fewer implausible values is better)
        if len(raw_weights) > 0:
            raw_implausible_rate = raw_implausible / len(raw_weights)
            filtered_implausible_rate = filtered_implausible / len(filtered_weights) if len(filtered_weights) > 0 else 0
            plausibility_improvement = (raw_implausible_rate - filtered_implausible_rate) * 100
            plausibility_improvements.append(plausibility_improvement)
        
        # 5. TEMPORAL CONSISTENCY (autocorrelation)
        if len(filtered_weights) > 10:
            # Calculate lag-1 autocorrelation (should be high for weight data)
            try:
                # Use pandas autocorrelation for robustness
                raw_series = pd.Series(raw_weights)
                filtered_series = pd.Series(filtered_weights)
                
                raw_autocorr = raw_series.autocorr(lag=1) if len(raw_weights) > 1 else 0
                filtered_autocorr = filtered_series.autocorr(lag=1) if len(filtered_weights) > 1 else 0
                
                # Higher autocorrelation = better temporal consistency
                consistency_scores.append({
                    'raw': raw_autocorr,
                    'filtered': filtered_autocorr,
                    'improvement': filtered_autocorr - raw_autocorr
                })
            except:
                pass
    
    # === ANALYSIS RESULTS ===
    
    # 1. NORMALITY ANALYSIS
    if normality_improvements:
        avg_normality = np.mean(normality_improvements)
        improved_count = sum(1 for x in normality_improvements if x > 0)
        
        logging.info("\n📊 DISTRIBUTION NORMALITY:")
        if avg_normality > 0.1:
            logging.info(f"  ✅ Filtering IMPROVES normality (+{avg_normality:.1%} average)")
            logging.info(f"     {improved_count}/{len(normality_improvements)} users show improvement")
        elif avg_normality < -0.1:
            logging.info(f"  ⚠️  Filtering REDUCES normality ({avg_normality:.1%} average)")
        else:
            logging.info(f"  ➖ Minimal impact on normality ({avg_normality:+.1%})")
    
    # 2. VARIANCE ANALYSIS  
    if variance_reductions:
        avg_var_reduction = np.mean(variance_reductions)
        median_var_reduction = np.median(variance_reductions)
        
        logging.info("\n📈 VARIANCE REDUCTION:")
        if avg_var_reduction > 10:
            logging.info(f"  ✅ Significant variance reduction: {avg_var_reduction:.1f}%")
            logging.info(f"     Median: {median_var_reduction:.1f}%")
            logging.info(f"     → Filtering effectively removes noise")
        elif avg_var_reduction > 0:
            logging.info(f"  ✓ Moderate variance reduction: {avg_var_reduction:.1f}%")
        else:
            logging.info(f"  ⚠️  Variance INCREASED by {abs(avg_var_reduction):.1f}%")
            logging.info(f"     → May be removing legitimate variation")
    
    # 3. SMOOTHNESS ANALYSIS
    if smoothness_gains:
        avg_smoothness = np.mean(smoothness_gains)
        smooth_improved = sum(1 for x in smoothness_gains if x > 0)
        
        logging.info("\n📉 TREND SMOOTHNESS:")
        if avg_smoothness > 20:
            logging.info(f"  ✅ Major smoothness improvement: +{avg_smoothness:.0f}%")
            logging.info(f"     {smooth_improved}/{len(smoothness_gains)} users have smoother trends")
        elif avg_smoothness > 0:
            logging.info(f"  ✓ Trends are {avg_smoothness:.0f}% smoother after filtering")
        else:
            logging.info(f"  ⚠️  Trends are {abs(avg_smoothness):.0f}% MORE erratic after filtering")
    
    # 4. CLINICAL PLAUSIBILITY
    if plausibility_improvements:
        avg_plausibility = np.mean(plausibility_improvements)
        
        logging.info("\n🏥 CLINICAL PLAUSIBILITY:")
        if avg_plausibility > 0:
            logging.info(f"  ✅ Filtering removes {avg_plausibility:.2f}% of implausible values")
            logging.info(f"     → Improved clinical reliability")
        else:
            logging.info(f"  ✓ No implausible values detected in either dataset")
    
    # 5. TEMPORAL CONSISTENCY
    if consistency_scores:
        avg_raw_consistency = np.mean([x['raw'] for x in consistency_scores])
        avg_filtered_consistency = np.mean([x['filtered'] for x in consistency_scores])
        consistency_improvement = avg_filtered_consistency - avg_raw_consistency
        
        logging.info("\n⏱️  TEMPORAL CONSISTENCY (autocorrelation):")
        logging.info(f"  • Raw data: {avg_raw_consistency:.3f}")
        logging.info(f"  • Filtered: {avg_filtered_consistency:.3f}")
        
        if consistency_improvement > 0.05:
            logging.info(f"  ✅ Filtering IMPROVES consistency (+{consistency_improvement:.3f})")
            logging.info(f"     → Weight changes are more predictable")
        elif consistency_improvement < -0.05:
            logging.info(f"  ⚠️  Filtering REDUCES consistency ({consistency_improvement:.3f})")
            logging.info(f"     → May be creating artificial gaps")
        else:
            logging.info(f"  ➖ Minimal impact on consistency ({consistency_improvement:+.3f})")
    
    # === OVERALL VERDICT ===
    logging.info("\n" + "="*50)
    logging.info("VERDICT: Does filtering improve clinical reliability?")
    logging.info("="*50)
    
    # Count positive indicators
    positive_indicators = 0
    negative_indicators = 0
    
    if normality_improvements and np.mean(normality_improvements) > 0:
        positive_indicators += 1
    elif normality_improvements and np.mean(normality_improvements) < -0.1:
        negative_indicators += 1
    
    if variance_reductions and np.mean(variance_reductions) > 5:
        positive_indicators += 1
    elif variance_reductions and np.mean(variance_reductions) < -5:
        negative_indicators += 1
    
    if smoothness_gains and np.mean(smoothness_gains) > 10:
        positive_indicators += 1
    elif smoothness_gains and np.mean(smoothness_gains) < -10:
        negative_indicators += 1
    
    if plausibility_improvements and np.mean(plausibility_improvements) > 0:
        positive_indicators += 1
    
    if consistency_scores and (np.mean([x['filtered'] for x in consistency_scores]) - 
                               np.mean([x['raw'] for x in consistency_scores])) > 0.02:
        positive_indicators += 1
    elif consistency_scores and (np.mean([x['filtered'] for x in consistency_scores]) - 
                                 np.mean([x['raw'] for x in consistency_scores])) < -0.05:
        negative_indicators += 1
    
    # Provide clear, actionable verdict
    if positive_indicators >= 3:
        logging.info("\n✅ YES - Filtering SIGNIFICANTLY improves data quality")
        logging.info(f"   • {positive_indicators}/5 metrics show improvement")
        logging.info("   • RECOMMENDATION: Continue using current filtering approach")
    elif positive_indicators > negative_indicators:
        logging.info("\n✓ YES - Filtering provides moderate improvements")
        logging.info(f"   • {positive_indicators}/5 metrics show improvement")
        logging.info("   • RECOMMENDATION: Fine-tune thresholds for better performance")
    elif negative_indicators > positive_indicators:
        logging.info("\n⚠️  NO - Filtering may be too aggressive")
        logging.info(f"   • {negative_indicators} metrics show degradation")
        logging.info("   • RECOMMENDATION: Review and relax filtering thresholds")
    else:
        logging.info("\n➖ NEUTRAL - Minimal impact observed")
        logging.info("   • Data quality is already high OR")
        logging.info("   • Current thresholds need adjustment")
        logging.info("   • RECOMMENDATION: Analyze specific outlier cases")
    
    # Key metrics summary
    if variance_reductions:
        logging.info(f"\n📊 KEY METRIC: Variance reduced by {np.mean(variance_reductions):.1f}%")
    if smoothness_gains:
        logging.info(f"📊 KEY METRIC: Smoothness improved by {np.mean(smoothness_gains):.0f}%")
    if consistency_scores:
        improvement = np.mean([x['improvement'] for x in consistency_scores])
        logging.info(f"📊 KEY METRIC: Temporal consistency {'improved' if improvement > 0 else 'degraded'} by {abs(improvement):.3f}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Simple report for user data analysis')
    parser.add_argument(
        '--employer',
        type=str,
        help='Filter users by employer name (e.g., AMAZON_EMPLOYER, APPLE_EMPLOYER)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--interval-analysis',
        action='store_true',
        help='Perform interval weight loss analysis (30-day intervals up to 360 days)'
    )
    parser.add_argument(
        '--export-intervals',
        type=str,
        help='Export interval analysis to CSV file'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of users to process (for testing)'
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    start_time = time.time()

    # Check files exist
    if not FILTERED_FILE.exists():
        logging.error(f"Error: {FILTERED_FILE} not found")
        return
    if not RAW_FILE.exists():
        logging.error(f"Error: {RAW_FILE} not found")
        return

    # Load employer filter if specified
    employer_user_ids = None
    user_start_dates = {}
    if args.employer:
        employer_user_ids, user_start_dates = load_employer_filter(args.employer)
        if not employer_user_ids:
            logging.error(f"No users found for employer '{args.employer}'")
            return
    else:
        # Load all user start dates if no employer filter
        if USER_EMPLOYERS_FILE.exists():
            user_employers_df = pd.read_csv(USER_EMPLOYERS_FILE, usecols=['user_id', 'start_date'])
            user_start_dates = dict(zip(user_employers_df['user_id'], user_employers_df['start_date']))

    # Load data efficiently
    df_filtered, df_raw = load_data_efficient(FILTERED_FILE, RAW_FILE, employer_user_ids)

    # Get unique users from both datasets (already in memory)
    filtered_users = set(df_filtered['user_id'].unique())
    raw_users = set(df_raw['user_id'].unique())

    # Find users that exist in BOTH datasets
    users_in_both = filtered_users & raw_users

    # Apply limit if specified (for testing)
    if args.limit and args.limit > 0:
        users_in_both = set(list(users_in_both)[:args.limit])

    # Apply employer filter if specified
    if employer_user_ids:
        # Only keep users that are in both datasets AND in employer list
        users_to_keep = users_in_both & employer_user_ids

        if not users_to_keep:
            logging.error(f"No users found for employer '{args.employer}' that exist in both datasets")
            return

        # Filter dataframes in place to save memory
        df_filtered = df_filtered[df_filtered['user_id'].isin(users_to_keep)]
        df_raw = df_raw[df_raw['user_id'].isin(users_to_keep)]
    else:
        # No employer filter, but still only keep users in both datasets
        users_to_keep = users_in_both
        df_filtered = df_filtered[df_filtered['user_id'].isin(users_to_keep)]
        df_raw = df_raw[df_raw['user_id'].isin(users_to_keep)]

    # Group by user_id for both datasets (more efficient grouping)
    filtered_by_user = df_filtered.groupby('user_id', sort=False)
    raw_by_user = df_raw.groupby('user_id', sort=False)

    # Keep data separate but accessible - now guaranteed both exist for each user
    user_data = {}
    for user_id in users_to_keep:
        user_data[user_id] = {
            'filtered': filtered_by_user.get_group(user_id),
            'raw': raw_by_user.get_group(user_id)
        }

    # Calculate start weights for each user using their individual start dates
    start_weights = {}
    users_without_start_date = []

    for user_id in users_to_keep:
        # Get user's start date
        start_date = user_start_dates.get(user_id)

        if start_date:
            raw_start = get_closest_weight(user_data[user_id]['raw'], start_date)
            filtered_start = get_closest_weight(user_data[user_id]['filtered'], start_date)

            start_weights[user_id] = {
                'start_date': start_date,
                'raw_start_weight': raw_start,
                'filtered_start_weight': filtered_start
            }
        else:
            users_without_start_date.append(user_id)

    # Count users with start weights
    users_with_raw_start = sum(1 for v in start_weights.values() if v['raw_start_weight'] is not None)
    users_with_filtered_start = sum(1 for v in start_weights.values() if v['filtered_start_weight'] is not None)
    users_with_both_starts = sum(1 for v in start_weights.values()
                                 if v['raw_start_weight'] is not None and v['filtered_start_weight'] is not None)

    # Output user count summary
    logging.info(f"\n=== DATA SUMMARY ===")
    logging.info(f"📊 Total users analyzed: {len(users_to_keep)}")
    logging.info(f"✓ Users with valid start weights: {users_with_both_starts} ({users_with_both_starts/len(users_to_keep)*100:.0f}%)")
    
    # Calculate filtering impact
    total_raw = len(df_raw)
    total_filtered = len(df_filtered)
    measurements_removed = total_raw - total_filtered
    removal_rate = (measurements_removed / total_raw * 100) if total_raw > 0 else 0
    
    logging.info(f"\n📈 FILTERING IMPACT:")
    logging.info(f"  • Raw measurements: {total_raw:,}")
    logging.info(f"  • After filtering: {total_filtered:,}")
    logging.info(f"  • Removed: {measurements_removed:,} ({removal_rate:.1f}%)")
    
    # Calculate average measurements per user
    avg_filtered = len(df_filtered) / len(users_to_keep)
    avg_raw = len(df_raw) / len(users_to_keep)
    avg_removed = avg_raw - avg_filtered
    logging.info(f"  • Per user avg: {avg_raw:.0f} → {avg_filtered:.0f} (−{avg_removed:.0f})")

    # Analyze filtering patterns
    analyze_filtering_patterns(df_raw, df_filtered, users_to_keep)
    
    # Calculate differences between raw and filtered start weights
    differences = []
    for user_id, weights in start_weights.items():
        raw_weight = weights['raw_start_weight']
        filtered_weight = weights['filtered_start_weight']

        if raw_weight is not None and filtered_weight is not None:
            abs_diff = abs(raw_weight - filtered_weight)
            pct_diff = (abs_diff / raw_weight * 100) if raw_weight > 0 else 0
            differences.append({
                'user_id': user_id,
                'raw': raw_weight,
                'filtered': filtered_weight,
                'abs_diff': abs_diff,
                'pct_diff': pct_diff
            })

    # Calculate similarity statistics
    if differences:
        abs_diffs = [d['abs_diff'] for d in differences]
        pct_diffs = [d['pct_diff'] for d in differences]

        identical_count = sum(1 for d in abs_diffs if d < 0.01)
        
        # Analyze start weight statistics - compare raw vs filtered
        raw_start_weights = [w['raw_start_weight'] for w in start_weights.values() 
                            if w['raw_start_weight'] is not None]
        filtered_start_weights = [w['filtered_start_weight'] for w in start_weights.values() 
                                 if w['filtered_start_weight'] is not None]
        
        if filtered_start_weights and raw_start_weights:
            avg_raw = sum(raw_start_weights) / len(raw_start_weights)
            avg_filtered = sum(filtered_start_weights) / len(filtered_start_weights)
            
            logging.info(f"\n📊 START WEIGHT COMPARISON:")
            logging.info(f"  • Raw average: {avg_raw:.1f} kg")
            logging.info(f"  • Filtered average: {avg_filtered:.1f} kg")
            logging.info(f"  • Difference: {abs(avg_raw - avg_filtered):.2f} kg")
            
            # Show range comparison
            logging.info(f"  • Raw range: {min(raw_start_weights):.1f} - {max(raw_start_weights):.1f} kg")
            logging.info(f"  • Filtered range: {min(filtered_start_weights):.1f} - {max(filtered_start_weights):.1f} kg")
        
        # Data quality assessment with detailed breakdown
        logging.info(f"\n🔍 RAW vs FILTERED ALIGNMENT:")
        
        # Categorize differences
        perfect_match = sum(1 for d in abs_diffs if d < 0.01)
        minor_diff = sum(1 for d in abs_diffs if 0.01 <= d < 1.0)
        moderate_diff = sum(1 for d in abs_diffs if 1.0 <= d < 5.0)
        major_diff = sum(1 for d in abs_diffs if d >= 5.0)
        
        total_users = len(differences)
        logging.info(f"  • Perfect match (<0.01kg): {perfect_match} users ({perfect_match/total_users*100:.0f}%)")
        logging.info(f"  • Minor diff (0.01-1kg): {minor_diff} users ({minor_diff/total_users*100:.0f}%)")
        logging.info(f"  • Moderate diff (1-5kg): {moderate_diff} users ({moderate_diff/total_users*100:.0f}%)")
        logging.info(f"  • Major diff (>5kg): {major_diff} users ({major_diff/total_users*100:.0f}%)")
        
        # Statistical summary
        avg_diff = sum(abs_diffs) / len(abs_diffs)
        median_diff = sorted(abs_diffs)[len(abs_diffs)//2]
        
        logging.info(f"\n📊 DIFFERENCE STATISTICS:")
        logging.info(f"  • Average difference: {avg_diff:.2f} kg")
        logging.info(f"  • Median difference: {median_diff:.2f} kg")
        logging.info(f"  • Maximum difference: {max(abs_diffs):.1f} kg")
        
        # Quality verdict
        if major_diff > 0:
            logging.info(f"\n⚠️  FILTERING ASSESSMENT: Aggressive")
            logging.info(f"  • {major_diff} users have significant data filtered")
            logging.info(f"  • Review outlier thresholds if unexpected")
        elif moderate_diff > total_users * 0.1:
            logging.info(f"\n📝 FILTERING ASSESSMENT: Moderate")
            logging.info(f"  • Most differences are within expected range")
            logging.info(f"  • Filtering appears to be working as intended")
        else:
            logging.info(f"\n✅ FILTERING ASSESSMENT: Conservative")
            logging.info(f"  • {(perfect_match + minor_diff)/total_users*100:.0f}% of users have minimal filtering")
            logging.info(f"  • Consider if more aggressive filtering is needed")

        # Sort by absolute difference and show top 5
        differences.sort(key=lambda x: x['abs_diff'], reverse=True)

    # Perform interval analysis if requested
    if args.interval_analysis:
        # Calculate interval analysis
        df_intervals = calculate_interval_analysis(user_data, start_weights)
        
        # Analyze and display results
        analyze_interval_results(df_intervals)
        
        # Export if requested
        if args.export_intervals:
            export_path = Path(args.export_intervals)
            df_intervals.to_csv(export_path, index=False)
            logging.info(f"\n✓ Detailed interval data exported to: {export_path}")
    
    # Analyze outlier effectiveness
    analyze_outlier_effectiveness(df_raw, df_filtered, users_to_keep)

    # Report total execution time
    logging.info(f"\nExecution time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()

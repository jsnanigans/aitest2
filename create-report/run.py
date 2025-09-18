#!/usr/bin/env python3
"""
Script to load and filter weight measurement data by employer.
Loads filtered users, matches with employer data, and retrieves corresponding raw measurements.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Visualization imports
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Load and filter weight data by employer')
    parser.add_argument(
        '--employer',
        type=str,
        help='Employer name to filter by (e.g., AMAZON_EMPLOYER). If not provided, loads all users.'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations comparing raw vs filtered data'
    )
    return parser.parse_args()


def load_filtered_users(filtered_file: str) -> pd.DataFrame:
    """
    Load filtered measurements and get unique users.

    Returns:
        DataFrame with filtered measurements
    """
    print("Loading filtered measurements...")
    filtered_df = pd.read_csv(filtered_file)
    filtered_df['effectiveDateTime'] = pd.to_datetime(filtered_df['effectiveDateTime'])
    print(f"  Loaded {len(filtered_df):,} filtered measurements")
    print(f"  Found {filtered_df['user_id'].nunique():,} unique filtered users")
    return filtered_df


def load_employer_data(employer_file: str, partners_file: str) -> pd.DataFrame:
    """
    Load employer data and merge with partner names.

    Returns:
        DataFrame with user_id, employer_id, employer_name, and start_date
    """
    print("\nLoading employer data...")

    # Load user-employer data
    employer_df = pd.read_csv(employer_file)
    employer_df['start_date'] = pd.to_datetime(employer_df['start_date'])
    print(f"  Loaded {len(employer_df):,} user-employer records")

    # Load partner names
    partners_df = pd.read_csv(partners_file)
    print(f"  Loaded {len(partners_df):,} partner definitions")

    # Merge to get employer names
    employer_with_names = employer_df.merge(
        partners_df,
        left_on='employer_id',
        right_on='id',
        how='left'
    )

    # Select and rename columns
    employer_with_names = employer_with_names[['user_id', 'employer_id', 'name', 'start_date']]
    employer_with_names = employer_with_names.rename(columns={'name': 'employer_name'})

    print(f"  Merged data has {employer_with_names['employer_name'].nunique()} unique employers")

    return employer_with_names


def filter_users_by_employer(filtered_df: pd.DataFrame, employer_df: pd.DataFrame,
                            employer_name: str = None) -> Tuple[List[str], pd.DataFrame]:
    """
    Filter users by employer and return list of user IDs.

    Returns:
        Tuple of (list of user_ids, employer_df for selected users)
    """
    # Get unique users from filtered data
    filtered_users = set(filtered_df['user_id'].unique())

    if employer_name:
        print(f"\nFiltering users for employer: {employer_name}")
        # Filter employer data by name
        employer_filtered = employer_df[employer_df['employer_name'] == employer_name].copy()

        if employer_filtered.empty:
            print(f"  WARNING: No users found for employer '{employer_name}'")
            print(f"  Available employers: {sorted(employer_df['employer_name'].dropna().unique())}")
            return [], pd.DataFrame()

        print(f"  Found {len(employer_filtered):,} users with employer {employer_name}")
    else:
        print("\nNo employer filter specified, using all users")
        employer_filtered = employer_df.copy()

    # Get intersection of filtered users and employer users
    employer_users = set(employer_filtered['user_id'].unique())
    selected_users = list(filtered_users & employer_users)

    print(f"  Selected {len(selected_users):,} users (intersection of filtered and employer data)")

    # Return only employer data for selected users
    employer_filtered = employer_filtered[employer_filtered['user_id'].isin(selected_users)]

    return selected_users, employer_filtered


def filter_users_by_program_duration(employer_df: pd.DataFrame, selected_users: List[str],
                                    min_days: int = 90, reference_date: str = '2025-09-05') -> Tuple[List[str], pd.DataFrame]:
    """
    Filter users who have been in the program for at least min_days.

    Args:
        employer_df: DataFrame with user_id and start_date
        selected_users: List of user IDs to filter
        min_days: Minimum number of days in program (default 90)
        reference_date: The date to consider as "today" (default '2025-09-05')

    Returns:
        Tuple of (filtered user_ids, filtered employer_df)
    """
    ref_date = pd.to_datetime(reference_date)

    print(f"\nFiltering users with at least {min_days} days in program")
    print(f"  Reference date (today): {reference_date}")

    # Filter employer data for selected users
    employer_filtered = employer_df[employer_df['user_id'].isin(selected_users)].copy()

    # Calculate days in program
    employer_filtered['days_in_program'] = (ref_date - employer_filtered['start_date']).dt.total_seconds() / 86400

    # Filter for users with at least min_days
    long_term_users = employer_filtered[employer_filtered['days_in_program'] >= min_days]

    # Get unique user IDs
    filtered_user_ids = list(long_term_users['user_id'].unique())

    # Stats
    total_before = len(selected_users)
    total_after = len(filtered_user_ids)
    removed = total_before - total_after

    print(f"  Users before filtering: {total_before:,}")
    print(f"  Users with {min_days}+ days: {total_after:,}")
    print(f"  Users removed: {removed:,} ({100*removed/total_before:.1f}%)")

    if total_after > 0:
        avg_days = long_term_users['days_in_program'].mean()
        median_days = long_term_users['days_in_program'].median()
        print(f"  Average days in program: {avg_days:.1f}")
        print(f"  Median days in program: {median_days:.1f}")

    return filtered_user_ids, long_term_users


def load_raw_data_for_users(raw_file: str, user_ids: List[str]) -> pd.DataFrame:
    """
    Load raw measurements only for specified users.
    Uses chunked reading for memory efficiency.

    Returns:
        DataFrame with raw measurements for selected users
    """
    print(f"\nLoading raw data for {len(user_ids):,} users...")

    # Convert to set for faster lookup
    user_id_set = set(user_ids)

    # Read in chunks to avoid loading entire file into memory
    chunks = []
    chunk_size = 50000  # Process 50k rows at a time

    for chunk in pd.read_csv(raw_file, chunksize=chunk_size):
        # Filter chunk for selected users
        filtered_chunk = chunk[chunk['user_id'].isin(user_id_set)]
        if not filtered_chunk.empty:
            chunks.append(filtered_chunk)

    # Combine all chunks
    if chunks:
        raw_df = pd.concat(chunks, ignore_index=True)
        raw_df['effectiveDateTime'] = pd.to_datetime(raw_df['effectiveDateTime'])
    else:
        raw_df = pd.DataFrame()

    print(f"  Loaded {len(raw_df):,} raw measurements for selected users")

    return raw_df


def create_user_data_structure(selected_users: List[str],
                              raw_df: pd.DataFrame,
                              filtered_df: pd.DataFrame,
                              employer_df: pd.DataFrame) -> Dict:
    """
    Create a dictionary structure with raw, filtered, and start_date for each user.
    Optimized version using groupby operations.

    Returns:
        Dictionary with user_id as key and dict of data as value
    """
    print(f"\nCreating user data structure for {len(selected_users):,} users...")

    # Pre-compute aggregations for efficiency
    raw_grouped = raw_df.groupby('user_id')
    filtered_grouped = filtered_df.groupby('user_id')

    # Get counts efficiently
    raw_counts = raw_df['user_id'].value_counts().to_dict()
    filtered_counts = filtered_df['user_id'].value_counts().to_dict()

    # Prepare employer data - get first employer name and min start date per user
    employer_summary = employer_df.groupby('user_id').agg({
        'start_date': 'min',
        'employer_name': 'first'
    })

    # Add days_in_program if it exists in employer_df
    if 'days_in_program' in employer_df.columns:
        days_in_program = employer_df.groupby('user_id')['days_in_program'].first()
        employer_summary['days_in_program'] = days_in_program

    employer_summary = employer_summary.to_dict('index')

    user_data = {}

    # Use set for faster lookup
    selected_users_set = set(selected_users)

    for user_id in selected_users_set:
        # Get employer info
        emp_info = employer_summary.get(user_id, {})

        user_data[user_id] = {
            'raw_data': raw_grouped.get_group(user_id) if user_id in raw_counts else pd.DataFrame(),
            'filtered_data': filtered_grouped.get_group(user_id) if user_id in filtered_counts else pd.DataFrame(),
            'start_date': emp_info.get('start_date'),
            'employer_name': emp_info.get('employer_name'),
            'days_in_program': emp_info.get('days_in_program'),
            'raw_count': raw_counts.get(user_id, 0),
            'filtered_count': filtered_counts.get(user_id, 0)
        }

    # Summary statistics (vectorized)
    total_raw = sum(raw_counts.get(uid, 0) for uid in selected_users_set)
    total_filtered = sum(filtered_counts.get(uid, 0) for uid in selected_users_set)

    print(f"  Total raw measurements: {total_raw:,}")
    print(f"  Total filtered measurements: {total_filtered:,}")
    print(f"  Average raw per user: {total_raw/len(selected_users):.1f}")
    print(f"  Average filtered per user: {total_filtered/len(selected_users):.1f}")

    return user_data


def find_closest_weight_to_start(user_data: Dict) -> Dict:
    """
    For each user, find the weight measurement closest to their start date.
    Analyzes both raw and filtered data.

    Returns:
        Updated user_data dictionary with closest weight information
    """
    print("\nAnalyzing weight measurements near start dates...")

    users_with_analysis = 0
    missing_start_dates = 0
    no_measurements_near_start = 0

    for user_id, data in user_data.items():
        start_date = data['start_date']

        if pd.isna(start_date) or start_date is None:
            missing_start_dates += 1
            data['closest_raw_weight'] = None
            data['closest_filtered_weight'] = None
            data['days_from_start_raw'] = None
            data['days_from_start_filtered'] = None
            continue

        # Find closest raw weight
        if not data['raw_data'].empty:
            raw_df = data['raw_data'].copy()
            raw_df['days_from_start'] = (raw_df['effectiveDateTime'] - start_date).dt.total_seconds() / 86400
            raw_df['abs_days'] = raw_df['days_from_start'].abs()

            closest_raw_idx = raw_df['abs_days'].idxmin()
            closest_raw = raw_df.loc[closest_raw_idx]

            data['closest_raw_weight'] = closest_raw['weight']
            data['closest_raw_date'] = closest_raw['effectiveDateTime']
            data['days_from_start_raw'] = closest_raw['days_from_start']
        else:
            data['closest_raw_weight'] = None
            data['closest_raw_date'] = None
            data['days_from_start_raw'] = None

        # Find closest filtered weight
        if not data['filtered_data'].empty:
            filtered_df = data['filtered_data'].copy()
            filtered_df['days_from_start'] = (filtered_df['effectiveDateTime'] - start_date).dt.total_seconds() / 86400
            filtered_df['abs_days'] = filtered_df['days_from_start'].abs()

            closest_filtered_idx = filtered_df['abs_days'].idxmin()
            closest_filtered = filtered_df.loc[closest_filtered_idx]

            data['closest_filtered_weight'] = closest_filtered['weight']
            data['closest_filtered_date'] = closest_filtered['effectiveDateTime']
            data['days_from_start_filtered'] = closest_filtered['days_from_start']
            data['closest_filtered_quality'] = closest_filtered.get('quality_score', None)
        else:
            data['closest_filtered_weight'] = None
            data['closest_filtered_date'] = None
            data['days_from_start_filtered'] = None
            data['closest_filtered_quality'] = None

        if data['closest_raw_weight'] is not None or data['closest_filtered_weight'] is not None:
            users_with_analysis += 1
        else:
            no_measurements_near_start += 1

    # Summary statistics
    print(f"  Users with start dates: {len(user_data) - missing_start_dates:,}")
    print(f"  Users with weight near start: {users_with_analysis:,}")
    print(f"  Missing start dates: {missing_start_dates:,}")
    print(f"  No measurements found: {no_measurements_near_start:,}")

    # Calculate averages for users with data
    raw_days_from_start = [abs(d['days_from_start_raw']) for d in user_data.values()
                           if d.get('days_from_start_raw') is not None]
    filtered_days_from_start = [abs(d['days_from_start_filtered']) for d in user_data.values()
                                if d.get('days_from_start_filtered') is not None]

    if raw_days_from_start:
        print(f"\n  Raw measurements:")
        print(f"    Average days from start: {np.mean(raw_days_from_start):.1f}")
        print(f"    Median days from start: {np.median(raw_days_from_start):.1f}")
        print(f"    Within 7 days: {sum(d <= 7 for d in raw_days_from_start):,}")
        print(f"    Within 30 days: {sum(d <= 30 for d in raw_days_from_start):,}")

    if filtered_days_from_start:
        print(f"\n  Filtered measurements:")
        print(f"    Average days from start: {np.mean(filtered_days_from_start):.1f}")
        print(f"    Median days from start: {np.median(filtered_days_from_start):.1f}")
        print(f"    Within 7 days: {sum(d <= 7 for d in filtered_days_from_start):,}")
        print(f"    Within 30 days: {sum(d <= 30 for d in filtered_days_from_start):,}")

    return user_data


def find_weight_at_90_days(user_data: Dict) -> Dict:
    """
    For each user, find the weight measurement closest to 90 days after start.
    Analyzes both raw and filtered data.

    Returns:
        Updated user_data dictionary with 90-day weight information
    """
    print("\nAnalyzing weight measurements at 90 days...")

    users_with_90d = 0
    missing_start_dates = 0
    no_measurements_near_90d = 0

    for user_id, data in user_data.items():
        start_date = data['start_date']

        if pd.isna(start_date) or start_date is None:
            missing_start_dates += 1
            data['weight_90d_raw'] = None
            data['weight_90d_filtered'] = None
            data['days_from_90d_raw'] = None
            data['days_from_90d_filtered'] = None
            continue

        # Calculate target date (90 days after start)
        target_date = start_date + pd.Timedelta(days=90)

        # Find closest raw weight to 90 days
        if not data['raw_data'].empty:
            raw_df = data['raw_data'].copy()
            raw_df['days_from_90d'] = (raw_df['effectiveDateTime'] - target_date).dt.total_seconds() / 86400
            raw_df['abs_days_from_90d'] = raw_df['days_from_90d'].abs()

            # Only consider measurements within reasonable range (e.g., within 30 days of 90-day mark)
            raw_near_90d = raw_df[raw_df['abs_days_from_90d'] <= 30]

            if not raw_near_90d.empty:
                closest_raw_idx = raw_near_90d['abs_days_from_90d'].idxmin()
                closest_raw = raw_near_90d.loc[closest_raw_idx]

                data['weight_90d_raw'] = closest_raw['weight']
                data['date_90d_raw'] = closest_raw['effectiveDateTime']
                data['days_from_90d_raw'] = closest_raw['days_from_90d']
            else:
                data['weight_90d_raw'] = None
                data['date_90d_raw'] = None
                data['days_from_90d_raw'] = None
        else:
            data['weight_90d_raw'] = None
            data['date_90d_raw'] = None
            data['days_from_90d_raw'] = None

        # Find closest filtered weight to 90 days
        if not data['filtered_data'].empty:
            filtered_df = data['filtered_data'].copy()
            filtered_df['days_from_90d'] = (filtered_df['effectiveDateTime'] - target_date).dt.total_seconds() / 86400
            filtered_df['abs_days_from_90d'] = filtered_df['days_from_90d'].abs()

            # Only consider measurements within reasonable range
            filtered_near_90d = filtered_df[filtered_df['abs_days_from_90d'] <= 30]

            if not filtered_near_90d.empty:
                closest_filtered_idx = filtered_near_90d['abs_days_from_90d'].idxmin()
                closest_filtered = filtered_near_90d.loc[closest_filtered_idx]

                data['weight_90d_filtered'] = closest_filtered['weight']
                data['date_90d_filtered'] = closest_filtered['effectiveDateTime']
                data['days_from_90d_filtered'] = closest_filtered['days_from_90d']
                data['quality_90d_filtered'] = closest_filtered.get('quality_score', None)
            else:
                data['weight_90d_filtered'] = None
                data['date_90d_filtered'] = None
                data['days_from_90d_filtered'] = None
                data['quality_90d_filtered'] = None
        else:
            data['weight_90d_filtered'] = None
            data['date_90d_filtered'] = None
            data['days_from_90d_filtered'] = None
            data['quality_90d_filtered'] = None

        if data['weight_90d_raw'] is not None or data['weight_90d_filtered'] is not None:
            users_with_90d += 1
        else:
            no_measurements_near_90d += 1

    # Summary statistics
    print(f"  Users with start dates: {len(user_data) - missing_start_dates:,}")
    print(f"  Users with weight near 90 days: {users_with_90d:,}")
    print(f"  Missing start dates: {missing_start_dates:,}")
    print(f"  No measurements near 90 days: {no_measurements_near_90d:,}")

    # Calculate averages for users with data
    raw_days_from_90d = [abs(d['days_from_90d_raw']) for d in user_data.values()
                         if d.get('days_from_90d_raw') is not None]
    filtered_days_from_90d = [abs(d['days_from_90d_filtered']) for d in user_data.values()
                              if d.get('days_from_90d_filtered') is not None]

    if raw_days_from_90d:
        print(f"\n  Raw measurements (90-day mark):")
        print(f"    Average days from 90d: {np.mean(raw_days_from_90d):.1f}")
        print(f"    Median days from 90d: {np.median(raw_days_from_90d):.1f}")
        print(f"    Within 7 days of 90d: {sum(d <= 7 for d in raw_days_from_90d):,}")
        print(f"    Within 14 days of 90d: {sum(d <= 14 for d in raw_days_from_90d):,}")

    if filtered_days_from_90d:
        print(f"\n  Filtered measurements (90-day mark):")
        print(f"    Average days from 90d: {np.mean(filtered_days_from_90d):.1f}")
        print(f"    Median days from 90d: {np.median(filtered_days_from_90d):.1f}")
        print(f"    Within 7 days of 90d: {sum(d <= 7 for d in filtered_days_from_90d):,}")
        print(f"    Within 14 days of 90d: {sum(d <= 14 for d in filtered_days_from_90d):,}")

    return user_data


def analyze_data_quality_improvements(user_data: Dict) -> Dict:
    """
    Analyze how well the filtering improved data quality by comparing raw and filtered data.

    Returns:
        Dictionary with detailed quality metrics
    """
    print("\n" + "="*60)
    print("DATA QUALITY IMPROVEMENT ANALYSIS")
    print("="*60)

    quality_metrics = {
        'total_users': len(user_data),
        'noise_reduction': {},
        'outlier_removal': {},
        'consistency_improvements': {},
        'trajectory_analysis': {},
        'statistical_improvements': {}
    }

    # Collect all raw and filtered weights for global analysis
    all_raw_weights = []
    all_filtered_weights = []
    weight_changes_raw = []
    weight_changes_filtered = []

    users_with_both_data = 0
    users_with_improved_consistency = 0
    users_with_reduced_variance = 0

    for user_id, data in user_data.items():
        if data['raw_data'].empty or data['filtered_data'].empty:
            continue

        users_with_both_data += 1

        raw_df = data['raw_data'].sort_values('effectiveDateTime')
        filtered_df = data['filtered_data'].sort_values('effectiveDateTime')

        raw_weights = raw_df['weight'].values
        filtered_weights = filtered_df['weight'].values

        all_raw_weights.extend(raw_weights)
        all_filtered_weights.extend(filtered_weights)

        # Calculate day-to-day changes
        if len(raw_weights) > 1:
            raw_changes = np.diff(raw_weights)
            weight_changes_raw.extend(raw_changes)

        if len(filtered_weights) > 1:
            filtered_changes = np.diff(filtered_weights)
            weight_changes_filtered.extend(filtered_changes)

        # Per-user variance comparison
        if len(raw_weights) > 2 and len(filtered_weights) > 2:
            raw_variance = np.var(raw_weights)
            filtered_variance = np.var(filtered_weights)

            if filtered_variance < raw_variance:
                users_with_reduced_variance += 1

            # Check consistency improvement (reduced max daily change)
            if len(raw_weights) > 1 and len(filtered_weights) > 1:
                max_raw_change = np.max(np.abs(np.diff(raw_weights)))
                max_filtered_change = np.max(np.abs(np.diff(filtered_weights)))

                if max_filtered_change < max_raw_change:
                    users_with_improved_consistency += 1

    # 1. Noise Reduction Analysis
    print("\n📊 NOISE REDUCTION METRICS")
    print("-" * 40)

    if all_raw_weights and all_filtered_weights:
        raw_std = np.std(all_raw_weights)
        filtered_std = np.std(all_filtered_weights)
        variance_reduction = 100 * (raw_std - filtered_std) / raw_std if raw_std > 0 else 0

        quality_metrics['noise_reduction'] = {
            'raw_std_dev': raw_std,
            'filtered_std_dev': filtered_std,
            'variance_reduction_pct': variance_reduction,
            'users_with_reduced_variance': users_with_reduced_variance,
            'pct_users_improved': 100 * users_with_reduced_variance / users_with_both_data if users_with_both_data > 0 else 0
        }

        print(f"  Standard Deviation (Raw): {raw_std:.2f} kg")
        print(f"  Standard Deviation (Filtered): {filtered_std:.2f} kg")
        print(f"  Variance Reduction: {variance_reduction:.1f}%")
        print(f"  Users with reduced variance: {users_with_reduced_variance:,} ({100*users_with_reduced_variance/users_with_both_data:.1f}%)")

    # 2. Outlier Removal Statistics
    print("\n🎯 OUTLIER REMOVAL STATISTICS")
    print("-" * 40)

    total_raw_measurements = sum(d['raw_count'] for d in user_data.values())
    total_filtered_measurements = sum(d['filtered_count'] for d in user_data.values())
    total_removed = total_raw_measurements - total_filtered_measurements
    removal_rate = 100 * total_removed / total_raw_measurements if total_raw_measurements > 0 else 0

    # Identify extreme outliers (measurements removed that were > 4.5 kg from median)
    extreme_outliers_removed = 0
    moderate_outliers_removed = 0

    for user_id, data in user_data.items():
        if data['raw_data'].empty or data['filtered_data'].empty:
            continue

        raw_weights_set = set(data['raw_data']['weight'].values)
        filtered_weights_set = set(data['filtered_data']['weight'].values)
        removed_weights = raw_weights_set - filtered_weights_set

        if removed_weights and len(filtered_weights_set) > 0:
            median_weight = np.median(list(filtered_weights_set))
            for removed_weight in removed_weights:
                diff = abs(removed_weight - median_weight)
                if diff > 4.5:  # ~10 kg in kg
                    extreme_outliers_removed += 1
                elif diff > 2.3:  # ~5 kg in kg
                    moderate_outliers_removed += 1

    quality_metrics['outlier_removal'] = {
        'total_measurements_removed': total_removed,
        'removal_rate_pct': removal_rate,
        'extreme_outliers_removed': extreme_outliers_removed,
        'moderate_outliers_removed': moderate_outliers_removed,
        'measurements_per_user_raw': total_raw_measurements / len(user_data) if len(user_data) > 0 else 0,
        'measurements_per_user_filtered': total_filtered_measurements / len(user_data) if len(user_data) > 0 else 0
    }

    print(f"  Total measurements removed: {total_removed:,} ({removal_rate:.1f}%)")
    print(f"  Extreme outliers (>4.5 kg): {extreme_outliers_removed:,}")
    print(f"  Moderate outliers (2.3-4.5 kg): {moderate_outliers_removed:,}")
    print(f"  Avg measurements per user (raw): {total_raw_measurements/len(user_data):.1f}")
    print(f"  Avg measurements per user (filtered): {total_filtered_measurements/len(user_data):.1f}")

    # 3. Consistency Improvements
    print("\n✅ CONSISTENCY IMPROVEMENTS")
    print("-" * 40)

    if weight_changes_raw and weight_changes_filtered:
        # Analyze day-to-day weight changes
        raw_change_std = np.std(weight_changes_raw)
        filtered_change_std = np.std(weight_changes_filtered)

        # Count physiologically implausible changes (> 2.3 kg/day)
        implausible_raw = sum(1 for c in weight_changes_raw if abs(c) > 2.3)  # ~5 kg
        implausible_filtered = sum(1 for c in weight_changes_filtered if abs(c) > 2.3)

        quality_metrics['consistency_improvements'] = {
            'raw_daily_change_std': raw_change_std,
            'filtered_daily_change_std': filtered_change_std,
            'consistency_improvement_pct': 100 * (raw_change_std - filtered_change_std) / raw_change_std if raw_change_std > 0 else 0,
            'implausible_changes_raw': implausible_raw,
            'implausible_changes_filtered': implausible_filtered,
            'implausible_reduction_pct': 100 * (implausible_raw - implausible_filtered) / implausible_raw if implausible_raw > 0 else 0,
            'users_with_improved_consistency': users_with_improved_consistency
        }

        print(f"  Daily change std dev (raw): {raw_change_std:.2f} kg")
        print(f"  Daily change std dev (filtered): {filtered_change_std:.2f} kg")
        print(f"  Consistency improvement: {100*(raw_change_std-filtered_change_std)/raw_change_std:.1f}%")
        print(f"  Implausible changes (>2.3 kg/day):")
        print(f"    Raw: {implausible_raw:,}")
        print(f"    Filtered: {implausible_filtered:,}")
        print(f"    Reduction: {100*(implausible_raw-implausible_filtered)/implausible_raw:.1f}%" if implausible_raw > 0 else "    Reduction: N/A")
        print(f"  Users with improved consistency: {users_with_improved_consistency:,}")

    # 4. Weight Trajectory Analysis
    print("\n📈 WEIGHT TRAJECTORY ANALYSIS")
    print("-" * 40)

    trajectories_aligned = 0
    trajectories_divergent = 0
    trajectory_details = []

    for user_id, data in user_data.items():
        start_raw = data.get('closest_raw_weight')
        start_filtered = data.get('closest_filtered_weight')
        end_raw = data.get('latest_raw_weight')
        end_filtered = data.get('latest_filtered_weight')

        if all(x is not None for x in [start_raw, start_filtered, end_raw, end_filtered]):
            change_raw = end_raw - start_raw
            change_filtered = end_filtered - start_filtered

            # Check if trajectories align (same direction and similar magnitude)
            if (change_raw * change_filtered) > 0:  # Same direction
                if abs(abs(change_raw) - abs(change_filtered)) < 0.9:  # Similar magnitude (~2 kg in kg)
                    trajectories_aligned += 1
                else:
                    trajectories_divergent += 1
            else:
                trajectories_divergent += 1

            trajectory_details.append({
                'user_id': user_id,
                'change_raw': change_raw,
                'change_filtered': change_filtered,
                'aligned': (change_raw * change_filtered) > 0
            })

    quality_metrics['trajectory_analysis'] = {
        'trajectories_analyzed': len(trajectory_details),
        'trajectories_aligned': trajectories_aligned,
        'trajectories_divergent': trajectories_divergent,
        'alignment_rate_pct': 100 * trajectories_aligned / len(trajectory_details) if trajectory_details else 0
    }

    print(f"  Trajectories analyzed: {len(trajectory_details):,}")
    print(f"  Aligned trajectories: {trajectories_aligned:,} ({100*trajectories_aligned/len(trajectory_details):.1f}%)" if trajectory_details else "  Aligned trajectories: N/A")
    print(f"  Divergent trajectories: {trajectories_divergent:,}")

    # 5. Statistical Distribution Improvements
    print("\n📉 STATISTICAL DISTRIBUTION IMPROVEMENTS")
    print("-" * 40)

    if all_raw_weights and all_filtered_weights:
        from scipy import stats

        # Test for normality (filtered data should be more normal)
        _, raw_normality_p = stats.normaltest(all_raw_weights)
        _, filtered_normality_p = stats.normaltest(all_filtered_weights)

        # Calculate skewness and kurtosis
        raw_skew = stats.skew(all_raw_weights)
        filtered_skew = stats.skew(all_filtered_weights)
        raw_kurtosis = stats.kurtosis(all_raw_weights)
        filtered_kurtosis = stats.kurtosis(all_filtered_weights)

        # Calculate IQR and identify outliers using IQR method
        raw_q1, raw_q3 = np.percentile(all_raw_weights, [25, 75])
        raw_iqr = raw_q3 - raw_q1
        filtered_q1, filtered_q3 = np.percentile(all_filtered_weights, [25, 75])
        filtered_iqr = filtered_q3 - filtered_q1

        quality_metrics['statistical_improvements'] = {
            'raw_normality_p': raw_normality_p,
            'filtered_normality_p': filtered_normality_p,
            'normality_improved': filtered_normality_p > raw_normality_p,
            'raw_skewness': raw_skew,
            'filtered_skewness': filtered_skew,
            'skewness_reduction_pct': 100 * (abs(raw_skew) - abs(filtered_skew)) / abs(raw_skew) if raw_skew != 0 else 0,
            'raw_kurtosis': raw_kurtosis,
            'filtered_kurtosis': filtered_kurtosis,
            'raw_iqr': raw_iqr,
            'filtered_iqr': filtered_iqr
        }

        print(f"  Normality test p-values:")
        print(f"    Raw: {raw_normality_p:.4f}")
        print(f"    Filtered: {filtered_normality_p:.4f}")
        print(f"    Improved: {'Yes' if filtered_normality_p > raw_normality_p else 'No'}")
        print(f"  Skewness:")
        print(f"    Raw: {raw_skew:.3f}")
        print(f"    Filtered: {filtered_skew:.3f}")
        print(f"    Reduction: {100*(abs(raw_skew)-abs(filtered_skew))/abs(raw_skew):.1f}%" if raw_skew != 0 else "    Reduction: N/A")
        print(f"  Kurtosis:")
        print(f"    Raw: {raw_kurtosis:.3f}")
        print(f"    Filtered: {filtered_kurtosis:.3f}")
        print(f"  IQR:")
        print(f"    Raw: {raw_iqr:.2f} kg")
        print(f"    Filtered: {filtered_iqr:.2f} kg")

    # 6. Summary Score
    print("\n🏆 OVERALL DATA QUALITY SCORE")
    print("-" * 40)

    # Calculate overall quality improvement score (0-100)
    quality_score = 0
    score_components = []

    if quality_metrics['noise_reduction'].get('variance_reduction_pct', 0) > 0:
        score_components.append(min(quality_metrics['noise_reduction']['variance_reduction_pct'], 25))

    if quality_metrics['consistency_improvements'].get('implausible_reduction_pct', 0) > 0:
        score_components.append(min(quality_metrics['consistency_improvements']['implausible_reduction_pct'] * 0.25, 25))

    if quality_metrics['trajectory_analysis'].get('alignment_rate_pct', 0) > 70:
        score_components.append(25)
    elif quality_metrics['trajectory_analysis'].get('alignment_rate_pct', 0) > 50:
        score_components.append(15)

    if quality_metrics['statistical_improvements'].get('normality_improved', False):
        score_components.append(25)

    quality_score = min(sum(score_components), 100)

    quality_metrics['overall_quality_score'] = quality_score
    quality_metrics['score_components'] = score_components

    print(f"  Overall Quality Improvement Score: {quality_score:.0f}/100")
    print(f"  Grade: ", end="")
    if quality_score >= 90:
        print("Excellent (A)")
    elif quality_score >= 80:
        print("Very Good (B)")
    elif quality_score >= 70:
        print("Good (C)")
    elif quality_score >= 60:
        print("Fair (D)")
    else:
        print("Needs Improvement (F)")

    print("\n" + "="*60)

    return quality_metrics


def find_latest_weight_values(user_data: Dict, reference_date: str = '2025-09-05') -> Dict:
    """
    For each user, find the latest (most recent) weight measurement.
    Analyzes both raw and filtered data.

    Args:
        user_data: Dictionary with user data
        reference_date: The date to consider as "today"

    Returns:
        Updated user_data dictionary with latest weight information
    """
    print("\nAnalyzing latest weight measurements...")

    ref_date = pd.to_datetime(reference_date)
    users_with_latest = 0
    no_measurements = 0

    for user_id, data in user_data.items():
        # Find latest raw weight
        if not data['raw_data'].empty:
            raw_df = data['raw_data'].copy()
            raw_df = raw_df[raw_df['effectiveDateTime'] <= ref_date]  # Only consider measurements before reference date

            if not raw_df.empty:
                latest_raw_idx = raw_df['effectiveDateTime'].idxmax()
                latest_raw = raw_df.loc[latest_raw_idx]

                data['latest_raw_weight'] = latest_raw['weight']
                data['latest_raw_date'] = latest_raw['effectiveDateTime']
                data['days_since_latest_raw'] = (ref_date - latest_raw['effectiveDateTime']).total_seconds() / 86400
            else:
                data['latest_raw_weight'] = None
                data['latest_raw_date'] = None
                data['days_since_latest_raw'] = None
        else:
            data['latest_raw_weight'] = None
            data['latest_raw_date'] = None
            data['days_since_latest_raw'] = None

        # Find latest filtered weight
        if not data['filtered_data'].empty:
            filtered_df = data['filtered_data'].copy()
            filtered_df = filtered_df[filtered_df['effectiveDateTime'] <= ref_date]

            if not filtered_df.empty:
                latest_filtered_idx = filtered_df['effectiveDateTime'].idxmax()
                latest_filtered = filtered_df.loc[latest_filtered_idx]

                data['latest_filtered_weight'] = latest_filtered['weight']
                data['latest_filtered_date'] = latest_filtered['effectiveDateTime']
                data['days_since_latest_filtered'] = (ref_date - latest_filtered['effectiveDateTime']).total_seconds() / 86400
                data['latest_filtered_quality'] = latest_filtered.get('quality_score', None)
            else:
                data['latest_filtered_weight'] = None
                data['latest_filtered_date'] = None
                data['days_since_latest_filtered'] = None
                data['latest_filtered_quality'] = None
        else:
            data['latest_filtered_weight'] = None
            data['latest_filtered_date'] = None
            data['days_since_latest_filtered'] = None
            data['latest_filtered_quality'] = None

        if data['latest_raw_weight'] is not None or data['latest_filtered_weight'] is not None:
            users_with_latest += 1
        else:
            no_measurements += 1

    # Summary statistics
    print(f"  Reference date: {reference_date}")
    print(f"  Users with latest measurements: {users_with_latest:,}")
    print(f"  Users with no measurements: {no_measurements:,}")

    # Calculate recency statistics
    raw_days_since = [d['days_since_latest_raw'] for d in user_data.values()
                      if d.get('days_since_latest_raw') is not None]
    filtered_days_since = [d['days_since_latest_filtered'] for d in user_data.values()
                          if d.get('days_since_latest_filtered') is not None]

    if raw_days_since:
        print(f"\n  Raw measurements recency:")
        print(f"    Average days since last: {np.mean(raw_days_since):.1f}")
        print(f"    Median days since last: {np.median(raw_days_since):.1f}")
        print(f"    Within last 7 days: {sum(d <= 7 for d in raw_days_since):,}")
        print(f"    Within last 30 days: {sum(d <= 30 for d in raw_days_since):,}")

    if filtered_days_since:
        print(f"\n  Filtered measurements recency:")
        print(f"    Average days since last: {np.mean(filtered_days_since):.1f}")
        print(f"    Median days since last: {np.median(filtered_days_since):.1f}")
        print(f"    Within last 7 days: {sum(d <= 7 for d in filtered_days_since):,}")
        print(f"    Within last 30 days: {sum(d <= 30 for d in filtered_days_since):,}")

    return user_data


def export_weights_to_csv(user_data: Dict, employer_filter: str = None, quality_metrics: Dict = None) -> str:
    """
    Export all weight measurements (start, 90-day, latest) to CSV for comparison.

    Args:
        user_data: Dictionary containing user analysis data
        employer_filter: Optional employer name filter that was applied

    Returns:
        Path to the created CSV file
    """
    # Create report_output directory if it doesn't exist
    output_dir = Path("report_output")
    output_dir.mkdir(exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if employer_filter:
        filename = f"weight_comparison_{employer_filter}_{timestamp}.csv"
    else:
        filename = f"weight_comparison_all_{timestamp}.csv"

    output_path = output_dir / filename

    # Prepare data for CSV
    csv_data = []

    for user_id, data in user_data.items():
        row = {
            'user_id': user_id,
            'employer_name': data.get('employer_name', ''),
            'start_date': data.get('start_date', ''),
            'days_in_program': data.get('days_in_program', ''),

            # Start weights
            'start_weight_raw': data.get('closest_raw_weight', ''),
            'start_weight_filtered': data.get('closest_filtered_weight', ''),
            'start_weight_diff': '',

            # 90-day weights
            'weight_90d_raw': data.get('weight_90d_raw', ''),
            'weight_90d_filtered': data.get('weight_90d_filtered', ''),
            'weight_90d_diff': '',

            # Latest weights
            'latest_weight_raw': data.get('latest_raw_weight', ''),
            'latest_weight_filtered': data.get('latest_filtered_weight', ''),
            'latest_weight_diff': '',

            # Weight changes over time (these are key metrics)
            'change_start_to_latest_raw': '',
            'change_start_to_latest_filtered': '',
            'change_total_diff': '',  # Difference between raw and filtered total change

            # Measurement counts
            'total_measurements': data.get('raw_count', 0),
            'measurements_removed': data.get('raw_count', 0) - data.get('filtered_count', 0),
            'removal_percentage': ''
        }

        # Calculate differences
        if row['start_weight_raw'] and row['start_weight_filtered']:
            row['start_weight_diff'] = float(row['start_weight_filtered']) - float(row['start_weight_raw'])

        if row['weight_90d_raw'] and row['weight_90d_filtered']:
            row['weight_90d_diff'] = float(row['weight_90d_filtered']) - float(row['weight_90d_raw'])

        if row['latest_weight_raw'] and row['latest_weight_filtered']:
            row['latest_weight_diff'] = float(row['latest_weight_filtered']) - float(row['latest_weight_raw'])

        # Calculate weight changes over time
        if row['start_weight_raw'] and row['latest_weight_raw']:
            row['change_start_to_latest_raw'] = float(row['latest_weight_raw']) - float(row['start_weight_raw'])

        if row['start_weight_filtered'] and row['latest_weight_filtered']:
            row['change_start_to_latest_filtered'] = float(row['latest_weight_filtered']) - float(row['start_weight_filtered'])

        # Calculate the difference in total change (key metric)
        if row['change_start_to_latest_raw'] != '' and row['change_start_to_latest_filtered'] != '':
            row['change_total_diff'] = float(row['change_start_to_latest_filtered']) - float(row['change_start_to_latest_raw'])

        # Calculate removal percentage
        if row['total_measurements'] > 0:
            row['removal_percentage'] = 100 * row['measurements_removed'] / row['total_measurements']

        # Add indicator flags for significant differences
        row['FLAG_significant_diff'] = ''
        row['FLAG_trajectory_divergence'] = ''
        row['FLAG_high_removal'] = ''
        row['FLAG_missing_data'] = ''
        row['ALERT_LEVEL'] = 'OK'
        row['ALERT_REASONS'] = ''

        alert_reasons = []
        alert_score = 0
        sig_diffs = []

        # Check weight differences at each time point
        if row['start_weight_diff'] != '':
            abs_diff = abs(float(row['start_weight_diff']))
            if abs_diff > 2.3:  # ~5 kg in kg
                sig_diffs.append(f'Start:{abs_diff:.1f}kg')
                alert_score += 2
            elif abs_diff > 0.5:  # ~1 lb in kg
                alert_score += 1

        if row['weight_90d_diff'] != '':
            abs_diff = abs(float(row['weight_90d_diff']))
            if abs_diff > 2.3:  # ~5 kg in kg
                sig_diffs.append(f'90d:{abs_diff:.1f}kg')
                alert_score += 2
            elif abs_diff > 0.5:  # ~1 lb in kg
                alert_score += 1

        if row['latest_weight_diff'] != '':
            abs_diff = abs(float(row['latest_weight_diff']))
            if abs_diff > 2.3:  # ~5 kg in kg
                sig_diffs.append(f'Latest:{abs_diff:.1f}kg')
                alert_score += 2
            elif abs_diff > 0.5:  # ~1 lb in kg
                alert_score += 1

        if sig_diffs:
            row['FLAG_significant_diff'] = ', '.join(sig_diffs)
            alert_reasons.append('Sig diffs: ' + ', '.join(sig_diffs))

        # Check for trajectory divergence (raw and filtered show different trends)
        if row['change_total_diff'] != '':
            diff = float(row['change_total_diff'])
            raw_change = float(row['change_start_to_latest_raw']) if row['change_start_to_latest_raw'] != '' else 0
            filtered_change = float(row['change_start_to_latest_filtered']) if row['change_start_to_latest_filtered'] != '' else 0

            # Check if one shows gain and other shows loss
            if (raw_change > 0 and filtered_change < 0) or (raw_change < 0 and filtered_change > 0):
                row['FLAG_trajectory_divergence'] = 'OPPOSITE'
                alert_reasons.append('Opposite trajectories')
                alert_score += 3
            # Check if difference in total change is significant
            elif abs(diff) > 2.3:  # ~5 kg in kg
                row['FLAG_trajectory_divergence'] = f'DIFF:{abs(diff):.1f}kg'
                alert_reasons.append(f'Trajectory diff {abs(diff):.1f}kg')
                alert_score += 2

        # Check for high removal rate
        if row['removal_percentage'] != '' and float(row['removal_percentage']) > 50:
            row['FLAG_high_removal'] = f'{row["removal_percentage"]:.0f}%'
            alert_reasons.append(f'{row["removal_percentage"]:.0f}% removed')
            alert_score += 1

        # Check for missing filtered data at key points
        missing_points = []
        if row['start_weight_filtered'] == '' and row['start_weight_raw'] != '':
            missing_points.append('start')
        if row['weight_90d_filtered'] == '' and row['weight_90d_raw'] != '':
            missing_points.append('90d')
        if row['latest_weight_filtered'] == '' and row['latest_weight_raw'] != '':
            missing_points.append('latest')

        if missing_points:
            row['FLAG_missing_data'] = ','.join(missing_points)
            alert_reasons.append(f'Missing: {",".join(missing_points)}')
            alert_score += len(missing_points)

        # Set overall alert level based on score
        if alert_score >= 5:
            row['ALERT_LEVEL'] = 'HIGH'
        elif alert_score >= 3:
            row['ALERT_LEVEL'] = 'MEDIUM'
        elif alert_score >= 1:
            row['ALERT_LEVEL'] = 'LOW'

        row['ALERT_REASONS'] = '; '.join(alert_reasons)

        csv_data.append(row)

    # Convert to DataFrame and save
    df = pd.DataFrame(csv_data)

    # Reorder columns to put alert columns first
    alert_cols = ['ALERT_LEVEL', 'ALERT_REASONS']
    id_cols = ['user_id', 'employer_name', 'days_in_program']
    flag_cols = [col for col in df.columns if col.startswith('FLAG_')]
    weight_cols = ['start_weight_raw', 'start_weight_filtered', 'start_weight_diff',
                   'weight_90d_raw', 'weight_90d_filtered', 'weight_90d_diff',
                   'latest_weight_raw', 'latest_weight_filtered', 'latest_weight_diff']
    change_cols = ['change_start_to_latest_raw', 'change_start_to_latest_filtered', 'change_total_diff']
    measure_cols = ['total_measurements', 'measurements_removed', 'removal_percentage']
    other_cols = [col for col in df.columns if col not in alert_cols + id_cols + flag_cols + weight_cols + change_cols + measure_cols]

    # Combine in logical order
    ordered_cols = alert_cols + id_cols + flag_cols + weight_cols + change_cols + measure_cols + other_cols
    # Remove duplicates while preserving order
    ordered_cols = list(dict.fromkeys(ordered_cols))
    # Only keep columns that exist
    ordered_cols = [col for col in ordered_cols if col in df.columns]

    df = df[ordered_cols]

    # Sort by alert level (HIGH first), then by user_id
    alert_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2, 'OK': 3}
    df['alert_sort'] = df['ALERT_LEVEL'].map(alert_order)
    df = df.sort_values(['alert_sort', 'user_id'])
    df = df.drop('alert_sort', axis=1)

    # Save to CSV
    df.to_csv(output_path, index=False, float_format='%.2f')

    print(f"\n📊 Weight comparison CSV exported to: {output_path}")
    print(f"  Total users in CSV: {len(df):,}")

    # Print summary statistics
    if not df.empty:
        # Alert summary
        print("\n  Alert Level Summary:")
        alert_counts = df['ALERT_LEVEL'].value_counts()
        for level in ['HIGH', 'MEDIUM', 'LOW', 'OK']:
            count = alert_counts.get(level, 0)
            pct = 100 * count / len(df)
            print(f"    {level}: {count:,} users ({pct:.1f}%)")

        # Flag summary
        print("\n  Flag Summary:")
        for flag_col in [col for col in df.columns if col.startswith('FLAG_')]:
            flagged = (df[flag_col] != '').sum()
            if flagged > 0:
                pct = 100 * flagged / len(df)
                flag_name = flag_col.replace('FLAG_', '').replace('_', ' ').title()
                print(f"    {flag_name}: {flagged:,} users ({pct:.1f}%)")

        # Add quality metrics summary if available
        if quality_metrics:
            print("\n  Data Quality Improvement Summary:")
            if quality_metrics.get('overall_quality_score'):
                score = quality_metrics['overall_quality_score']
                print(f"    Overall Quality Score: {score:.0f}/100", end=" - ")
                if score >= 90:
                    print("Excellent (A)")
                elif score >= 80:
                    print("Very Good (B)")
                elif score >= 70:
                    print("Good (C)")
                elif score >= 60:
                    print("Fair (D)")
                else:
                    print("Needs Improvement (F)")

            if quality_metrics.get('outlier_removal'):
                or_m = quality_metrics['outlier_removal']
                print(f"    Measurements removed: {or_m.get('total_measurements_removed', 0):,} ({or_m.get('removal_rate_pct', 0):.1f}%)")
                print(f"    Extreme outliers removed: {or_m.get('extreme_outliers_removed', 0):,}")

            if quality_metrics.get('noise_reduction'):
                nr = quality_metrics['noise_reduction']
                print(f"    Variance reduction: {nr.get('variance_reduction_pct', 0):.1f}%")

            if quality_metrics.get('consistency_improvements'):
                ci = quality_metrics['consistency_improvements']
                print(f"    Consistency improvement: {ci.get('consistency_improvement_pct', 0):.1f}%")

        print("\n  Summary of weight differences (filtered - raw):")
        for col in ['start_weight_diff', 'weight_90d_diff', 'latest_weight_diff']:
            if col in df.columns:
                valid_diffs = df[col].replace('', np.nan).dropna()
                if not valid_diffs.empty:
                    col_name = col.replace('_diff', '').replace('_', ' ').title()
                    print(f"    {col_name}:")
                    print(f"      Mean: {valid_diffs.mean():.2f} kg")
                    print(f"      Median: {valid_diffs.median():.2f} kg")
                    print(f"      Std Dev: {valid_diffs.std():.2f} kg")

        # Summary of trajectory differences
        if 'change_total_diff' in df.columns:
            valid_traj = df['change_total_diff'].replace('', np.nan).dropna()
            if not valid_traj.empty:
                print(f"\n  Trajectory Difference (filtered vs raw total change):")
                print(f"    Mean: {valid_traj.mean():.2f} kg")
                print(f"    Median: {valid_traj.median():.2f} kg")
                print(f"    Users with opposite trends: {(df['FLAG_trajectory_divergence'] == 'OPPOSITE').sum():,}")

    return str(output_path)


def export_analysis_to_markdown(user_data: Dict, employer_filter: str = None, quality_metrics: Dict = None) -> str:
    """
    Export the analysis results to a markdown file with focus on raw vs filtered differences.

    Args:
        user_data: Dictionary containing user analysis data
        employer_filter: Optional employer name filter that was applied

    Returns:
        Path to the created markdown file
    """
    # Create report_output directory if it doesn't exist
    output_dir = Path("report_output")
    output_dir.mkdir(exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if employer_filter:
        filename = f"weight_analysis_{employer_filter}_{timestamp}.md"
    else:
        filename = f"weight_analysis_all_{timestamp}.md"

    output_path = output_dir / filename

    # Collect statistics for the report
    total_users = len(user_data)
    users_with_start = sum(1 for d in user_data.values() if d.get('start_date') is not None)
    users_with_raw = sum(1 for d in user_data.values() if d.get('closest_raw_weight') is not None)
    users_with_filtered = sum(1 for d in user_data.values() if d.get('closest_filtered_weight') is not None)

    # Calculate detailed statistics
    raw_days = [abs(d['days_from_start_raw']) for d in user_data.values()
                if d.get('days_from_start_raw') is not None]
    filtered_days = [abs(d['days_from_start_filtered']) for d in user_data.values()
                     if d.get('days_from_start_filtered') is not None]

    raw_weights = [d['closest_raw_weight'] for d in user_data.values()
                   if d.get('closest_raw_weight') is not None]
    filtered_weights = [d['closest_filtered_weight'] for d in user_data.values()
                        if d.get('closest_filtered_weight') is not None]

    quality_scores = [d['closest_filtered_quality'] for d in user_data.values()
                      if d.get('closest_filtered_quality') is not None]

    # Calculate differences for users with both raw and filtered
    weight_differences = []
    users_with_differences = []
    for user_id, data in user_data.items():
        if data.get('closest_raw_weight') is not None and data.get('closest_filtered_weight') is not None:
            diff = data['closest_filtered_weight'] - data['closest_raw_weight']
            weight_differences.append(diff)
            users_with_differences.append({
                'user_id': user_id,
                'difference': diff,
                'raw_weight': data['closest_raw_weight'],
                'filtered_weight': data['closest_filtered_weight'],
                'quality_score': data.get('closest_filtered_quality', 0)
            })

    # Group by employer
    employer_groups = {}
    for user_id, data in user_data.items():
        employer = data.get('employer_name', 'Unknown')
        if employer not in employer_groups:
            employer_groups[employer] = []
        employer_groups[employer].append(data)

    # Write markdown report
    with open(output_path, 'w') as f:
        f.write("# Weight Analysis Report: Raw vs Filtered Comparison\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if employer_filter:
            f.write(f"**Employer Filter:** {employer_filter}\n")
        f.write("\n---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        total_raw_measurements = sum(d['raw_count'] for d in user_data.values())
        total_filtered_measurements = sum(d['filtered_count'] for d in user_data.values())
        measurements_removed = total_raw_measurements - total_filtered_measurements
        removal_rate = 100 * measurements_removed / total_raw_measurements if total_raw_measurements > 0 else 0

        f.write(f"- **Total Users:** {total_users:,}\n")
        f.write(f"- **Raw Measurements:** {total_raw_measurements:,}\n")
        f.write(f"- **Filtered Measurements:** {total_filtered_measurements:,}\n")
        f.write(f"- **Measurements Removed:** {measurements_removed:,} ({removal_rate:.1f}%)\n")
        f.write(f"- **Average Removal Rate per User:** {removal_rate:.1f}%\n\n")

        # Key Differences Section
        f.write("## 🔍 Key Differences: Raw vs Filtered\n\n")

        if weight_differences:
            f.write("### Weight Value Differences at Start\n\n")
            abs_diffs = [abs(d) for d in weight_differences]
            f.write(f"- **Users with Both Measurements:** {len(weight_differences):,}\n")
            f.write(f"- **Average Difference:** {np.mean(weight_differences):.2f} kg\n")
            f.write(f"- **Median Difference:** {np.median(weight_differences):.2f} kg\n")
            f.write(f"- **Average Absolute Difference:** {np.mean(abs_diffs):.2f} kg\n")
            f.write(f"- **Max Absolute Difference:** {np.max(abs_diffs):.2f} kg\n")

            # Count significant differences
            sig_diffs = sum(1 for d in abs_diffs if d > 0.5)  # ~1 lb in kg
            large_diffs = sum(1 for d in abs_diffs if d > 2.3)  # ~5 kg in kg
            f.write(f"- **Differences > 0.5 kg:** {sig_diffs:,} ({100*sig_diffs/len(weight_differences):.1f}%)\n")
            f.write(f"- **Differences > 2.3 kg:** {large_diffs:,} ({100*large_diffs/len(weight_differences):.1f}%)\n\n")

        # Weight Loss/Gain Analysis
        f.write("### 🏋️ Weight Loss Analysis\n\n")

        # Calculate average weights and weight changes
        avg_start_raw = []
        avg_start_filtered = []
        avg_90d_raw = []
        avg_90d_filtered = []
        avg_latest_raw = []
        avg_latest_filtered = []

        weight_loss_90d_raw = []
        weight_loss_90d_filtered = []
        weight_loss_latest_raw = []
        weight_loss_latest_filtered = []

        for user_id, data in user_data.items():
            # Collect start weights
            if data.get('closest_raw_weight') is not None:
                avg_start_raw.append(data['closest_raw_weight'])
            if data.get('closest_filtered_weight') is not None:
                avg_start_filtered.append(data['closest_filtered_weight'])

            # Collect 90-day weights
            if data.get('weight_90d_raw') is not None:
                avg_90d_raw.append(data['weight_90d_raw'])
            if data.get('weight_90d_filtered') is not None:
                avg_90d_filtered.append(data['weight_90d_filtered'])

            # Collect latest weights
            if data.get('latest_raw_weight') is not None:
                avg_latest_raw.append(data['latest_raw_weight'])
            if data.get('latest_filtered_weight') is not None:
                avg_latest_filtered.append(data['latest_filtered_weight'])

            # Calculate 90-day weight loss (start to 90 days)
            if data.get('closest_raw_weight') is not None and data.get('weight_90d_raw') is not None:
                loss_90d_raw = data['closest_raw_weight'] - data['weight_90d_raw']
                weight_loss_90d_raw.append(loss_90d_raw)

            if data.get('closest_filtered_weight') is not None and data.get('weight_90d_filtered') is not None:
                loss_90d_filtered = data['closest_filtered_weight'] - data['weight_90d_filtered']
                weight_loss_90d_filtered.append(loss_90d_filtered)

            # Calculate total weight loss (start to latest)
            if data.get('closest_raw_weight') is not None and data.get('latest_raw_weight') is not None:
                loss_latest_raw = data['closest_raw_weight'] - data['latest_raw_weight']
                weight_loss_latest_raw.append(loss_latest_raw)

            if data.get('closest_filtered_weight') is not None and data.get('latest_filtered_weight') is not None:
                loss_latest_filtered = data['closest_filtered_weight'] - data['latest_filtered_weight']
                weight_loss_latest_filtered.append(loss_latest_filtered)

        # Weight Averages Table
        f.write("#### Average Weights at Key Points\n\n")
        f.write("| Time Point | Raw Data (kg) | Filtered Data (kg) | Difference | N (Raw) | N (Filtered) |\n")
        f.write("|------------|----------------|---------------------|------------|---------|--------------|\n")

        if avg_start_raw and avg_start_filtered:
            raw_start = np.mean(avg_start_raw)
            filtered_start = np.mean(avg_start_filtered)
            diff_start = filtered_start - raw_start
            f.write(f"| **Start Weight** | {raw_start:.1f} ± {np.std(avg_start_raw):.1f} | ")
            f.write(f"{filtered_start:.1f} ± {np.std(avg_start_filtered):.1f} | ")
            f.write(f"{diff_start:+.1f} | {len(avg_start_raw):,} | {len(avg_start_filtered):,} |\n")

        if avg_90d_raw and avg_90d_filtered:
            raw_90d = np.mean(avg_90d_raw)
            filtered_90d = np.mean(avg_90d_filtered)
            diff_90d = filtered_90d - raw_90d
            f.write(f"| **90-Day Weight** | {raw_90d:.1f} ± {np.std(avg_90d_raw):.1f} | ")
            f.write(f"{filtered_90d:.1f} ± {np.std(avg_90d_filtered):.1f} | ")
            f.write(f"{diff_90d:+.1f} | {len(avg_90d_raw):,} | {len(avg_90d_filtered):,} |\n")

        if avg_latest_raw and avg_latest_filtered:
            raw_latest = np.mean(avg_latest_raw)
            filtered_latest = np.mean(avg_latest_filtered)
            diff_latest = filtered_latest - raw_latest
            f.write(f"| **Latest Weight** | {raw_latest:.1f} ± {np.std(avg_latest_raw):.1f} | ")
            f.write(f"{filtered_latest:.1f} ± {np.std(avg_latest_filtered):.1f} | ")
            f.write(f"{diff_latest:+.1f} | {len(avg_latest_raw):,} | {len(avg_latest_filtered):,} |\n")

        f.write("\n*Values shown as mean ± standard deviation*\n\n")

        # Weight Loss Comparison Table
        f.write("#### Average Weight Loss Outcomes\n\n")
        f.write("| Period | Raw Data (kg) | Filtered Data (kg) | Difference | Better Outcome |\n")
        f.write("|--------|----------------|---------------------|------------|----------------|\n")

        if weight_loss_90d_raw and weight_loss_90d_filtered:
            raw_loss_90d = np.mean(weight_loss_90d_raw)
            filtered_loss_90d = np.mean(weight_loss_90d_filtered)
            diff_loss_90d = filtered_loss_90d - raw_loss_90d
            better_90d = "Filtered" if filtered_loss_90d > raw_loss_90d else "Raw" if raw_loss_90d > filtered_loss_90d else "Same"

            # Calculate percentage of users who lost weight
            raw_losers_90d = sum(1 for x in weight_loss_90d_raw if x > 0)
            filtered_losers_90d = sum(1 for x in weight_loss_90d_filtered if x > 0)

            f.write(f"| **Start → 90 Days** | {raw_loss_90d:.1f} ± {np.std(weight_loss_90d_raw):.1f} | ")
            f.write(f"{filtered_loss_90d:.1f} ± {np.std(weight_loss_90d_filtered):.1f} | ")
            f.write(f"{diff_loss_90d:+.1f} | **{better_90d}** |\n")

            f.write(f"| *% Lost Weight* | {100*raw_losers_90d/len(weight_loss_90d_raw):.1f}% | ")
            f.write(f"{100*filtered_losers_90d/len(weight_loss_90d_filtered):.1f}% | ")
            f.write(f"{100*filtered_losers_90d/len(weight_loss_90d_filtered) - 100*raw_losers_90d/len(weight_loss_90d_raw):+.1f}% | - |\n")

        if weight_loss_latest_raw and weight_loss_latest_filtered:
            raw_loss_latest = np.mean(weight_loss_latest_raw)
            filtered_loss_latest = np.mean(weight_loss_latest_filtered)
            diff_loss_latest = filtered_loss_latest - raw_loss_latest
            better_latest = "Filtered" if filtered_loss_latest > raw_loss_latest else "Raw" if raw_loss_latest > filtered_loss_latest else "Same"

            # Calculate percentage of users who lost weight
            raw_losers_latest = sum(1 for x in weight_loss_latest_raw if x > 0)
            filtered_losers_latest = sum(1 for x in weight_loss_latest_filtered if x > 0)

            f.write(f"| **Start → Latest** | {raw_loss_latest:.1f} ± {np.std(weight_loss_latest_raw):.1f} | ")
            f.write(f"{filtered_loss_latest:.1f} ± {np.std(weight_loss_latest_filtered):.1f} | ")
            f.write(f"{diff_loss_latest:+.1f} | **{better_latest}** |\n")

            f.write(f"| *% Lost Weight* | {100*raw_losers_latest/len(weight_loss_latest_raw):.1f}% | ")
            f.write(f"{100*filtered_losers_latest/len(weight_loss_latest_filtered):.1f}% | ")
            f.write(f"{100*filtered_losers_latest/len(weight_loss_latest_filtered) - 100*raw_losers_latest/len(weight_loss_latest_raw):+.1f}% | - |\n")

        f.write("\n*Positive values indicate weight loss, negative values indicate weight gain*\n\n")

        # Distribution of Weight Loss
        if weight_loss_latest_raw and weight_loss_latest_filtered:
            f.write("#### Weight Loss Distribution (Start → Latest)\n\n")

            # Categorize weight changes (converting from kg to kg thresholds)
            def categorize_weight_change(changes):
                gained_5_plus = sum(1 for x in changes if x <= -2.3)  # gained 5+ kg
                gained_0_5 = sum(1 for x in changes if -2.3 < x < 0)  # gained 0-5 kg
                maintained = sum(1 for x in changes if 0 <= x < 0.9)  # maintained ±2 kg
                lost_2_5 = sum(1 for x in changes if 0.9 <= x < 2.3)  # lost 2-5 kg
                lost_5_10 = sum(1 for x in changes if 2.3 <= x < 4.5)  # lost 5-10 kg
                lost_10_plus = sum(1 for x in changes if x >= 4.5)  # lost 10+ kg
                return gained_5_plus, gained_0_5, maintained, lost_2_5, lost_5_10, lost_10_plus

            raw_cats = categorize_weight_change(weight_loss_latest_raw)
            filtered_cats = categorize_weight_change(weight_loss_latest_filtered)

            f.write("| Category | Raw Data | Filtered Data | Difference |\n")
            f.write("|----------|----------|---------------|------------|\n")

            categories = [
                ("Lost 4.5+ kg", raw_cats[5], filtered_cats[5]),  # 10+ kg
                ("Lost 2.3-4.5 kg", raw_cats[4], filtered_cats[4]),  # 5-10 kg
                ("Lost 0.9-2.3 kg", raw_cats[3], filtered_cats[3]),  # 2-5 kg
                ("Maintained (±0.9 kg)", raw_cats[2], filtered_cats[2]),  # ±2 kg
                ("Gained 0-2.3 kg", raw_cats[1], filtered_cats[1]),  # 0-5 kg
                ("Gained 2.3+ kg", raw_cats[0], filtered_cats[0])  # 5+ kg
            ]

            total_raw = len(weight_loss_latest_raw)
            total_filtered = len(weight_loss_latest_filtered)

            for cat_name, raw_count, filtered_count in categories:
                raw_pct = 100 * raw_count / total_raw
                filtered_pct = 100 * filtered_count / total_filtered
                diff_pct = filtered_pct - raw_pct
                f.write(f"| {cat_name} | {raw_count:,} ({raw_pct:.1f}%) | ")
                f.write(f"{filtered_count:,} ({filtered_pct:.1f}%) | ")
                f.write(f"{diff_pct:+.1f}% |\n")

            f.write("\n")

        # Latest Weight Analysis
        f.write("### 📍 Latest Weight Values\n\n")

        latest_raw_weights = [d['latest_raw_weight'] for d in user_data.values()
                             if d.get('latest_raw_weight') is not None]
        latest_filtered_weights = [d['latest_filtered_weight'] for d in user_data.values()
                                  if d.get('latest_filtered_weight') is not None]

        if latest_raw_weights and latest_filtered_weights:
            latest_differences = []
            for user_id, data in user_data.items():
                if data.get('latest_raw_weight') is not None and data.get('latest_filtered_weight') is not None:
                    diff = data['latest_filtered_weight'] - data['latest_raw_weight']
                    latest_differences.append(diff)

            f.write(f"- **Users with Latest Raw Weight:** {len(latest_raw_weights):,}\n")
            f.write(f"- **Users with Latest Filtered Weight:** {len(latest_filtered_weights):,}\n")
            f.write(f"- **Users with Both Latest Weights:** {len(latest_differences):,}\n\n")

            if latest_differences:
                abs_latest_diffs = [abs(d) for d in latest_differences]
                f.write("#### Latest Weight Comparison\n\n")
                f.write(f"- **Mean Latest Raw Weight:** {np.mean(latest_raw_weights):.1f} kg\n")
                f.write(f"- **Mean Latest Filtered Weight:** {np.mean(latest_filtered_weights):.1f} kg\n")
                f.write(f"- **Average Difference:** {np.mean(latest_differences):.2f} kg\n")
                f.write(f"- **Median Difference:** {np.median(latest_differences):.2f} kg\n")
                f.write(f"- **Average Absolute Difference:** {np.mean(abs_latest_diffs):.2f} kg\n")

                sig_latest = sum(1 for d in abs_latest_diffs if d > 0.5)  # ~1 lb in kg
                large_latest = sum(1 for d in abs_latest_diffs if d > 2.3)  # ~5 lbs in kg
                f.write(f"- **Differences > 0.5 kg:** {sig_latest:,} ({100*sig_latest/len(latest_differences):.1f}%)\n")
                f.write(f"- **Differences > 2.3 kg:** {large_latest:,} ({100*large_latest/len(latest_differences):.1f}%)\n\n")

        # Comparative Table
        f.write("### 📊 Side-by-Side Comparison\n\n")
        f.write("| Metric | Raw Data | Filtered Data | Difference |\n")
        f.write("|--------|----------|---------------|------------|\n")

        # Measurements count
        avg_raw_count = np.mean([d['raw_count'] for d in user_data.values()])
        avg_filtered_count = np.mean([d['filtered_count'] for d in user_data.values()])
        f.write(f"| **Total Measurements** | {total_raw_measurements:,} | {total_filtered_measurements:,} | -{measurements_removed:,} ({removal_rate:.1f}%) |\n")
        f.write(f"| **Avg per User** | {avg_raw_count:.1f} | {avg_filtered_count:.1f} | -{avg_raw_count - avg_filtered_count:.1f} |\n")

        # Weight statistics comparison
        if raw_weights and filtered_weights:
            raw_mean = np.mean(raw_weights)
            filtered_mean = np.mean(filtered_weights)
            raw_median = np.median(raw_weights)
            filtered_median = np.median(filtered_weights)
            raw_std = np.std(raw_weights)
            filtered_std = np.std(filtered_weights)

            f.write(f"| **Mean Weight** | {raw_mean:.1f} kg | {filtered_mean:.1f} kg | {filtered_mean - raw_mean:+.1f} kg |\n")
            f.write(f"| **Median Weight** | {raw_median:.1f} kg | {filtered_median:.1f} kg | {filtered_median - raw_median:+.1f} kg |\n")
            f.write(f"| **Std Dev** | {raw_std:.1f} kg | {filtered_std:.1f} kg | {filtered_std - raw_std:+.1f} kg |\n")
            f.write(f"| **Min Weight** | {np.min(raw_weights):.1f} kg | {np.min(filtered_weights):.1f} kg | {np.min(filtered_weights) - np.min(raw_weights):+.1f} kg |\n")
            f.write(f"| **Max Weight** | {np.max(raw_weights):.1f} kg | {np.max(filtered_weights):.1f} kg | {np.max(filtered_weights) - np.max(raw_weights):+.1f} kg |\n")

        # Timing comparison
        if raw_days and filtered_days:
            f.write(f"| **Avg Days from Start** | {np.mean(raw_days):.1f} | {np.mean(filtered_days):.1f} | {np.mean(filtered_days) - np.mean(raw_days):+.1f} |\n")
            f.write(f"| **Within 7 days** | {100*sum(d <= 7 for d in raw_days)/len(raw_days):.1f}% | {100*sum(d <= 7 for d in filtered_days)/len(filtered_days):.1f}% | - |\n")

        f.write("\n")

        # Quality Analysis
        if quality_scores:
            f.write("### 🎯 Quality Score Analysis (Filtered Data)\n\n")
            f.write(f"- **Mean Quality Score:** {np.mean(quality_scores):.3f}\n")
            f.write(f"- **Median Quality Score:** {np.median(quality_scores):.3f}\n")
            high_quality = sum(1 for q in quality_scores if q > 0.8)
            low_quality = sum(1 for q in quality_scores if q < 0.2)
            f.write(f"- **High Quality (>0.8):** {high_quality:,} ({100*high_quality/len(quality_scores):.1f}%)\n")
            f.write(f"- **Low Quality (<0.2):** {low_quality:,} ({100*low_quality/len(quality_scores):.1f}%)\n\n")

        # Top Differences
        if users_with_differences:
            f.write("## 📈 Largest Weight Differences\n\n")
            # Sort by absolute difference
            sorted_diffs = sorted(users_with_differences, key=lambda x: abs(x['difference']), reverse=True)[:10]

            f.write("### Users with Largest Absolute Differences\n\n")
            f.write("| User ID | Raw Weight | Filtered Weight | Difference | Quality Score |\n")
            f.write("|---------|------------|-----------------|------------|---------------|\n")

            for item in sorted_diffs[:10]:
                f.write(f"| {item['user_id'][:8]}... | {item['raw_weight']:.1f} kg | {item['filtered_weight']:.1f} kg | ")
                f.write(f"{item['difference']:+.1f} kg | {item['quality_score']:.3f} |\n")

            f.write("\n")

        # Employer breakdown with differences
        if len(employer_groups) > 1:
            f.write("## 🏢 Breakdown by Employer\n\n")
            f.write("| Employer | Users | Avg Raw | Avg Filtered | Avg Removed | Removal Rate |\n")
            f.write("|----------|-------|---------|--------------|-------------|-------------|\n")

            for employer in sorted(employer_groups.keys()):
                users = employer_groups[employer]
                total_raw_emp = sum(u['raw_count'] for u in users)
                total_filtered_emp = sum(u['filtered_count'] for u in users)
                avg_raw = np.mean([u['raw_count'] for u in users])
                avg_filtered = np.mean([u['filtered_count'] for u in users])
                removal_rate_emp = 100 * (total_raw_emp - total_filtered_emp) / total_raw_emp if total_raw_emp > 0 else 0

                employer_display = employer if employer else "Unknown"
                f.write(f"| {employer_display} | {len(users):,} | {avg_raw:.1f} | {avg_filtered:.1f} | ")
                f.write(f"{avg_raw - avg_filtered:.1f} | {removal_rate_emp:.1f}% |\n")

        # Distribution Analysis
        f.write("\n## 📉 Distribution Analysis\n\n")

        if weight_differences:
            f.write("### Weight Difference Distribution\n\n")
            percentiles = [10, 25, 50, 75, 90]
            f.write("| Percentile | Difference (kg) |\n")
            f.write("|------------|------------------|\n")
            for p in percentiles:
                val = np.percentile(weight_differences, p)
                f.write(f"| {p}th | {val:.2f} |\n")

        # User Impact Summary
        f.write("\n## 👥 User Impact Summary\n\n")

        users_no_change = sum(1 for d in weight_differences if abs(d) < 0.05)  # ~0.1 lbs
        users_minor_change = sum(1 for d in weight_differences if 0.05 <= abs(d) < 0.5)  # 0.1-1 lb
        users_moderate_change = sum(1 for d in weight_differences if 0.5 <= abs(d) < 2.3)  # 1-5 lbs
        users_major_change = sum(1 for d in weight_differences if abs(d) >= 2.3)  # 5+ lbs

        total_compared = len(weight_differences)
        if total_compared > 0:
            f.write("### Change Categories\n\n")
            f.write(f"- **No Change (<0.05 kg):** {users_no_change:,} ({100*users_no_change/total_compared:.1f}%)\n")
            f.write(f"- **Minor Change (0.05-0.5 kg):** {users_minor_change:,} ({100*users_minor_change/total_compared:.1f}%)\n")
            f.write(f"- **Moderate Change (0.5-2.3 kg):** {users_moderate_change:,} ({100*users_moderate_change/total_compared:.1f}%)\n")
            f.write(f"- **Major Change (>2.3 kg):** {users_major_change:,} ({100*users_major_change/total_compared:.1f}%)\n\n")

        # Data Quality Analysis Section (if metrics available)
        if quality_metrics:
            f.write("\n## 🎯 DATA QUALITY IMPROVEMENT ANALYSIS\n\n")

            # Overall Score
            if 'overall_quality_score' in quality_metrics:
                score = quality_metrics['overall_quality_score']
                f.write(f"### Overall Quality Improvement Score: {score:.0f}/100\n\n")
                f.write("**Grade:** ")
                if score >= 90:
                    f.write("Excellent (A)\n")
                elif score >= 80:
                    f.write("Very Good (B)\n")
                elif score >= 70:
                    f.write("Good (C)\n")
                elif score >= 60:
                    f.write("Fair (D)\n")
                else:
                    f.write("Needs Improvement (F)\n")
                f.write("\n")

            # Noise Reduction
            if quality_metrics.get('noise_reduction'):
                f.write("### 📊 Noise Reduction\n\n")
                nr = quality_metrics['noise_reduction']
                f.write(f"- **Variance Reduction:** {nr.get('variance_reduction_pct', 0):.1f}%\n")
                f.write(f"- **Standard Deviation:** {nr.get('raw_std_dev', 0):.2f} → {nr.get('filtered_std_dev', 0):.2f} kg\n")
                f.write(f"- **Users Improved:** {nr.get('users_with_reduced_variance', 0):,} ({nr.get('pct_users_improved', 0):.1f}%)\n\n")

            # Outlier Removal
            if quality_metrics.get('outlier_removal'):
                f.write("### 🎯 Outlier Removal\n\n")
                or_metrics = quality_metrics['outlier_removal']
                f.write(f"- **Measurements Removed:** {or_metrics.get('total_measurements_removed', 0):,} ({or_metrics.get('removal_rate_pct', 0):.1f}%)\n")
                f.write(f"- **Extreme Outliers (>4.5 kg):** {or_metrics.get('extreme_outliers_removed', 0):,}\n")
                f.write(f"- **Moderate Outliers (2.3-4.5 kg):** {or_metrics.get('moderate_outliers_removed', 0):,}\n\n")

            # Consistency Improvements
            if quality_metrics.get('consistency_improvements'):
                f.write("### ✅ Consistency Improvements\n\n")
                ci = quality_metrics['consistency_improvements']
                f.write(f"- **Daily Change Consistency:** {ci.get('consistency_improvement_pct', 0):.1f}% improvement\n")
                f.write(f"- **Implausible Changes (>2.3 kg/day):** {ci.get('implausible_changes_raw', 0)} → {ci.get('implausible_changes_filtered', 0)}\n")
                if ci.get('implausible_reduction_pct', 0) > 0:
                    f.write(f"- **Reduction:** {ci.get('implausible_reduction_pct', 0):.1f}%\n")
                f.write("\n")

            # Trajectory Analysis
            if quality_metrics.get('trajectory_analysis'):
                f.write("### 📈 Weight Trajectory Alignment\n\n")
                ta = quality_metrics['trajectory_analysis']
                f.write(f"- **Trajectories Analyzed:** {ta.get('trajectories_analyzed', 0):,}\n")
                f.write(f"- **Aligned (Same Direction):** {ta.get('trajectories_aligned', 0):,} ({ta.get('alignment_rate_pct', 0):.1f}%)\n")
                f.write(f"- **Divergent:** {ta.get('trajectories_divergent', 0):,}\n\n")

            # Statistical Improvements
            if quality_metrics.get('statistical_improvements'):
                f.write("### 📉 Statistical Distribution\n\n")
                si = quality_metrics['statistical_improvements']
                f.write(f"- **Normality Improved:** {'Yes' if si.get('normality_improved', False) else 'No'}\n")
                f.write(f"- **Skewness:** {si.get('raw_skewness', 0):.3f} → {si.get('filtered_skewness', 0):.3f}\n")
                f.write(f"- **IQR:** {si.get('raw_iqr', 0):.2f} → {si.get('filtered_iqr', 0):.2f} kg\n\n")

        # Sample user details with comparisons
        f.write("## 📝 Sample User Comparisons\n\n")
        sample_users = list(user_data.keys())[:5]
        for user_id in sample_users:
            data = user_data[user_id]
            f.write(f"### User: `{user_id}`\n\n")
            f.write(f"- **Employer:** {data.get('employer_name', 'Unknown')}\n")
            f.write(f"- **Start Date:** {data.get('start_date', 'N/A')}\n")
            f.write(f"- **Measurements:** Raw: {data['raw_count']}, Filtered: {data['filtered_count']} ")
            f.write(f"(Removed: {data['raw_count'] - data['filtered_count']})\n")

            if data.get('closest_raw_weight') is not None and data.get('closest_filtered_weight') is not None:
                diff = data['closest_filtered_weight'] - data['closest_raw_weight']
                f.write(f"- **Weight Comparison:**\n")
                f.write(f"  - Raw: {data['closest_raw_weight']:.1f} kg\n")
                f.write(f"  - Filtered: {data['closest_filtered_weight']:.1f} kg\n")
                f.write(f"  - **Difference: {diff:+.2f} kg**\n")
                if data.get('closest_filtered_quality') is not None:
                    f.write(f"  - Quality Score: {data['closest_filtered_quality']:.3f}\n")
            elif data.get('closest_raw_weight') is not None:
                f.write(f"- **Weight:** Raw only: {data['closest_raw_weight']:.1f} kg\n")
            elif data.get('closest_filtered_weight') is not None:
                f.write(f"- **Weight:** Filtered only: {data['closest_filtered_weight']:.1f} kg\n")

            f.write("\n")

    print(f"\n📊 Analysis report exported to: {output_path}")
    return str(output_path)


def create_visualizations(user_data: Dict, quality_metrics: Dict, employer_filter: str = None) -> List[str]:
    """
    Create comprehensive visualizations comparing raw vs filtered weight data.

    Args:
        user_data: Dictionary containing user analysis data
        quality_metrics: Dictionary with quality improvement metrics
        employer_filter: Optional employer name filter that was applied

    Returns:
        List of paths to created visualization files
    """
    # Create output directories
    output_dir = Path("report_output")
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    created_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # 1. Before/After Timeline Comparison
    print("\n📈 Creating timeline comparison...")
    timeline_file = create_timeline_comparison(user_data, viz_dir, timestamp, employer_filter)
    if timeline_file:
        created_files.append(timeline_file)

    # 2. Outlier Detection Scatter Plot
    print("🎯 Creating outlier detection plot...")
    outlier_file = create_outlier_scatter(user_data, viz_dir, timestamp, employer_filter)
    if outlier_file:
        created_files.append(outlier_file)

    # 3. Variance Reduction Bar Charts
    print("📊 Creating variance reduction charts...")
    variance_file = create_variance_charts(user_data, quality_metrics, viz_dir, timestamp, employer_filter)
    if variance_file:
        created_files.append(variance_file)

    # 4. Daily Change Distribution
    print("📉 Creating daily change distributions...")
    daily_file = create_daily_change_distributions(user_data, viz_dir, timestamp, employer_filter)
    if daily_file:
        created_files.append(daily_file)

    # 5. Trajectory Alignment Scatter
    print("🔄 Creating trajectory alignment scatter...")
    trajectory_file = create_trajectory_alignment(user_data, viz_dir, timestamp, employer_filter)
    if trajectory_file:
        created_files.append(trajectory_file)

    # 6. Quality Score Heatmap
    print("🌡️ Creating quality score heatmap...")
    heatmap_file = create_quality_heatmap(user_data, viz_dir, timestamp, employer_filter)
    if heatmap_file:
        created_files.append(heatmap_file)

    # 7. Statistical Distribution Overlay
    print("📊 Creating statistical distribution overlay...")
    dist_file = create_distribution_overlay(user_data, viz_dir, timestamp, employer_filter)
    if dist_file:
        created_files.append(dist_file)

    # 8. Create summary dashboard (HTML with all Plotly charts)
    if PLOTLY_AVAILABLE:
        print("🎯 Creating interactive dashboard...")
        dashboard_file = create_interactive_dashboard(user_data, quality_metrics, viz_dir, timestamp, employer_filter)
        if dashboard_file:
            created_files.append(dashboard_file)

    print(f"\n✅ Created {len(created_files)} visualization files")
    return created_files


def create_timeline_comparison(user_data: Dict, viz_dir: Path, timestamp: str, employer_filter: str = None) -> Optional[str]:
    """Create before/after timeline comparison showing raw vs filtered data."""
    if not PLOTLY_AVAILABLE:
        print("  ⚠️ Plotly not available, skipping interactive timeline")
        return None

    # Sample up to 10 users with good data
    sample_users = []
    for user_id, data in user_data.items():
        if not data['raw_data'].empty and not data['filtered_data'].empty:
            if len(data['raw_data']) > 10 and len(data['filtered_data']) > 5:
                sample_users.append(user_id)
        if len(sample_users) >= 10:
            break

    if not sample_users:
        print("  ⚠️ No users with sufficient data for timeline comparison")
        return None

    # Create subplot figure
    fig = make_subplots(
        rows=len(sample_users[:5]), cols=2,
        subplot_titles=[f"User {uid[:8]}" for uid in sample_users[:5] for _ in range(2)],
        vertical_spacing=0.05,
        horizontal_spacing=0.1,
        column_titles=["Raw Data", "Filtered Data"]
    )

    for idx, user_id in enumerate(sample_users[:5]):
        data = user_data[user_id]
        row = idx + 1

        # Raw data
        raw_df = data['raw_data'].sort_values('effectiveDateTime')
        fig.add_trace(
            go.Scatter(
                x=raw_df['effectiveDateTime'],
                y=raw_df['weight'],
                mode='markers+lines',
                name=f"Raw {user_id[:8]}",
                marker=dict(size=4, color='lightcoral'),
                line=dict(width=1, color='lightcoral'),
                showlegend=(idx == 0)
            ),
            row=row, col=1
        )

        # Filtered data
        filtered_df = data['filtered_data'].sort_values('effectiveDateTime')
        fig.add_trace(
            go.Scatter(
                x=filtered_df['effectiveDateTime'],
                y=filtered_df['weight'],
                mode='markers+lines',
                name=f"Filtered {user_id[:8]}",
                marker=dict(size=6, color='steelblue'),
                line=dict(width=2, color='steelblue'),
                showlegend=(idx == 0)
            ),
            row=row, col=2
        )

        # Add removed points as red X's on the raw plot
        raw_times = set(raw_df['effectiveDateTime'].values)
        filtered_times = set(filtered_df['effectiveDateTime'].values)
        removed_times = raw_times - filtered_times

        if removed_times:
            removed_df = raw_df[raw_df['effectiveDateTime'].isin(removed_times)]
            fig.add_trace(
                go.Scatter(
                    x=removed_df['effectiveDateTime'],
                    y=removed_df['weight'],
                    mode='markers',
                    name='Removed',
                    marker=dict(size=8, color='red', symbol='x'),
                    showlegend=(idx == 0)
                ),
                row=row, col=1
            )

    # Update layout
    title = f"Timeline Comparison: Raw vs Filtered Weight Data"
    if employer_filter:
        title += f" - {employer_filter}"

    fig.update_layout(
        title=title,
        height=200 * len(sample_users[:5]),
        showlegend=True,
        hovermode='x unified'
    )

    # Save as HTML
    filename = f"timeline_comparison_{employer_filter if employer_filter else 'all'}_{timestamp}.html"
    filepath = viz_dir / filename
    fig.write_html(filepath)

    return str(filepath)


def create_outlier_scatter(user_data: Dict, viz_dir: Path, timestamp: str, employer_filter: str = None) -> Optional[str]:
    """Create scatter plot showing outliers that were removed."""
    if not MATPLOTLIB_AVAILABLE:
        print("  ⚠️ Matplotlib not available, skipping outlier scatter")
        return None

    # Collect all data points
    all_raw_points = []
    all_filtered_points = []
    removed_points = []

    for user_id, data in user_data.items():
        if data['raw_data'].empty or data['filtered_data'].empty:
            continue

        raw_df = data['raw_data']
        filtered_df = data['filtered_data']

        # Create a set of (datetime, weight) tuples for comparison
        filtered_set = set(zip(filtered_df['effectiveDateTime'], filtered_df['weight']))

        for _, row in raw_df.iterrows():
            point = (row['effectiveDateTime'], row['weight'])
            all_raw_points.append(row['weight'])

            if point not in filtered_set:
                removed_points.append(row['weight'])
            else:
                all_filtered_points.append(row['weight'])

    if not all_raw_points:
        print("  ⚠️ No data points for outlier scatter")
        return None

    # Create the scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Histogram of removed values
    if removed_points:
        ax1.hist(removed_points, bins=50, color='red', alpha=0.6, label='Removed Outliers')
        ax1.hist(all_filtered_points, bins=50, color='steelblue', alpha=0.6, label='Retained Values')
        ax1.set_xlabel('Weight (kg)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Removed vs Retained Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Right plot: Box plot comparison
    box_data = []
    labels = []

    if all_raw_points:
        box_data.append(all_raw_points)
        labels.append('Raw Data')

    if all_filtered_points:
        box_data.append(all_filtered_points)
        labels.append('Filtered Data')

    if removed_points:
        box_data.append(removed_points)
        labels.append('Removed Outliers')

    bp = ax2.boxplot(box_data, labels=labels, patch_artist=True)

    # Color the boxes
    colors = ['lightcoral', 'steelblue', 'red']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel('Weight (kg)')
    ax2.set_title('Box Plot: Data Distribution Comparison')
    ax2.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Total Raw: {len(all_raw_points):,}\n"
    stats_text += f"Retained: {len(all_filtered_points):,}\n"
    stats_text += f"Removed: {len(removed_points):,}\n"
    stats_text += f"Removal Rate: {100*len(removed_points)/len(all_raw_points):.1f}%"

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Overall title
    title = f"Outlier Detection Analysis"
    if employer_filter:
        title += f" - {employer_filter}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save as PNG
    filename = f"outlier_detection_{employer_filter if employer_filter else 'all'}_{timestamp}.png"
    filepath = viz_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return str(filepath)


def create_variance_charts(user_data: Dict, quality_metrics: Dict, viz_dir: Path, timestamp: str, employer_filter: str = None) -> Optional[str]:
    """Create bar charts showing variance reduction."""
    if not MATPLOTLIB_AVAILABLE:
        print("  ⚠️ Matplotlib not available, skipping variance charts")
        return None

    # Calculate per-user variance
    user_variances = []

    for user_id, data in user_data.items():
        if data['raw_data'].empty or data['filtered_data'].empty:
            continue

        raw_weights = data['raw_data']['weight'].values
        filtered_weights = data['filtered_data']['weight'].values

        if len(raw_weights) > 2 and len(filtered_weights) > 2:
            raw_var = np.var(raw_weights)
            filtered_var = np.var(filtered_weights)
            reduction = 100 * (raw_var - filtered_var) / raw_var if raw_var > 0 else 0

            user_variances.append({
                'user_id': user_id[:8],
                'raw_variance': raw_var,
                'filtered_variance': filtered_var,
                'reduction_pct': reduction
            })

    if not user_variances:
        print("  ⚠️ No variance data to plot")
        return None

    # Sort by reduction percentage
    user_variances.sort(key=lambda x: x['reduction_pct'], reverse=True)

    # Take top 20 users with highest reduction
    top_users = user_variances[:20]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Variance comparison for top users
    x_pos = np.arange(len(top_users))
    user_ids = [u['user_id'] for u in top_users]
    raw_vars = [u['raw_variance'] for u in top_users]
    filtered_vars = [u['filtered_variance'] for u in top_users]

    width = 0.35
    ax1.bar(x_pos - width/2, raw_vars, width, label='Raw Variance', color='lightcoral', alpha=0.7)
    ax1.bar(x_pos + width/2, filtered_vars, width, label='Filtered Variance', color='steelblue', alpha=0.7)

    ax1.set_xlabel('User ID')
    ax1.set_ylabel('Variance (kg²)')
    ax1.set_title('Variance Comparison: Top 20 Users')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(user_ids, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Right plot: Overall variance metrics
    if quality_metrics and 'noise_reduction' in quality_metrics:
        nr = quality_metrics['noise_reduction']

        categories = ['Standard\nDeviation', 'Variance', 'Users\nImproved']
        raw_values = [
            nr.get('raw_std_dev', 0),
            nr.get('raw_std_dev', 0) ** 2,
            100  # Baseline for percentage
        ]
        filtered_values = [
            nr.get('filtered_std_dev', 0),
            nr.get('filtered_std_dev', 0) ** 2,
            nr.get('pct_users_improved', 0)
        ]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax2.bar(x - width/2, raw_values[:2] + [100], width, label='Raw/Baseline', color='lightcoral', alpha=0.7)
        bars2 = ax2.bar(x + width/2, filtered_values[:2] + [filtered_values[2]], width, label='Filtered/Improved', color='steelblue', alpha=0.7)

        ax2.set_ylabel('Value')
        ax2.set_title('Overall Noise Reduction Metrics')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars1[:2]:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            label = f'{height:.1f}' if height > 10 else f'{height:.1f}%'
            ax2.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # Overall title
    title = f"Variance Reduction Analysis"
    if employer_filter:
        title += f" - {employer_filter}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save as PNG
    filename = f"variance_reduction_{employer_filter if employer_filter else 'all'}_{timestamp}.png"
    filepath = viz_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return str(filepath)


def create_daily_change_distributions(user_data: Dict, viz_dir: Path, timestamp: str, employer_filter: str = None) -> Optional[str]:
    """Create violin/box plots showing daily weight change distributions."""
    if not MATPLOTLIB_AVAILABLE:
        print("  ⚠️ Matplotlib not available, skipping daily change distributions")
        return None

    # Collect daily changes
    raw_changes = []
    filtered_changes = []

    for user_id, data in user_data.items():
        if data['raw_data'].empty or data['filtered_data'].empty:
            continue

        # Calculate daily changes for raw data
        raw_df = data['raw_data'].sort_values('effectiveDateTime')
        if len(raw_df) > 1:
            raw_weights = raw_df['weight'].values
            raw_daily = np.diff(raw_weights)
            raw_changes.extend(raw_daily)

        # Calculate daily changes for filtered data
        filtered_df = data['filtered_data'].sort_values('effectiveDateTime')
        if len(filtered_df) > 1:
            filtered_weights = filtered_df['weight'].values
            filtered_daily = np.diff(filtered_weights)
            filtered_changes.extend(filtered_daily)

    if not raw_changes or not filtered_changes:
        print("  ⚠️ Insufficient data for daily change distributions")
        return None

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Violin plot
    data_for_violin = [raw_changes, filtered_changes]
    parts = axes[0, 0].violinplot(data_for_violin, positions=[1, 2], widths=0.7,
                                   showmeans=True, showmedians=True)

    # Color the violin plots
    colors = ['lightcoral', 'steelblue']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    axes[0, 0].set_xticks([1, 2])
    axes[0, 0].set_xticklabels(['Raw Data', 'Filtered Data'])
    axes[0, 0].set_ylabel('Daily Weight Change (kg)')
    axes[0, 0].set_title('Distribution of Daily Weight Changes')
    axes[0, 0].grid(True, alpha=0.3)

    # Add horizontal line at ±5 kg (physiological limit)
    axes[0, 0].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='±5 kg threshold')
    axes[0, 0].axhline(y=-5, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].legend()

    # Top right: Histogram comparison
    bins = np.linspace(-10, 10, 41)
    axes[0, 1].hist(raw_changes, bins=bins, alpha=0.5, label='Raw Data', color='lightcoral', density=True)
    axes[0, 1].hist(filtered_changes, bins=bins, alpha=0.5, label='Filtered Data', color='steelblue', density=True)
    axes[0, 1].set_xlabel('Daily Weight Change (kg)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Histogram of Daily Changes')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=5, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].axvline(x=-5, color='red', linestyle='--', alpha=0.5)

    # Bottom left: Q-Q plot to check normality
    from scipy import stats

    # Q-Q plot for raw changes
    stats.probplot(raw_changes, dist="norm", plot=axes[1, 0])
    axes[1, 0].get_lines()[0].set_markerfacecolor('lightcoral')
    axes[1, 0].get_lines()[0].set_markeredgecolor('lightcoral')
    axes[1, 0].get_lines()[0].set_markersize(3)
    axes[1, 0].set_title('Q-Q Plot: Raw Daily Changes')
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom right: Q-Q plot for filtered changes
    stats.probplot(filtered_changes, dist="norm", plot=axes[1, 1])
    axes[1, 1].get_lines()[0].set_markerfacecolor('steelblue')
    axes[1, 1].get_lines()[0].set_markeredgecolor('steelblue')
    axes[1, 1].get_lines()[0].set_markersize(3)
    axes[1, 1].set_title('Q-Q Plot: Filtered Daily Changes')
    axes[1, 1].grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Raw Data:\n"
    stats_text += f"  Mean: {np.mean(raw_changes):.2f} kg\n"
    stats_text += f"  Std: {np.std(raw_changes):.2f} kg\n"
    stats_text += f"  >2.3 kg: {sum(abs(c) > 2.3 for c in raw_changes)}\n\n"
    stats_text += f"Filtered Data:\n"
    stats_text += f"  Mean: {np.mean(filtered_changes):.2f} kg\n"
    stats_text += f"  Std: {np.std(filtered_changes):.2f} kg\n"
    stats_text += f"  >2.3 kg: {sum(abs(c) > 2.3 for c in filtered_changes)}"

    fig.text(0.02, 0.5, stats_text, transform=fig.transFigure,
             verticalalignment='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Overall title
    title = f"Daily Weight Change Analysis"
    if employer_filter:
        title += f" - {employer_filter}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save as PNG
    filename = f"daily_changes_{employer_filter if employer_filter else 'all'}_{timestamp}.png"
    filepath = viz_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return str(filepath)


def create_trajectory_alignment(user_data: Dict, viz_dir: Path, timestamp: str, employer_filter: str = None) -> Optional[str]:
    """Create scatter plot showing trajectory alignment between raw and filtered data."""
    if not PLOTLY_AVAILABLE:
        print("  ⚠️ Plotly not available, skipping trajectory alignment")
        return None

    # Collect trajectory data
    trajectory_data = []

    for user_id, data in user_data.items():
        start_raw = data.get('closest_raw_weight')
        start_filtered = data.get('closest_filtered_weight')
        end_raw = data.get('latest_raw_weight')
        end_filtered = data.get('latest_filtered_weight')

        if all(x is not None for x in [start_raw, start_filtered, end_raw, end_filtered]):
            change_raw = end_raw - start_raw
            change_filtered = end_filtered - start_filtered

            trajectory_data.append({
                'user_id': user_id[:12],
                'raw_change': change_raw,
                'filtered_change': change_filtered,
                'aligned': (change_raw * change_filtered) > 0,
                'difference': abs(change_raw - change_filtered)
            })

    if not trajectory_data:
        print("  ⚠️ No trajectory data available")
        return None

    # Create DataFrame
    df = pd.DataFrame(trajectory_data)

    # Create scatter plot
    fig = go.Figure()

    # Add diagonal reference line
    min_val = min(df['raw_change'].min(), df['filtered_change'].min())
    max_val = max(df['raw_change'].max(), df['filtered_change'].max())

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Alignment',
        line=dict(color='gray', dash='dash'),
        showlegend=True
    ))

    # Add scatter points colored by alignment
    aligned_df = df[df['aligned']]
    misaligned_df = df[~df['aligned']]

    if not aligned_df.empty:
        fig.add_trace(go.Scatter(
            x=aligned_df['raw_change'],
            y=aligned_df['filtered_change'],
            mode='markers',
            name='Aligned Trajectories',
            marker=dict(
                size=8,
                color='green',
                opacity=0.6,
                line=dict(width=1, color='darkgreen')
            ),
            text=aligned_df['user_id'],
            hovertemplate='User: %{text}<br>Raw Change: %{x:.1f} kg<br>Filtered Change: %{y:.1f} kg'
        ))

    if not misaligned_df.empty:
        fig.add_trace(go.Scatter(
            x=misaligned_df['raw_change'],
            y=misaligned_df['filtered_change'],
            mode='markers',
            name='Misaligned Trajectories',
            marker=dict(
                size=10,
                color='red',
                symbol='x',
                opacity=0.7,
                line=dict(width=2, color='darkred')
            ),
            text=misaligned_df['user_id'],
            hovertemplate='User: %{text}<br>Raw Change: %{x:.1f} kg<br>Filtered Change: %{y:.1f} kg'
        ))

    # Add quadrant annotations
    fig.add_annotation(x=10, y=10, text="Both Gain", showarrow=False, font=dict(size=12, color="gray"))
    fig.add_annotation(x=-10, y=-10, text="Both Loss", showarrow=False, font=dict(size=12, color="gray"))
    fig.add_annotation(x=10, y=-10, text="Raw Gain\nFiltered Loss", showarrow=False, font=dict(size=12, color="red"))
    fig.add_annotation(x=-10, y=10, text="Raw Loss\nFiltered Gain", showarrow=False, font=dict(size=12, color="red"))

    # Update layout
    title = f"Weight Trajectory Alignment: Raw vs Filtered"
    if employer_filter:
        title += f" - {employer_filter}"

    fig.update_layout(
        title=title,
        xaxis_title="Raw Weight Change (kg)",
        yaxis_title="Filtered Weight Change (kg)",
        hovermode='closest',
        width=800,
        height=800,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='lightgray'),
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='lightgray')
    )

    # Add statistics box
    aligned_count = len(aligned_df)
    misaligned_count = len(misaligned_df)
    total_count = aligned_count + misaligned_count
    alignment_rate = 100 * aligned_count / total_count if total_count > 0 else 0

    stats_text = f"Total Users: {total_count}<br>"
    stats_text += f"Aligned: {aligned_count} ({alignment_rate:.1f}%)<br>"
    stats_text += f"Misaligned: {misaligned_count} ({100-alignment_rate:.1f}%)"

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=stats_text,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12),
        align="left"
    )

    # Save as HTML
    filename = f"trajectory_alignment_{employer_filter if employer_filter else 'all'}_{timestamp}.html"
    filepath = viz_dir / filename
    fig.write_html(filepath)

    return str(filepath)


def create_quality_heatmap(user_data: Dict, viz_dir: Path, timestamp: str, employer_filter: str = None) -> Optional[str]:
    """Create heatmap showing quality scores across users and time."""
    if not MATPLOTLIB_AVAILABLE:
        print("  ⚠️ Matplotlib not available, skipping quality heatmap")
        return None

    # Sample users with quality scores
    sample_users = []
    quality_matrix = []
    user_labels = []

    for user_id, data in user_data.items():
        if data['filtered_data'].empty:
            continue

        filtered_df = data['filtered_data'].sort_values('effectiveDateTime')
        if 'quality_score' in filtered_df.columns and len(filtered_df) > 5:
            # Get quality scores (sample up to 50 measurements evenly)
            total_measurements = len(filtered_df)
            sample_size = min(50, total_measurements)
            indices = np.linspace(0, total_measurements-1, sample_size, dtype=int)

            sampled_scores = filtered_df.iloc[indices]['quality_score'].values
            quality_matrix.append(sampled_scores)
            user_labels.append(user_id[:8])

            if len(quality_matrix) >= 30:  # Limit to 30 users for readability
                break

    if not quality_matrix:
        print("  ⚠️ No quality score data available")
        return None

    # Pad arrays to same length
    max_length = max(len(row) for row in quality_matrix)
    padded_matrix = []
    for row in quality_matrix:
        padded = np.pad(row, (0, max_length - len(row)), constant_values=np.nan)
        padded_matrix.append(padded)

    quality_array = np.array(padded_matrix)

    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [3, 1]})

    # Main heatmap
    im = ax1.imshow(quality_array, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    ax1.set_yticks(np.arange(len(user_labels)))
    ax1.set_yticklabels(user_labels, fontsize=8)
    ax1.set_xlabel('Measurement Index (Time →)')
    ax1.set_ylabel('User ID')
    ax1.set_title('Quality Scores Over Time')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Quality Score', rotation=270, labelpad=15)

    # Right plot: Average quality score per user
    avg_scores = np.nanmean(quality_array, axis=1)
    y_pos = np.arange(len(user_labels))

    # Color bars based on score
    colors = ['red' if s < 0.5 else 'yellow' if s < 0.8 else 'green' for s in avg_scores]
    bars = ax2.barh(y_pos, avg_scores, color=colors, alpha=0.7)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(user_labels, fontsize=8)
    ax2.set_xlabel('Average Quality Score')
    ax2.set_title('User Average Scores')
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, avg_scores)):
        ax2.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}', va='center', fontsize=7)

    # Overall title
    title = f"Quality Score Analysis"
    if employer_filter:
        title += f" - {employer_filter}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Add statistics
    overall_avg = np.nanmean(quality_array)
    high_quality = np.sum(quality_array > 0.8) / np.sum(~np.isnan(quality_array)) * 100
    low_quality = np.sum(quality_array < 0.5) / np.sum(~np.isnan(quality_array)) * 100

    stats_text = f"Overall Average: {overall_avg:.3f}\n"
    stats_text += f"High Quality (>0.8): {high_quality:.1f}%\n"
    stats_text += f"Low Quality (<0.5): {low_quality:.1f}%"

    fig.text(0.02, 0.02, stats_text, transform=fig.transFigure,
             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save as PNG
    filename = f"quality_heatmap_{employer_filter if employer_filter else 'all'}_{timestamp}.png"
    filepath = viz_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return str(filepath)


def create_distribution_overlay(user_data: Dict, viz_dir: Path, timestamp: str, employer_filter: str = None) -> Optional[str]:
    """Create KDE plots overlaying raw and filtered distributions."""
    if not MATPLOTLIB_AVAILABLE:
        print("  ⚠️ Matplotlib not available, skipping distribution overlay")
        return None

    # Collect all weights
    all_raw_weights = []
    all_filtered_weights = []

    for user_id, data in user_data.items():
        if not data['raw_data'].empty:
            all_raw_weights.extend(data['raw_data']['weight'].values)
        if not data['filtered_data'].empty:
            all_filtered_weights.extend(data['filtered_data']['weight'].values)

    if not all_raw_weights or not all_filtered_weights:
        print("  ⚠️ Insufficient data for distribution overlay")
        return None

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: KDE overlay
    from scipy.stats import gaussian_kde

    kde_raw = gaussian_kde(all_raw_weights)
    kde_filtered = gaussian_kde(all_filtered_weights)

    x_range = np.linspace(min(all_raw_weights), max(all_raw_weights), 1000)

    axes[0, 0].plot(x_range, kde_raw(x_range), color='lightcoral', label='Raw Data', linewidth=2, alpha=0.7)
    axes[0, 0].plot(x_range, kde_filtered(x_range), color='steelblue', label='Filtered Data', linewidth=2)
    axes[0, 0].fill_between(x_range, kde_raw(x_range), alpha=0.3, color='lightcoral')
    axes[0, 0].fill_between(x_range, kde_filtered(x_range), alpha=0.3, color='steelblue')
    axes[0, 0].set_xlabel('Weight (kg)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Kernel Density Estimation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Cumulative distribution
    axes[0, 1].hist(all_raw_weights, bins=100, cumulative=True, density=True,
                    alpha=0.5, label='Raw Data', color='lightcoral', histtype='step', linewidth=2)
    axes[0, 1].hist(all_filtered_weights, bins=100, cumulative=True, density=True,
                    alpha=0.5, label='Filtered Data', color='steelblue', histtype='step', linewidth=2)
    axes[0, 1].set_xlabel('Weight (kg)')
    axes[0, 1].set_ylabel('Cumulative Probability')
    axes[0, 1].set_title('Cumulative Distribution Function')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom left: Statistical comparison
    categories = ['Mean', 'Median', 'Std Dev', 'IQR', 'Skewness', 'Kurtosis']

    raw_stats = [
        np.mean(all_raw_weights),
        np.median(all_raw_weights),
        np.std(all_raw_weights),
        np.percentile(all_raw_weights, 75) - np.percentile(all_raw_weights, 25),
        stats.skew(all_raw_weights),
        stats.kurtosis(all_raw_weights)
    ]

    filtered_stats = [
        np.mean(all_filtered_weights),
        np.median(all_filtered_weights),
        np.std(all_filtered_weights),
        np.percentile(all_filtered_weights, 75) - np.percentile(all_filtered_weights, 25),
        stats.skew(all_filtered_weights),
        stats.kurtosis(all_filtered_weights)
    ]

    x_pos = np.arange(len(categories))
    width = 0.35

    bars1 = axes[1, 0].bar(x_pos - width/2, raw_stats, width, label='Raw Data', color='lightcoral', alpha=0.7)
    bars2 = axes[1, 0].bar(x_pos + width/2, filtered_stats, width, label='Filtered Data', color='steelblue', alpha=0.7)

    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Statistical Metrics Comparison')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(categories, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].annotate(f'{height:.2f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=8, rotation=0)

    # Bottom right: Percentile comparison
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    raw_percentiles = [np.percentile(all_raw_weights, p) for p in percentiles]
    filtered_percentiles = [np.percentile(all_filtered_weights, p) for p in percentiles]

    axes[1, 1].plot(percentiles, raw_percentiles, 'o-', color='lightcoral', label='Raw Data', linewidth=2, markersize=8)
    axes[1, 1].plot(percentiles, filtered_percentiles, 's-', color='steelblue', label='Filtered Data', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Percentile')
    axes[1, 1].set_ylabel('Weight (kg)')
    axes[1, 1].set_title('Percentile Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Overall title
    title = f"Statistical Distribution Analysis"
    if employer_filter:
        title += f" - {employer_filter}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Add summary text
    improvement_text = "Improvements:\n"
    improvement_text += f"• Std Dev: {(raw_stats[2] - filtered_stats[2])/raw_stats[2]*100:.1f}% reduction\n"
    improvement_text += f"• IQR: {(raw_stats[3] - filtered_stats[3])/raw_stats[3]*100:.1f}% reduction\n"
    improvement_text += f"• Skewness: {abs(filtered_stats[4]) < abs(raw_stats[4]) and 'Improved' or 'Worsened'}\n"
    improvement_text += f"• Sample Size: {len(all_raw_weights):,} → {len(all_filtered_weights):,}"

    fig.text(0.02, 0.02, improvement_text, transform=fig.transFigure,
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()

    # Save as PNG
    filename = f"distribution_overlay_{employer_filter if employer_filter else 'all'}_{timestamp}.png"
    filepath = viz_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return str(filepath)


def create_interactive_dashboard(user_data: Dict, quality_metrics: Dict, viz_dir: Path, timestamp: str, employer_filter: str = None) -> Optional[str]:
    """Create an interactive dashboard with all Plotly visualizations."""
    if not PLOTLY_AVAILABLE:
        print("  ⚠️ Plotly not available, skipping interactive dashboard")
        return None

    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Weight Trajectory Comparison",
            "Quality Score Distribution",
            "Outlier Detection",
            "Variance Reduction",
            "Daily Change Distribution",
            "Overall Metrics"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "box"}],
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "violin"}, {"type": "indicator"}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.15
    )

    # 1. Weight Trajectory Comparison (sample 5 users)
    sample_users = []
    for user_id, data in user_data.items():
        if not data['raw_data'].empty and not data['filtered_data'].empty:
            if len(data['raw_data']) > 10:
                sample_users.append(user_id)
        if len(sample_users) >= 5:
            break

    for idx, user_id in enumerate(sample_users[:3]):
        data = user_data[user_id]
        raw_df = data['raw_data'].sort_values('effectiveDateTime')
        filtered_df = data['filtered_data'].sort_values('effectiveDateTime')

        fig.add_trace(
            go.Scatter(
                x=raw_df['effectiveDateTime'],
                y=raw_df['weight'],
                mode='lines',
                name=f"Raw {user_id[:8]}",
                line=dict(width=1, color=f'rgba(255, 0, 0, {0.3 + idx*0.2})'),
                legendgroup="raw",
                showlegend=(idx == 0)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=filtered_df['effectiveDateTime'],
                y=filtered_df['weight'],
                mode='lines',
                name=f"Filtered {user_id[:8]}",
                line=dict(width=2, color=f'rgba(0, 0, 255, {0.3 + idx*0.2})'),
                legendgroup="filtered",
                showlegend=(idx == 0)
            ),
            row=1, col=1
        )

    # 2. Quality Score Distribution
    quality_scores = []
    for user_id, data in user_data.items():
        if not data['filtered_data'].empty and 'quality_score' in data['filtered_data'].columns:
            quality_scores.extend(data['filtered_data']['quality_score'].dropna().values)

    if quality_scores:
        fig.add_trace(
            go.Box(y=quality_scores, name="Quality Scores", marker_color='green'),
            row=1, col=2
        )

    # 3. Outlier Detection Scatter
    all_raw_weights = []
    all_filtered_weights = []
    for user_id, data in user_data.items():
        if not data['raw_data'].empty:
            all_raw_weights.extend(data['raw_data']['weight'].values)
        if not data['filtered_data'].empty:
            all_filtered_weights.extend(data['filtered_data']['weight'].values)

    removed_count = len(all_raw_weights) - len(all_filtered_weights)

    fig.add_trace(
        go.Scatter(
            x=np.random.randn(len(all_raw_weights)),
            y=all_raw_weights,
            mode='markers',
            name='Raw Data',
            marker=dict(size=3, color='lightcoral', opacity=0.5)
        ),
        row=2, col=1
    )

    # 4. Variance Reduction Bar Chart
    if quality_metrics and 'noise_reduction' in quality_metrics:
        nr = quality_metrics['noise_reduction']

        fig.add_trace(
            go.Bar(
                x=['Raw Std Dev', 'Filtered Std Dev'],
                y=[nr.get('raw_std_dev', 0), nr.get('filtered_std_dev', 0)],
                marker_color=['lightcoral', 'steelblue'],
                text=[f"{nr.get('raw_std_dev', 0):.2f}", f"{nr.get('filtered_std_dev', 0):.2f}"],
                textposition='auto'
            ),
            row=2, col=2
        )

    # 5. Daily Change Distribution
    raw_changes = []
    filtered_changes = []
    for user_id, data in user_data.items():
        if not data['raw_data'].empty and len(data['raw_data']) > 1:
            raw_df = data['raw_data'].sort_values('effectiveDateTime')
            raw_changes.extend(np.diff(raw_df['weight'].values))

        if not data['filtered_data'].empty and len(data['filtered_data']) > 1:
            filtered_df = data['filtered_data'].sort_values('effectiveDateTime')
            filtered_changes.extend(np.diff(filtered_df['weight'].values))

    if raw_changes and filtered_changes:
        fig.add_trace(
            go.Violin(y=raw_changes, name='Raw Changes', side='negative', marker_color='lightcoral'),
            row=3, col=1
        )
        fig.add_trace(
            go.Violin(y=filtered_changes, name='Filtered Changes', side='positive', marker_color='steelblue'),
            row=3, col=1
        )

    # 6. Overall Metrics Indicator
    if quality_metrics and 'overall_quality_score' in quality_metrics:
        score = quality_metrics['overall_quality_score']

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Quality Score"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 80], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75,
                                   'value': 90}}
            ),
            row=3, col=2
        )

    # Update layout
    title = f"Interactive Dashboard: Raw vs Filtered Analysis"
    if employer_filter:
        title += f" - {employer_filter}"

    fig.update_layout(
        title=title,
        height=1200,
        showlegend=True,
        hovermode='x unified'
    )

    # Update axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Weight (kg)", row=1, col=1)
    fig.update_yaxes(title_text="Quality Score", row=1, col=2)
    fig.update_xaxes(title_text="Random X", row=2, col=1)
    fig.update_yaxes(title_text="Weight (kg)", row=2, col=1)
    fig.update_yaxes(title_text="Standard Deviation", row=2, col=2)
    fig.update_yaxes(title_text="Daily Change (kg)", row=3, col=1)

    # Save as HTML
    filename = f"interactive_dashboard_{employer_filter if employer_filter else 'all'}_{timestamp}.html"
    filepath = viz_dir / filename
    fig.write_html(filepath)

    return str(filepath)


def main():
    """Main entry point."""
    args = parse_arguments()

    # Define file paths
    raw_file = "../data/2025-09-05_nocon.csv"
    filtered_file = "../data/2025-09-05_nocon_filtered.csv"
    employer_file = "../data/2025-09-17-user-employers.csv"
    partners_file = "../data/partners.csv"

    # Check files exist
    for file_path in [raw_file, filtered_file, employer_file, partners_file]:
        if not Path(file_path).exists():
            print(f"Error: File not found at {file_path}")
            sys.exit(1)

    print("="*60)
    print("LOADING AND FILTERING WEIGHT DATA")
    print("="*60)

    try:
        # Step 1: Load filtered users
        filtered_df = load_filtered_users(filtered_file)

        # Step 2: Load employer data with names
        employer_df = load_employer_data(employer_file, partners_file)

        # Step 3: Filter users by employer if specified
        selected_users, employer_filtered = filter_users_by_employer(
            filtered_df, employer_df, args.employer
        )

        if not selected_users:
            print("\nNo users found matching criteria. Exiting.")
            sys.exit(0)

        # Step 4: Filter users by program duration (90+ days)
        reference_date = '2025-09-05'
        selected_users, employer_filtered = filter_users_by_program_duration(
            employer_filtered, selected_users, min_days=90, reference_date=reference_date
        )

        if not selected_users:
            print("\nNo users with 90+ days in program. Exiting.")
            sys.exit(0)

        # Step 5: Load raw data only for selected users
        raw_df = load_raw_data_for_users(raw_file, selected_users)

        # Step 6: Filter dataframes to only selected users
        filtered_df = filtered_df[filtered_df['user_id'].isin(selected_users)]

        # Step 7: Create user data structure
        user_data = create_user_data_structure(
            selected_users, raw_df, filtered_df, employer_filtered
        )

        # Step 8: Analyze weights near start dates
        user_data = find_closest_weight_to_start(user_data)

        # Step 9: Analyze weights at 90 days
        user_data = find_weight_at_90_days(user_data)

        # Step 10: Analyze latest weight values
        user_data = find_latest_weight_values(user_data, reference_date=reference_date)

        # Step 11: Analyze data quality improvements
        quality_metrics = analyze_data_quality_improvements(user_data)

        # Step 12: Export weight data to CSV (now includes quality metrics)
        csv_path = export_weights_to_csv(user_data, args.employer, quality_metrics)

        # Step 13: Export analysis to markdown (now includes quality metrics)
        report_path = export_analysis_to_markdown(user_data, args.employer, quality_metrics)

        # Step 14: Generate visualizations if requested
        if args.visualize:
            visualization_files = create_visualizations(user_data, quality_metrics, args.employer)
            print(f"\nVisualization files created:")
            for viz_file in visualization_files:
                print(f"  - {viz_file}")

        print("\n" + "="*60)
        print(f"Ready to analyze {len(user_data):,} users (90+ days in program)")

        if args.employer:
            print(f"Employer filter: {args.employer}")

        # Show sample of users with analysis
        sample_users = list(user_data.keys())[:5]
        print(f"\nSample users with start weight analysis:")

        for user_id in sample_users[:3]:
            data = user_data[user_id]
            print(f"\n  User {user_id}:")
            print(f"    Employer: {data['employer_name']}")
            print(f"    Start date: {data['start_date']}")
            print(f"    Days in program: {data.get('days_in_program', 'N/A')}")
            print(f"    Measurements: {data['raw_count']} raw, {data['filtered_count']} filtered")

            if data.get('closest_raw_weight') is not None:
                print(f"    Closest raw weight to start: {data['closest_raw_weight']:.1f} kg")
                print(f"      Date: {data['closest_raw_date']}")
                print(f"      Days from start: {data['days_from_start_raw']:.1f}")

            if data.get('closest_filtered_weight') is not None:
                print(f"    Closest filtered weight to start: {data['closest_filtered_weight']:.1f} kg")
                print(f"      Date: {data['closest_filtered_date']}")
                print(f"      Days from start: {data['days_from_start_filtered']:.1f}")
                if data.get('closest_filtered_quality') is not None:
                    print(f"      Quality score: {data['closest_filtered_quality']:.3f}")

            if data.get('weight_90d_raw') is not None:
                print(f"    90-day raw weight: {data['weight_90d_raw']:.1f} kg")
                print(f"      Date: {data['date_90d_raw']}")
                print(f"      Days from 90d mark: {data['days_from_90d_raw']:.1f}")

            if data.get('weight_90d_filtered') is not None:
                print(f"    90-day filtered weight: {data['weight_90d_filtered']:.1f} kg")
                print(f"      Date: {data['date_90d_filtered']}")
                print(f"      Days from 90d mark: {data['days_from_90d_filtered']:.1f}")

            if data.get('latest_raw_weight') is not None:
                print(f"    Latest raw weight: {data['latest_raw_weight']:.1f} kg")
                print(f"      Date: {data['latest_raw_date']}")
                print(f"      Days since: {data['days_since_latest_raw']:.1f}")

            if data.get('latest_filtered_weight') is not None:
                print(f"    Latest filtered weight: {data['latest_filtered_weight']:.1f} kg")
                print(f"      Date: {data['latest_filtered_date']}")
                print(f"      Days since: {data['days_since_latest_filtered']:.1f}")

        return user_data, quality_metrics

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    result = main()
    if result:
        user_data, quality_metrics = result

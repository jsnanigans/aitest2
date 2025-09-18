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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Load and filter weight data by employer')
    parser.add_argument(
        '--employer',
        type=str,
        help='Employer name to filter by (e.g., AMAZON_EMPLOYER). If not provided, loads all users.'
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


def export_weights_to_csv(user_data: Dict, employer_filter: str = None) -> str:
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
            if abs_diff > 5:
                sig_diffs.append(f'Start:{abs_diff:.0f}lb')
                alert_score += 2
            elif abs_diff > 1:
                alert_score += 1

        if row['weight_90d_diff'] != '':
            abs_diff = abs(float(row['weight_90d_diff']))
            if abs_diff > 5:
                sig_diffs.append(f'90d:{abs_diff:.0f}lb')
                alert_score += 2
            elif abs_diff > 1:
                alert_score += 1

        if row['latest_weight_diff'] != '':
            abs_diff = abs(float(row['latest_weight_diff']))
            if abs_diff > 5:
                sig_diffs.append(f'Latest:{abs_diff:.0f}lb')
                alert_score += 2
            elif abs_diff > 1:
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
            elif abs(diff) > 5:
                row['FLAG_trajectory_divergence'] = f'DIFF:{abs(diff):.0f}lb'
                alert_reasons.append(f'Trajectory diff {abs(diff):.0f}lb')
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

        print("\n  Summary of weight differences (filtered - raw):")
        for col in ['start_weight_diff', 'weight_90d_diff', 'latest_weight_diff']:
            if col in df.columns:
                valid_diffs = df[col].replace('', np.nan).dropna()
                if not valid_diffs.empty:
                    col_name = col.replace('_diff', '').replace('_', ' ').title()
                    print(f"    {col_name}:")
                    print(f"      Mean: {valid_diffs.mean():.2f} lbs")
                    print(f"      Median: {valid_diffs.median():.2f} lbs")
                    print(f"      Std Dev: {valid_diffs.std():.2f} lbs")

        # Summary of trajectory differences
        if 'change_total_diff' in df.columns:
            valid_traj = df['change_total_diff'].replace('', np.nan).dropna()
            if not valid_traj.empty:
                print(f"\n  Trajectory Difference (filtered vs raw total change):")
                print(f"    Mean: {valid_traj.mean():.2f} lbs")
                print(f"    Median: {valid_traj.median():.2f} lbs")
                print(f"    Users with opposite trends: {(df['FLAG_trajectory_divergence'] == 'OPPOSITE').sum():,}")

    return str(output_path)


def export_analysis_to_markdown(user_data: Dict, employer_filter: str = None) -> str:
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
            f.write(f"- **Average Difference:** {np.mean(weight_differences):.2f} lbs\n")
            f.write(f"- **Median Difference:** {np.median(weight_differences):.2f} lbs\n")
            f.write(f"- **Average Absolute Difference:** {np.mean(abs_diffs):.2f} lbs\n")
            f.write(f"- **Max Absolute Difference:** {np.max(abs_diffs):.2f} lbs\n")

            # Count significant differences
            sig_diffs = sum(1 for d in abs_diffs if d > 1.0)
            large_diffs = sum(1 for d in abs_diffs if d > 5.0)
            f.write(f"- **Differences > 1 lb:** {sig_diffs:,} ({100*sig_diffs/len(weight_differences):.1f}%)\n")
            f.write(f"- **Differences > 5 lbs:** {large_diffs:,} ({100*large_diffs/len(weight_differences):.1f}%)\n\n")

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
                f.write(f"- **Mean Latest Raw Weight:** {np.mean(latest_raw_weights):.1f} lbs\n")
                f.write(f"- **Mean Latest Filtered Weight:** {np.mean(latest_filtered_weights):.1f} lbs\n")
                f.write(f"- **Average Difference:** {np.mean(latest_differences):.2f} lbs\n")
                f.write(f"- **Median Difference:** {np.median(latest_differences):.2f} lbs\n")
                f.write(f"- **Average Absolute Difference:** {np.mean(abs_latest_diffs):.2f} lbs\n")

                sig_latest = sum(1 for d in abs_latest_diffs if d > 1.0)
                large_latest = sum(1 for d in abs_latest_diffs if d > 5.0)
                f.write(f"- **Differences > 1 lb:** {sig_latest:,} ({100*sig_latest/len(latest_differences):.1f}%)\n")
                f.write(f"- **Differences > 5 lbs:** {large_latest:,} ({100*large_latest/len(latest_differences):.1f}%)\n\n")

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

            f.write(f"| **Mean Weight** | {raw_mean:.1f} lbs | {filtered_mean:.1f} lbs | {filtered_mean - raw_mean:+.1f} lbs |\n")
            f.write(f"| **Median Weight** | {raw_median:.1f} lbs | {filtered_median:.1f} lbs | {filtered_median - raw_median:+.1f} lbs |\n")
            f.write(f"| **Std Dev** | {raw_std:.1f} lbs | {filtered_std:.1f} lbs | {filtered_std - raw_std:+.1f} lbs |\n")
            f.write(f"| **Min Weight** | {np.min(raw_weights):.1f} lbs | {np.min(filtered_weights):.1f} lbs | {np.min(filtered_weights) - np.min(raw_weights):+.1f} lbs |\n")
            f.write(f"| **Max Weight** | {np.max(raw_weights):.1f} lbs | {np.max(filtered_weights):.1f} lbs | {np.max(filtered_weights) - np.max(raw_weights):+.1f} lbs |\n")

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
                f.write(f"| {item['user_id'][:8]}... | {item['raw_weight']:.1f} lbs | {item['filtered_weight']:.1f} lbs | ")
                f.write(f"{item['difference']:+.1f} lbs | {item['quality_score']:.3f} |\n")

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
            f.write("| Percentile | Difference (lbs) |\n")
            f.write("|------------|------------------|\n")
            for p in percentiles:
                val = np.percentile(weight_differences, p)
                f.write(f"| {p}th | {val:.2f} |\n")

        # User Impact Summary
        f.write("\n## 👥 User Impact Summary\n\n")

        users_no_change = sum(1 for d in weight_differences if abs(d) < 0.1)
        users_minor_change = sum(1 for d in weight_differences if 0.1 <= abs(d) < 1.0)
        users_moderate_change = sum(1 for d in weight_differences if 1.0 <= abs(d) < 5.0)
        users_major_change = sum(1 for d in weight_differences if abs(d) >= 5.0)

        total_compared = len(weight_differences)
        if total_compared > 0:
            f.write("### Change Categories\n\n")
            f.write(f"- **No Change (<0.1 lbs):** {users_no_change:,} ({100*users_no_change/total_compared:.1f}%)\n")
            f.write(f"- **Minor Change (0.1-1 lb):** {users_minor_change:,} ({100*users_minor_change/total_compared:.1f}%)\n")
            f.write(f"- **Moderate Change (1-5 lbs):** {users_moderate_change:,} ({100*users_moderate_change/total_compared:.1f}%)\n")
            f.write(f"- **Major Change (>5 lbs):** {users_major_change:,} ({100*users_major_change/total_compared:.1f}%)\n\n")

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
                f.write(f"  - Raw: {data['closest_raw_weight']:.1f} lbs\n")
                f.write(f"  - Filtered: {data['closest_filtered_weight']:.1f} lbs\n")
                f.write(f"  - **Difference: {diff:+.2f} lbs**\n")
                if data.get('closest_filtered_quality') is not None:
                    f.write(f"  - Quality Score: {data['closest_filtered_quality']:.3f}\n")
            elif data.get('closest_raw_weight') is not None:
                f.write(f"- **Weight:** Raw only: {data['closest_raw_weight']:.1f} lbs\n")
            elif data.get('closest_filtered_weight') is not None:
                f.write(f"- **Weight:** Filtered only: {data['closest_filtered_weight']:.1f} lbs\n")

            f.write("\n")

    print(f"\n📊 Analysis report exported to: {output_path}")
    return str(output_path)


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

        # Step 11: Export weight data to CSV
        csv_path = export_weights_to_csv(user_data, args.employer)

        # Step 12: Export analysis to markdown
        report_path = export_analysis_to_markdown(user_data, args.employer)

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
                print(f"    Closest raw weight to start: {data['closest_raw_weight']:.1f} lbs")
                print(f"      Date: {data['closest_raw_date']}")
                print(f"      Days from start: {data['days_from_start_raw']:.1f}")

            if data.get('closest_filtered_weight') is not None:
                print(f"    Closest filtered weight to start: {data['closest_filtered_weight']:.1f} lbs")
                print(f"      Date: {data['closest_filtered_date']}")
                print(f"      Days from start: {data['days_from_start_filtered']:.1f}")
                if data.get('closest_filtered_quality') is not None:
                    print(f"      Quality score: {data['closest_filtered_quality']:.3f}")

            if data.get('weight_90d_raw') is not None:
                print(f"    90-day raw weight: {data['weight_90d_raw']:.1f} lbs")
                print(f"      Date: {data['date_90d_raw']}")
                print(f"      Days from 90d mark: {data['days_from_90d_raw']:.1f}")

            if data.get('weight_90d_filtered') is not None:
                print(f"    90-day filtered weight: {data['weight_90d_filtered']:.1f} lbs")
                print(f"      Date: {data['date_90d_filtered']}")
                print(f"      Days from 90d mark: {data['days_from_90d_filtered']:.1f}")

            if data.get('latest_raw_weight') is not None:
                print(f"    Latest raw weight: {data['latest_raw_weight']:.1f} lbs")
                print(f"      Date: {data['latest_raw_date']}")
                print(f"      Days since: {data['days_since_latest_raw']:.1f}")

            if data.get('latest_filtered_weight') is not None:
                print(f"    Latest filtered weight: {data['latest_filtered_weight']:.1f} lbs")
                print(f"      Date: {data['latest_filtered_date']}")
                print(f"      Days since: {data['days_since_latest_filtered']:.1f}")

        return user_data

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    user_data = main()
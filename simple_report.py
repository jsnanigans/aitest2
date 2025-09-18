#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import argparse
from typing import Optional, Set, Union
from datetime import datetime

def load_employer_filter(employer_name: str) -> tuple[Set[str], dict]:
    """Load user IDs and start dates for a specific employer (matching report.py logic)

    Returns:
        Tuple of (set of user IDs, dict of user_id -> start_date)
    """
    # Load partners CSV to get employer_id from name
    partners_path = Path("data/partners.csv")
    if not partners_path.exists():
        print(f"Warning: Partners file not found: {partners_path}")
        return set(), {}

    partners_df = pd.read_csv(partners_path)
    employer_rows = partners_df[partners_df['name'] == employer_name]

    if employer_rows.empty:
        available_employers = partners_df[partners_df['name'].str.contains('_EMPLOYER', na=False)]['name'].unique()
        print(f"Employer '{employer_name}' not found. Available employers: {', '.join(sorted(available_employers))}")
        return set(), {}

    employer_id = employer_rows.iloc[0]['id']
    print(f"Found employer_id: {employer_id} for '{employer_name}'")

    # Load user-employer mappings
    user_employers_path = Path("data/2025-09-17-user-employers.csv")
    if not user_employers_path.exists():
        print(f"Warning: User employers file not found: {user_employers_path}")
        return set(), {}

    user_employers_df = pd.read_csv(user_employers_path)
    employer_data = user_employers_df[user_employers_df['employer_id'] == employer_id]

    # Create user ID set and start date mapping
    user_ids = set(employer_data['user_id'].values)
    user_start_dates = dict(zip(employer_data['user_id'], employer_data['start_date']))

    return user_ids, user_start_dates

def get_closest_weight(df: pd.DataFrame, target_date: Union[str, datetime], max_days: int = 10) -> Optional[float]:
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

    # Ensure effectiveDateTime is datetime type
    if 'effectiveDateTime' not in df_copy.columns:
        print("Warning: effectiveDateTime column not found")
        return None

    # Ensure weight column exists
    if 'weight' not in df_copy.columns:
        print("Warning: weight column not found")
        return None

    df_copy['effectiveDateTime'] = pd.to_datetime(df_copy['effectiveDateTime'])

    # Calculate time difference from target date
    df_copy['time_diff'] = abs(df_copy['effectiveDateTime'] - target_date)

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

def main():
    # Parse arguments (matching report.py style)
    parser = argparse.ArgumentParser(description='Simple report for user data analysis')
    parser.add_argument(
        '--employer',
        type=str,
        help='Filter users by employer name (e.g., AMAZON_EMPLOYER, APPLE_EMPLOYER)'
    )
    args = parser.parse_args()

    # Define file paths
    filtered_file = Path("filtered.csv")
    raw_file = Path("data/2025-09-05_nocon.csv")

    # Check files exist
    if not filtered_file.exists():
        print(f"Error: {filtered_file} not found")
        return
    if not raw_file.exists():
        print(f"Error: {raw_file} not found")
        return

    # Load employer filter if specified
    employer_user_ids = None
    user_start_dates = {}
    if args.employer:
        print(f"Loading employer filter for '{args.employer}'...")
        employer_user_ids, user_start_dates = load_employer_filter(args.employer)
        if employer_user_ids:
            print(f"Found {len(employer_user_ids)} users for employer '{args.employer}'")
        else:
            print(f"No users found for employer '{args.employer}'")
            return
    else:
        # Load all user start dates if no employer filter
        user_employers_path = Path("data/2025-09-17-user-employers.csv")
        if user_employers_path.exists():
            user_employers_df = pd.read_csv(user_employers_path)
            user_start_dates = dict(zip(user_employers_df['user_id'], user_employers_df['start_date']))

    # Load filtered data
    df_filtered = pd.read_csv(filtered_file)

    # Load raw data
    df_raw = pd.read_csv(raw_file)

    # Get unique users from both datasets
    filtered_users = set(df_filtered['user_id'].unique())
    raw_users = set(df_raw['user_id'].unique())

    # Find users that exist in BOTH datasets
    users_in_both = filtered_users & raw_users

    # Apply employer filter if specified
    if employer_user_ids:
        # Only keep users that are in both datasets AND in employer list
        users_to_keep = users_in_both & employer_user_ids
        print(f"Found {len(users_to_keep)} employer users in both datasets")

        if not users_to_keep:
            print(f"No users found for employer '{args.employer}' that exist in both datasets")
            return

        # Filter dataframes to only these users
        df_filtered = df_filtered[df_filtered['user_id'].isin(users_to_keep)]
        df_raw = df_raw[df_raw['user_id'].isin(users_to_keep)]

        print(f"Filtered data: {len(df_filtered)} measurements")
        print(f"Raw data: {len(df_raw)} rows")
    else:
        # No employer filter, but still only keep users in both datasets
        users_to_keep = users_in_both
        df_filtered = df_filtered[df_filtered['user_id'].isin(users_to_keep)]
        df_raw = df_raw[df_raw['user_id'].isin(users_to_keep)]
        print(f"Keeping {len(users_to_keep)} users that exist in both datasets")

    # Group by user_id for both datasets
    filtered_by_user = df_filtered.groupby('user_id')
    raw_by_user = df_raw.groupby('user_id')

    # Keep data separate but accessible - now guaranteed both exist for each user
    user_data = {}
    for user_id in users_to_keep:
        user_data[user_id] = {
            'filtered': filtered_by_user.get_group(user_id),
            'raw': raw_by_user.get_group(user_id)
        }

    # Calculate start weights for each user using their individual start dates
    print(f"\nCalculating start weights using individual user start dates...")

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

    if users_without_start_date:
        print(f"Warning: {len(users_without_start_date)} users have no start date in user-employers.csv")

    # Count users with start weights
    users_with_raw_start = sum(1 for v in start_weights.values() if v['raw_start_weight'] is not None)
    users_with_filtered_start = sum(1 for v in start_weights.values() if v['filtered_start_weight'] is not None)
    users_with_both_starts = sum(1 for v in start_weights.values()
                                 if v['raw_start_weight'] is not None and v['filtered_start_weight'] is not None)

    # Output user count summary
    print(f"\n--- Summary ---")
    print(f"Total users kept in memory: {len(users_to_keep)}")
    print(f"Total filtered measurements: {len(df_filtered)}")
    print(f"Total raw measurements: {len(df_raw)}")
    print(f"\n--- Start Weight Summary ---")
    print(f"Users with raw start weight: {users_with_raw_start}")
    print(f"Users with filtered start weight: {users_with_filtered_start}")
    print(f"Users with both start weights: {users_with_both_starts}")

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
        small_diff_count = sum(1 for d in abs_diffs if d < 1.0)

        print(f"\n--- Start Weight Similarity Analysis ---")
        print(f"Identical weights (diff < 0.01 kg): {identical_count} ({identical_count/len(differences)*100:.1f}%)")
        print(f"Similar weights (diff < 1.0 kg): {small_diff_count} ({small_diff_count/len(differences)*100:.1f}%)")
        print(f"Average absolute difference: {sum(abs_diffs)/len(abs_diffs):.2f} kg")
        print(f"Maximum absolute difference: {max(abs_diffs):.2f} kg")
        print(f"Average percentage difference: {sum(pct_diffs)/len(pct_diffs):.2f}%")

        # Sort by absolute difference and show top 5
        differences.sort(key=lambda x: x['abs_diff'], reverse=True)

if __name__ == "__main__":
    main()

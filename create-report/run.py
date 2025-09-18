#!/usr/bin/env python3
"""
Script to load and filter weight measurement data by employer.
Loads filtered users, matches with employer data, and retrieves corresponding raw measurements.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple


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
    }).to_dict('index')

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

        # Step 4: Load raw data only for selected users
        raw_df = load_raw_data_for_users(raw_file, selected_users)

        # Step 5: Filter dataframes to only selected users
        filtered_df = filtered_df[filtered_df['user_id'].isin(selected_users)]

        # Step 6: Create user data structure
        user_data = create_user_data_structure(
            selected_users, raw_df, filtered_df, employer_filtered
        )

        print("\n" + "="*60)
        print("DATA LOADING COMPLETE")
        print("="*60)
        print(f"Ready to analyze {len(user_data):,} users")

        if args.employer:
            print(f"Employer filter: {args.employer}")

        # Show sample of users
        sample_users = list(user_data.keys())[:5]
        print(f"\nSample users: {sample_users}")

        for user_id in sample_users[:2]:
            data = user_data[user_id]
            print(f"\n  User {user_id}:")
            print(f"    Raw measurements: {data['raw_count']}")
            print(f"    Filtered measurements: {data['filtered_count']}")
            print(f"    Start date: {data['start_date']}")
            print(f"    Employer: {data['employer_name']}")

        return user_data

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    user_data = main()
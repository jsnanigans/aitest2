#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import argparse
from typing import Optional, Set

def load_employer_filter(employer_name: str) -> Set[str]:
    """Load user IDs for a specific employer (matching report.py logic)"""
    # Load partners CSV to get employer_id from name
    partners_path = Path("data/partners.csv")
    if not partners_path.exists():
        print(f"Warning: Partners file not found: {partners_path}")
        return set()

    partners_df = pd.read_csv(partners_path)
    employer_rows = partners_df[partners_df['name'] == employer_name]

    if employer_rows.empty:
        available_employers = partners_df[partners_df['name'].str.contains('_EMPLOYER', na=False)]['name'].unique()
        print(f"Employer '{employer_name}' not found. Available employers: {', '.join(sorted(available_employers))}")
        return set()

    employer_id = employer_rows.iloc[0]['id']
    print(f"Found employer_id: {employer_id} for '{employer_name}'")

    # Load user-employer mappings
    user_employers_path = Path("data/2025-09-17-user-employers.csv")
    if not user_employers_path.exists():
        print(f"Warning: User employers file not found: {user_employers_path}")
        return set()

    user_employers_df = pd.read_csv(user_employers_path)
    employer_users = user_employers_df[user_employers_df['employer_id'] == employer_id]['user_id']

    return set(employer_users.values)

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
    if args.employer:
        print(f"Loading employer filter for '{args.employer}'...")
        employer_user_ids = load_employer_filter(args.employer)
        if employer_user_ids:
            print(f"Found {len(employer_user_ids)} users for employer '{args.employer}'")
        else:
            print(f"No users found for employer '{args.employer}'")
            return

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

    # Output user count summary
    print(f"\n--- Summary ---")
    print(f"Total users kept in memory: {len(users_to_keep)}")
    print(f"Total filtered measurements: {len(df_filtered)}")
    print(f"Total raw measurements: {len(df_raw)}")

if __name__ == "__main__":
    main()

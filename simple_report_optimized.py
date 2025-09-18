#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Optional, Set, Union, Dict, Tuple
from datetime import datetime
import time
import logging

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
        logging.warning(f"Partners file not found: {PARTNERS_FILE}")
        return set(), {}

    partners_df = pd.read_csv(PARTNERS_FILE, usecols=['id', 'name'])
    employer_rows = partners_df[partners_df['name'] == employer_name]

    if employer_rows.empty:
        available_employers = partners_df[partners_df['name'].str.contains('_EMPLOYER', na=False)]['name'].unique()
        logging.error(f"Employer '{employer_name}' not found. Available employers: {', '.join(sorted(available_employers))}")
        return set(), {}

    employer_id = employer_rows.iloc[0]['id']
    logging.info(f"Found employer_id: {employer_id} for '{employer_name}'")

    # Load user-employer mappings
    if not USER_EMPLOYERS_FILE.exists():
        logging.warning(f"User employers file not found: {USER_EMPLOYERS_FILE}")
        return set(), {}

    user_employers_df = pd.read_csv(USER_EMPLOYERS_FILE, usecols=['user_id', 'employer_id', 'start_date'])
    employer_data = user_employers_df[user_employers_df['employer_id'] == employer_id]

    # Create user ID set and start date mapping
    user_ids = set(employer_data['user_id'].values)
    user_start_dates = dict(zip(employer_data['user_id'], employer_data['start_date']))

    return user_ids, user_start_dates

def get_closest_weight_vectorized(df: pd.DataFrame, start_dates: Dict[str, str], max_days: int = MAX_DAYS_WEIGHT_WINDOW) -> Dict[str, Optional[float]]:
    """
    Vectorized version to find closest weights for multiple users at once.
    
    Args:
        df: DataFrame with 'user_id', 'effectiveDateTime' and 'weight' columns
        start_dates: Dictionary mapping user_id to start_date
        max_days: Maximum days before/after target date to search
        
    Returns:
        Dictionary mapping user_id to closest weight
    """
    if df.empty or not start_dates:
        return {uid: None for uid in start_dates}
    
    results = {}
    time_window = pd.Timedelta(days=max_days)
    
    # Group by user once
    grouped = df.groupby('user_id')
    
    for user_id, start_date in start_dates.items():
        if user_id not in grouped.groups:
            results[user_id] = None
            continue
            
        user_df = grouped.get_group(user_id)
        target_date = pd.to_datetime(start_date)
        
        # Calculate time differences
        time_diffs = (user_df['effectiveDateTime'] - target_date).abs()
        
        # Filter within window
        mask = time_diffs <= time_window
        if not mask.any():
            results[user_id] = None
            continue
        
        # Get closest weight
        closest_idx = time_diffs[mask].idxmin()
        results[user_id] = user_df.loc[closest_idx, 'weight']
    
    return results

def load_data_efficient(filtered_path: Path, raw_path: Path, user_ids: Optional[Set[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data efficiently with optional user filtering and optimized datetime parsing."""
    start_time = time.time()

    # Determine columns to load
    filtered_cols = ['user_id', 'effectiveDateTime', 'weight']
    raw_cols = ['user_id', 'effectiveDateTime', 'weight']

    # Load with user filtering if provided
    if user_ids:
        # Convert to list for isin optimization
        user_list = list(user_ids)
        
        df_filtered = pd.read_csv(
            filtered_path, 
            usecols=filtered_cols,
            parse_dates=['effectiveDateTime'],  # Parse dates during read
            date_format='ISO8601'  # Specify format for faster parsing
        )
        df_filtered = df_filtered[df_filtered['user_id'].isin(user_list)]

        df_raw = pd.read_csv(
            raw_path, 
            usecols=raw_cols,
            parse_dates=['effectiveDateTime'],
            date_format='ISO8601'
        )
        df_raw = df_raw[df_raw['user_id'].isin(user_list)]
    else:
        df_filtered = pd.read_csv(
            filtered_path, 
            usecols=filtered_cols,
            parse_dates=['effectiveDateTime'],
            date_format='ISO8601'
        )
        df_raw = pd.read_csv(
            raw_path, 
            usecols=raw_cols,
            parse_dates=['effectiveDateTime'],
            date_format='ISO8601'
        )

    logging.info(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    return df_filtered, df_raw

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Optimized simple report for user data analysis')
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
        logging.info(f"Loading employer filter for '{args.employer}'...")
        employer_user_ids, user_start_dates = load_employer_filter(args.employer)
        if employer_user_ids:
            logging.info(f"Found {len(employer_user_ids)} users for employer '{args.employer}'")
        else:
            logging.error(f"No users found for employer '{args.employer}'")
            return
    else:
        # Load all user start dates if no employer filter
        if USER_EMPLOYERS_FILE.exists():
            user_employers_df = pd.read_csv(USER_EMPLOYERS_FILE, usecols=['user_id', 'start_date'])
            user_start_dates = dict(zip(user_employers_df['user_id'], user_employers_df['start_date']))

    # Load data efficiently with optimized datetime parsing
    df_filtered, df_raw = load_data_efficient(FILTERED_FILE, RAW_FILE, employer_user_ids)

    # Get unique users from both datasets (already in memory)
    filtered_users = set(df_filtered['user_id'].unique())
    raw_users = set(df_raw['user_id'].unique())

    # Find users that exist in BOTH datasets
    users_in_both = filtered_users & raw_users

    # Apply employer filter if specified
    if employer_user_ids:
        # Only keep users that are in both datasets AND in employer list
        users_to_keep = users_in_both & employer_user_ids
        logging.info(f"Found {len(users_to_keep)} employer users in both datasets")

        if not users_to_keep:
            logging.error(f"No users found for employer '{args.employer}' that exist in both datasets")
            return

        # Filter dataframes using boolean indexing (faster than isin for large sets)
        user_list = list(users_to_keep)
        df_filtered = df_filtered[df_filtered['user_id'].isin(user_list)]
        df_raw = df_raw[df_raw['user_id'].isin(user_list)]

        logging.info(f"Filtered data: {len(df_filtered)} measurements")
        logging.info(f"Raw data: {len(df_raw)} rows")
    else:
        # No employer filter, but still only keep users in both datasets
        users_to_keep = users_in_both
        user_list = list(users_to_keep)
        df_filtered = df_filtered[df_filtered['user_id'].isin(user_list)]
        df_raw = df_raw[df_raw['user_id'].isin(user_list)]
        logging.info(f"Keeping {len(users_to_keep)} users that exist in both datasets")

    # Filter start dates to only include users we're keeping
    filtered_start_dates = {uid: date for uid, date in user_start_dates.items() if uid in users_to_keep}

    # Calculate start weights using vectorized approach
    logging.info("\nCalculating start weights using individual user start dates...")
    
    raw_start_weights = get_closest_weight_vectorized(df_raw, filtered_start_dates)
    filtered_start_weights = get_closest_weight_vectorized(df_filtered, filtered_start_dates)

    # Combine results
    start_weights = {}
    users_without_start_date = []
    
    for user_id in users_to_keep:
        if user_id in filtered_start_dates:
            start_weights[user_id] = {
                'start_date': filtered_start_dates[user_id],
                'raw_start_weight': raw_start_weights.get(user_id),
                'filtered_start_weight': filtered_start_weights.get(user_id)
            }
        else:
            users_without_start_date.append(user_id)

    if users_without_start_date:
        logging.warning(f"{len(users_without_start_date)} users have no start date in user-employers.csv")

    # Count users with start weights using numpy for efficiency
    raw_weights = [v['raw_start_weight'] for v in start_weights.values()]
    filtered_weights = [v['filtered_start_weight'] for v in start_weights.values()]
    
    users_with_raw_start = sum(1 for w in raw_weights if w is not None)
    users_with_filtered_start = sum(1 for w in filtered_weights if w is not None)
    users_with_both_starts = sum(1 for r, f in zip(raw_weights, filtered_weights) if r is not None and f is not None)

    # Output user count summary
    logging.info(f"\n--- Summary ---")
    logging.info(f"Total users kept in memory: {len(users_to_keep)}")
    logging.info(f"Total filtered measurements: {len(df_filtered)}")
    logging.info(f"Total raw measurements: {len(df_raw)}")
    logging.info(f"\n--- Start Weight Summary ---")
    logging.info(f"Users with raw start weight: {users_with_raw_start}")
    logging.info(f"Users with filtered start weight: {users_with_filtered_start}")
    logging.info(f"Users with both start weights: {users_with_both_starts}")

    # Calculate differences using numpy arrays for speed
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

    # Calculate similarity statistics using numpy
    if differences:
        abs_diffs = np.array([d['abs_diff'] for d in differences])
        pct_diffs = np.array([d['pct_diff'] for d in differences])

        identical_count = np.sum(abs_diffs < 0.01)
        small_diff_count = np.sum(abs_diffs < 1.0)

        logging.info(f"\n--- Start Weight Similarity Analysis ---")
        logging.info(f"Identical weights (diff < 0.01 kg): {identical_count} ({identical_count/len(differences)*100:.1f}%)")
        logging.info(f"Similar weights (diff < 1.0 kg): {small_diff_count} ({small_diff_count/len(differences)*100:.1f}%)")
        logging.info(f"Average absolute difference: {np.mean(abs_diffs):.2f} kg")
        logging.info(f"Maximum absolute difference: {np.max(abs_diffs):.2f} kg")
        logging.info(f"Average percentage difference: {np.mean(pct_diffs):.2f}%")

    # Report total execution time
    logging.info(f"\n--- Performance ---")
    logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

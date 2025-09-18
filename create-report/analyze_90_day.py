#!/usr/bin/env python3
"""
90-Day Weight Loss Analysis Module
Compares raw vs filtered data for users with 90+ days in program
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Constants
DATA_DIR = Path("../data")
EXPORT_DATE = datetime(2025, 9, 11)  # Data export date
NINETY_DAYS_BEFORE = EXPORT_DATE - timedelta(days=90)
USER_EMPLOYERS_FILE = DATA_DIR / "2025-09-17-user-employers.csv"
# These can be overridden by run_analysis.py
RAW_FILE = DATA_DIR / "2025-09-05_nocon.csv"
FILTERED_FILE = DATA_DIR / "2025-09-05_nocon_filtered.csv"
PARTNERS_FILE = DATA_DIR / "partners.csv"

def load_eligible_users(employer_filter: Optional[str] = None) -> Dict[str, datetime]:
    """
    Load users who have been in the program for at least 90 days AND exist in the filtered data.
    
    Returns:
        Dict mapping user_id to start_date for eligible users
    """
    if not USER_EMPLOYERS_FILE.exists():
        raise FileNotFoundError(f"User employers file not found: {USER_EMPLOYERS_FILE}")
    
    # Load user-employer data
    df = pd.read_csv(USER_EMPLOYERS_FILE)
    df['start_date'] = pd.to_datetime(df['start_date'])
    
    # Filter for employer if specified
    if employer_filter:
        if not PARTNERS_FILE.exists():
            logging.warning(f"Partners file not found, skipping employer filter")
        else:
            partners_df = pd.read_csv(PARTNERS_FILE)
            employer_row = partners_df[partners_df['name'] == employer_filter]
            if not employer_row.empty:
                employer_id = employer_row.iloc[0]['id']
                df = df[df['employer_id'] == employer_id]
                logging.info(f"Filtered to {len(df)} users from {employer_filter}")
    
    # Filter for users with 90+ days in program
    eligible_df = df[df['start_date'] <= NINETY_DAYS_BEFORE]
    logging.info(f"Found {len(eligible_df)} users with 90+ days in program")
    
    # OPTIMIZATION: Only include users that exist in the filtered CSV
    # This avoids processing users with no data
    if FILTERED_FILE.exists():
        logging.info("Loading filtered data to identify users with measurements...")
        df_filtered_users = pd.read_csv(FILTERED_FILE, usecols=['user_id'])
        filtered_user_ids = set(df_filtered_users['user_id'].unique())
        eligible_df = eligible_df[eligible_df['user_id'].isin(filtered_user_ids)]
        logging.info(f"Reduced to {len(eligible_df)} users who have filtered weight data")
    
    # Create mapping
    user_start_dates = dict(zip(eligible_df['user_id'], eligible_df['start_date']))
    
    logging.info(f"Will analyze {len(user_start_dates)} eligible users")
    return user_start_dates

def get_weight_at_date(df: pd.DataFrame, target_date: datetime, window_days: int = 20) -> Optional[float]:
    """
    Get weight measurement closest to target date within window.
    
    Args:
        df: DataFrame with effectiveDateTime and weight columns
        target_date: Target date to find weight
        window_days: Maximum days before/after to search
        
    Returns:
        Weight value or None if not found
    """
    if df.empty:
        return None
    
    # Calculate time differences
    df = df.copy()
    df['time_diff'] = abs(df['effectiveDateTime'] - target_date)
    
    # Filter to window
    window = timedelta(days=window_days)
    df_window = df[df['time_diff'] <= window]
    
    if df_window.empty:
        return None
    
    # Get closest measurement
    closest_idx = df_window['time_diff'].idxmin()
    return df_window.loc[closest_idx, 'weight']

def calculate_90_day_metrics(user_start_dates: Dict[str, datetime]) -> pd.DataFrame:
    """
    Calculate 90-day weight loss metrics for eligible users.
    
    Returns:
        DataFrame with columns:
        - user_id
        - start_date
        - raw_start_weight
        - raw_90_day_weight
        - raw_loss_kg
        - raw_loss_pct
        - filtered_start_weight
        - filtered_90_day_weight
        - filtered_loss_kg
        - filtered_loss_pct
        - difference_pct (filtered - raw)
    """
    # Load data files - but only for users we care about
    logging.info("Loading weight data files...")
    
    # Get the list of user IDs we need
    target_user_ids = list(user_start_dates.keys())
    logging.info(f"Loading data for {len(target_user_ids)} target users...")
    
    # Load full data first
    df_raw = pd.read_csv(RAW_FILE, usecols=['user_id', 'effectiveDateTime', 'weight'])
    df_filtered = pd.read_csv(FILTERED_FILE, usecols=['user_id', 'effectiveDateTime', 'weight'])
    
    # Filter to only our target users IMMEDIATELY
    df_raw = df_raw[df_raw['user_id'].isin(target_user_ids)]
    df_filtered = df_filtered[df_filtered['user_id'].isin(target_user_ids)]
    
    logging.info(f"Filtered to {len(df_raw['user_id'].unique())} users in raw data")
    logging.info(f"Filtered to {len(df_filtered['user_id'].unique())} users in filtered data")
    
    # Convert datetime columns
    df_raw['effectiveDateTime'] = pd.to_datetime(df_raw['effectiveDateTime'])
    df_filtered['effectiveDateTime'] = pd.to_datetime(df_filtered['effectiveDateTime'])
    
    results = []
    processed = 0
    skipped_no_data = 0
    
    for user_id, start_date in user_start_dates.items():
        # Get user's data
        user_raw = df_raw[df_raw['user_id'] == user_id]
        user_filtered = df_filtered[df_filtered['user_id'] == user_id]
        
        # Skip if no data
        if user_raw.empty and user_filtered.empty:
            skipped_no_data += 1
            continue
        
        # Calculate 90-day date
        day_90 = start_date + timedelta(days=90)
        
        # Get weights at start and 90 days
        raw_start = get_weight_at_date(user_raw, start_date)
        raw_90 = get_weight_at_date(user_raw, day_90)
        filtered_start = get_weight_at_date(user_filtered, start_date)
        filtered_90 = get_weight_at_date(user_filtered, day_90)
        
        # Calculate losses
        raw_loss_kg = None
        raw_loss_pct = None
        if raw_start and raw_90:
            raw_loss_kg = raw_start - raw_90
            raw_loss_pct = (raw_loss_kg / raw_start) * 100
        
        filtered_loss_kg = None
        filtered_loss_pct = None
        if filtered_start and filtered_90:
            filtered_loss_kg = filtered_start - filtered_90
            filtered_loss_pct = (filtered_loss_kg / filtered_start) * 100
        
        # Calculate difference
        difference_pct = None
        if raw_loss_pct is not None and filtered_loss_pct is not None:
            difference_pct = filtered_loss_pct - raw_loss_pct
        
        results.append({
            'user_id': user_id,
            'start_date': start_date,
            'raw_start_weight': raw_start,
            'raw_90_day_weight': raw_90,
            'raw_loss_kg': raw_loss_kg,
            'raw_loss_pct': raw_loss_pct,
            'filtered_start_weight': filtered_start,
            'filtered_90_day_weight': filtered_90,
            'filtered_loss_kg': filtered_loss_kg,
            'filtered_loss_pct': filtered_loss_pct,
            'difference_pct': difference_pct
        })
        
        processed += 1
        if processed % 100 == 0:
            logging.info(f"Processed {processed} users...")
    
    df_results = pd.DataFrame(results)
    
    if skipped_no_data > 0:
        logging.info(f"Skipped {skipped_no_data} users with no weight data")
    
    logging.info(f"Completed analysis for {len(df_results)} users with data")
    
    return df_results

def generate_summary_statistics(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for the 90-day analysis.
    
    Returns:
        Dictionary with key statistics
    """
    # Filter to users with complete data
    complete_data = df[
        (df['raw_loss_pct'].notna()) & 
        (df['filtered_loss_pct'].notna())
    ]
    
    stats = {
        'total_users': len(df),
        'users_with_complete_data': len(complete_data),
        'raw_avg_loss_pct': complete_data['raw_loss_pct'].mean() if not complete_data.empty else 0,
        'filtered_avg_loss_pct': complete_data['filtered_loss_pct'].mean() if not complete_data.empty else 0,
        'avg_difference_pct': complete_data['difference_pct'].mean() if not complete_data.empty else 0,
        'raw_success_rate': (complete_data['raw_loss_pct'] > 0).mean() * 100 if not complete_data.empty else 0,
        'filtered_success_rate': (complete_data['filtered_loss_pct'] > 0).mean() * 100 if not complete_data.empty else 0,
        'median_raw_loss_pct': complete_data['raw_loss_pct'].median() if not complete_data.empty else 0,
        'median_filtered_loss_pct': complete_data['filtered_loss_pct'].median() if not complete_data.empty else 0,
    }
    
    # Add distribution statistics
    if not complete_data.empty:
        stats['raw_std_dev'] = complete_data['raw_loss_pct'].std()
        stats['filtered_std_dev'] = complete_data['filtered_loss_pct'].std()
        
        # Count users by outcome
        stats['both_show_loss'] = ((complete_data['raw_loss_pct'] > 0) & 
                                   (complete_data['filtered_loss_pct'] > 0)).sum()
        stats['only_filtered_shows_loss'] = ((complete_data['raw_loss_pct'] <= 0) & 
                                             (complete_data['filtered_loss_pct'] > 0)).sum()
        stats['only_raw_shows_loss'] = ((complete_data['raw_loss_pct'] > 0) & 
                                        (complete_data['filtered_loss_pct'] <= 0)).sum()
        stats['both_show_gain'] = ((complete_data['raw_loss_pct'] <= 0) & 
                                   (complete_data['filtered_loss_pct'] <= 0)).sum()
    
    return stats

def identify_case_studies(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Identify representative users for case studies.
    
    Returns:
        Dictionary with case study categories and user data
    """
    complete_data = df[
        (df['raw_loss_pct'].notna()) & 
        (df['filtered_loss_pct'].notna())
    ]
    
    if complete_data.empty:
        return {}
    
    cases = {}
    
    # High success (>10% loss in both)
    high_success = complete_data[
        (complete_data['raw_loss_pct'] > 10) & 
        (complete_data['filtered_loss_pct'] > 10)
    ]
    if not high_success.empty:
        cases['high_success'] = high_success.iloc[0]
    
    # Moderate success (5-10% loss)
    moderate_success = complete_data[
        (complete_data['filtered_loss_pct'] > 5) & 
        (complete_data['filtered_loss_pct'] <= 10)
    ]
    if not moderate_success.empty:
        cases['moderate_success'] = moderate_success.iloc[0]
    
    # Filtering made difference (>2% improvement)
    filtering_helped = complete_data[complete_data['difference_pct'] > 2]
    if not filtering_helped.empty:
        cases['filtering_helped'] = filtering_helped.nlargest(1, 'difference_pct').iloc[0]
    
    # Filtering hurt results (made it worse)
    filtering_hurt = complete_data[complete_data['difference_pct'] < -1]
    if not filtering_hurt.empty:
        cases['filtering_hurt'] = filtering_hurt.nsmallest(1, 'difference_pct').iloc[0]
    
    # Minimal filtering impact
    minimal_impact = complete_data[abs(complete_data['difference_pct']) < 0.5]
    if not minimal_impact.empty:
        cases['minimal_impact'] = minimal_impact.iloc[0]
    
    return cases

def main(employer_filter: Optional[str] = None, output_dir: Path = Path(".")):
    """
    Run the 90-day analysis and save results.
    
    Args:
        employer_filter: Optional employer name to filter by
        output_dir: Directory to save output files
    """
    # Load eligible users
    user_start_dates = load_eligible_users(employer_filter)
    
    if not user_start_dates:
        logging.error("No eligible users found")
        return
    
    # Calculate metrics
    df_metrics = calculate_90_day_metrics(user_start_dates)
    
    # Save raw data
    output_file = output_dir / "90_day_analysis.csv"
    df_metrics.to_csv(output_file, index=False)
    logging.info(f"Saved detailed results to {output_file}")
    
    # Generate summary statistics
    stats = generate_summary_statistics(df_metrics)
    
    # Print summary
    logging.info("\n" + "="*60)
    logging.info("90-DAY WEIGHT LOSS ANALYSIS SUMMARY")
    logging.info("="*60)
    logging.info(f"Total eligible users: {stats['total_users']}")
    logging.info(f"Users with complete data: {stats['users_with_complete_data']}")
    logging.info(f"\nAverage Weight Loss:")
    logging.info(f"  Raw data: {stats['raw_avg_loss_pct']:.2f}%")
    logging.info(f"  Filtered data: {stats['filtered_avg_loss_pct']:.2f}%")
    logging.info(f"  Difference: {stats['avg_difference_pct']:+.2f}%")
    logging.info(f"\nSuccess Rate (% who lost weight):")
    logging.info(f"  Raw data: {stats['raw_success_rate']:.1f}%")
    logging.info(f"  Filtered data: {stats['filtered_success_rate']:.1f}%")
    logging.info(f"\nMedian Weight Loss:")
    logging.info(f"  Raw data: {stats['median_raw_loss_pct']:.2f}%")
    logging.info(f"  Filtered data: {stats['median_filtered_loss_pct']:.2f}%")
    
    # Identify case studies
    cases = identify_case_studies(df_metrics)
    if cases:
        logging.info("\n" + "="*60)
        logging.info("IDENTIFIED CASE STUDIES")
        logging.info("="*60)
        for case_type, user_data in cases.items():
            logging.info(f"\n{case_type.upper().replace('_', ' ')}:")
            logging.info(f"  User ID: {user_data['user_id']}")
            logging.info(f"  Raw loss: {user_data['raw_loss_pct']:.2f}%")
            logging.info(f"  Filtered loss: {user_data['filtered_loss_pct']:.2f}%")
            logging.info(f"  Difference: {user_data['difference_pct']:+.2f}%")
    
    return df_metrics, stats, cases

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="90-day weight loss analysis")
    parser.add_argument('--employer', type=str, help='Filter by employer name')
    parser.add_argument('--output-dir', type=Path, default=Path("."), 
                       help='Output directory for results')
    args = parser.parse_args()
    
    main(args.employer, args.output_dir)
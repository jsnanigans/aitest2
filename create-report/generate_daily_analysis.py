#!/usr/bin/env python3
"""
Daily Weight Analysis Module
Generates detailed day-by-day comparison of raw vs filtered weight data
Uses the same "closest value" logic as the 90-day analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import logging
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Import from existing module
from analyze_90_day import get_weight_at_date, RAW_FILE, FILTERED_FILE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_weight_with_offset(df: pd.DataFrame, target_date: datetime, 
                          window_days: int = 20) -> Tuple[Optional[float], Optional[int]]:
    """
    Get weight and days offset from target date.
    
    Args:
        df: DataFrame with effectiveDateTime and weight columns
        target_date: Target date to find weight
        window_days: Maximum days before/after to search
        
    Returns:
        Tuple of (weight, days_offset) or (None, None) if not found
    """
    if df.empty:
        return None, None
    
    # Calculate time differences
    df = df.copy()
    df['time_diff'] = (df['effectiveDateTime'] - target_date).dt.days
    df['abs_diff'] = abs(df['time_diff'])
    
    # Filter to window
    df_window = df[df['abs_diff'] <= window_days]
    
    if df_window.empty:
        return None, None
    
    # Get closest measurement
    closest_idx = df_window['abs_diff'].idxmin()
    return df_window.loc[closest_idx, 'weight'], int(df_window.loc[closest_idx, 'time_diff'])

def preprocess_user_data(df: pd.DataFrame, user_ids: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Pre-filter and index data by user for faster lookups.
    
    Args:
        df: Full dataset
        user_ids: List of user IDs to process
        
    Returns:
        Dictionary mapping user_id to their DataFrame
    """
    # Filter to target users first
    df_filtered = df[df['user_id'].isin(user_ids)]
    
    # Group by user and store as dict for O(1) lookup
    user_data = {}
    for user_id in user_ids:
        user_df = df_filtered[df_filtered['user_id'] == user_id].copy()
        if not user_df.empty:
            # Sort by date for potential optimizations
            user_df = user_df.sort_values('effectiveDateTime')
            user_data[user_id] = user_df
        else:
            user_data[user_id] = pd.DataFrame()
    
    return user_data

def process_user_batch(
    batch_users: List[str],
    user_start_dates: Dict[str, datetime],
    raw_user_data: Dict[str, pd.DataFrame],
    filtered_user_data: Dict[str, pd.DataFrame],
    max_days: int = 180
) -> List[Dict]:
    """
    Process a batch of users and generate their daily records.
    
    Args:
        batch_users: List of user IDs in this batch
        user_start_dates: Mapping of user_id to start_date
        raw_user_data: Pre-processed raw data by user
        filtered_user_data: Pre-processed filtered data by user
        max_days: Maximum days to analyze
        
    Returns:
        List of record dictionaries
    """
    batch_records = []
    
    for user_id in batch_users:
        start_date = user_start_dates[user_id]
        user_raw = raw_user_data.get(user_id, pd.DataFrame())
        user_filtered = filtered_user_data.get(user_id, pd.DataFrame())
        
        # Skip if no data at all
        if user_raw.empty and user_filtered.empty:
            continue
        
        # Get start weights for reference (day 0)
        start_raw = get_weight_at_date(user_raw, start_date) if not user_raw.empty else None
        start_filtered = get_weight_at_date(user_filtered, start_date) if not user_filtered.empty else None
        
        # Generate daily records
        for day_num in range(max_days + 1):
            current_date = start_date + timedelta(days=day_num)
            
            # Get weights with timing info
            raw_weight, raw_offset = get_weight_with_offset(user_raw, current_date) if not user_raw.empty else (None, None)
            filtered_weight, filtered_offset = get_weight_with_offset(user_filtered, current_date) if not user_filtered.empty else (None, None)
            
            # Calculate cumulative losses
            raw_loss_kg = None
            raw_loss_pct = None
            if start_raw and raw_weight:
                raw_loss_kg = start_raw - raw_weight
                raw_loss_pct = (raw_loss_kg / start_raw) * 100
            
            filtered_loss_kg = None
            filtered_loss_pct = None
            if start_filtered and filtered_weight:
                filtered_loss_kg = start_filtered - filtered_weight
                filtered_loss_pct = (filtered_loss_kg / start_filtered) * 100
            
            # Calculate divergence
            div_kg = None
            div_pct = None
            if filtered_weight is not None and raw_weight is not None:
                div_kg = filtered_weight - raw_weight
            if filtered_loss_pct is not None and raw_loss_pct is not None:
                div_pct = filtered_loss_pct - raw_loss_pct
            
            batch_records.append({
                'user_id': user_id,
                'day_number': day_num,
                'date': current_date.strftime('%Y-%m-%d'),
                'raw_weight': round(raw_weight, 2) if raw_weight else None,
                'raw_days_offset': raw_offset,
                'filtered_weight': round(filtered_weight, 2) if filtered_weight else None,
                'filtered_days_offset': filtered_offset,
                'raw_cumulative_loss_kg': round(raw_loss_kg, 2) if raw_loss_kg else None,
                'raw_cumulative_loss_pct': round(raw_loss_pct, 2) if raw_loss_pct else None,
                'filtered_cumulative_loss_kg': round(filtered_loss_kg, 2) if filtered_loss_kg else None,
                'filtered_cumulative_loss_pct': round(filtered_loss_pct, 2) if filtered_loss_pct else None,
                'divergence_kg': round(div_kg, 2) if div_kg else None,
                'divergence_pct': round(div_pct, 2) if div_pct else None,
                'has_raw_measurement': raw_weight is not None,
                'has_filtered_measurement': filtered_weight is not None
            })
    
    return batch_records

def generate_daily_report(
    user_start_dates: Dict[str, datetime],
    output_path: Path,
    max_days: int = 180,
    batch_size: int = 50
) -> Dict:
    """
    Generate detailed daily weight analysis for all users.
    
    Args:
        user_start_dates: Dict mapping user_id to start_date
        output_path: Directory to save output files
        max_days: Maximum days to analyze (default 180)
        batch_size: Users to process per batch
        
    Returns:
        Summary statistics dictionary
    """
    start_time = time.time()
    output_file = output_path / "daily_weight_analysis.csv"
    
    # Load data once - try cache first
    load_start = time.time()

    try:
        from data_cache import data_cache
        logging.info("Loading weight data from cache...")
        weight_cols = ['user_id', 'effectiveDateTime', 'weight']
        df_raw = data_cache.get_dataframe(RAW_FILE, weight_cols)
        df_filtered = data_cache.get_dataframe(FILTERED_FILE, weight_cols)
    except:
        # Fallback to direct loading
        logging.info("Loading weight data files directly...")

        if not RAW_FILE.exists():
            logging.error(f"Raw file not found: {RAW_FILE}")
            return {}

        if not FILTERED_FILE.exists():
            logging.error(f"Filtered file not found: {FILTERED_FILE}")
            return {}

        # Load with minimal columns for memory efficiency
        df_raw = pd.read_csv(RAW_FILE, usecols=['user_id', 'effectiveDateTime', 'weight'])
        df_filtered = pd.read_csv(FILTERED_FILE, usecols=['user_id', 'effectiveDateTime', 'weight'])

        # Convert datetime columns
        df_raw['effectiveDateTime'] = pd.to_datetime(df_raw['effectiveDateTime'])
        df_filtered['effectiveDateTime'] = pd.to_datetime(df_filtered['effectiveDateTime'])
    
    logging.info(f"Data loaded in {time.time() - load_start:.1f} seconds")
    logging.info(f"Raw data: {len(df_raw):,} records for {df_raw['user_id'].nunique():,} users")
    logging.info(f"Filtered data: {len(df_filtered):,} records for {df_filtered['user_id'].nunique():,} users")
    
    # Pre-process data for performance
    user_ids = list(user_start_dates.keys())
    logging.info(f"Pre-processing data for {len(user_ids)} target users...")
    preprocess_start = time.time()
    
    raw_user_data = preprocess_user_data(df_raw, user_ids)
    filtered_user_data = preprocess_user_data(df_filtered, user_ids)
    
    logging.info(f"Pre-processing completed in {time.time() - preprocess_start:.1f} seconds")
    
    # Free original dataframes from memory
    del df_raw
    del df_filtered
    
    # Initialize tracking variables
    first_batch = True
    total_records = 0
    total_users_processed = 0
    
    # Process users in batches - use parallel processing for large datasets
    logging.info(f"Processing {len(user_ids)} users in batches of {batch_size}...")
    processing_start = time.time()

    # Determine if we should use parallel processing
    use_parallel = len(user_ids) > 100

    if use_parallel:
        # Parallel processing for large datasets
        logging.info(f"Using parallel processing with {multiprocessing.cpu_count()} CPUs...")

        # Process all batches in parallel and collect results
        all_records = []
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = []
            for batch_start in range(0, len(user_ids), batch_size):
                batch_end = min(batch_start + batch_size, len(user_ids))
                batch_users = user_ids[batch_start:batch_end]

                future = executor.submit(
                    process_user_batch,
                    batch_users,
                    user_start_dates,
                    raw_user_data,
                    filtered_user_data,
                    max_days
                )
                futures.append((batch_start, future))

            # Collect results in order
            for batch_start, future in futures:
                batch_records = future.result()
                all_records.extend(batch_records)
                total_records += len(batch_records)

                # Progress reporting
                if batch_start % (batch_size * 10) == 0:
                    logging.info(f"Processed {batch_start + batch_size} users...")

        # Write all records at once
        if all_records:
            df_all = pd.DataFrame(all_records)
            df_all.to_csv(output_file, index=False)
            total_users_processed = len(user_ids)

    else:
        # Sequential processing for small datasets
        for batch_start in range(0, len(user_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(user_ids))
            batch_users = user_ids[batch_start:batch_end]

            batch_time_start = time.time()

            # Process batch
            batch_records = process_user_batch(
                batch_users,
                user_start_dates,
                raw_user_data,
                filtered_user_data,
                max_days
            )

            # Write batch to CSV
            if batch_records:
                df_batch = pd.DataFrame(batch_records)
                df_batch.to_csv(output_file, mode='w' if first_batch else 'a',
                               header=first_batch, index=False)
                first_batch = False
                total_records += len(batch_records)
                total_users_processed += len(batch_users)

            # Progress reporting
            batch_time = time.time() - batch_time_start
            avg_time_per_user = batch_time / len(batch_users)
        remaining_users = len(user_ids) - batch_end
        eta_seconds = remaining_users * avg_time_per_user
        
        logging.info(f"  Batch {batch_start+1}-{batch_end}: "
                    f"{len(batch_records):,} records in {batch_time:.1f}s "
                    f"(ETA: {eta_seconds:.0f}s)")
    
    processing_time = time.time() - processing_start
    logging.info(f"Processing completed in {processing_time:.1f} seconds")
    
    # Generate summary statistics
    logging.info("Generating summary statistics...")
    summary = generate_summary_stats(output_file, total_records, len(user_ids), max_days)
    summary['processing_time_seconds'] = time.time() - start_time
    summary['records_per_second'] = total_records / summary['processing_time_seconds'] if total_records > 0 else 0
    
    # Save summary
    summary_file = output_path / "daily_analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logging.info(f"Analysis complete! Total time: {summary['processing_time_seconds']:.1f} seconds")
    logging.info(f"Output saved to: {output_file}")
    logging.info(f"Summary saved to: {summary_file}")
    
    return summary

def generate_summary_stats(csv_file: Path, total_records: int, 
                          total_users: int, max_days: int) -> Dict:
    """
    Generate summary statistics from the output CSV.
    
    Args:
        csv_file: Path to the generated CSV file
        total_records: Total number of records written
        total_users: Total number of users processed
        max_days: Maximum days analyzed
        
    Returns:
        Dictionary of summary statistics
    """
    summary = {
        'total_records': total_records,
        'total_users': total_users,
        'max_days_analyzed': max_days,
        'output_file': str(csv_file),
        'generated_at': datetime.now().isoformat()
    }
    
    if csv_file.exists() and total_records > 0:
        # Read a sample to calculate statistics
        try:
            # Read only specific columns for memory efficiency
            df_sample = pd.read_csv(csv_file, 
                                   usecols=['day_number', 'has_raw_measurement', 
                                           'has_filtered_measurement', 'divergence_pct'])
            
            # Data availability stats
            summary['avg_raw_data_availability'] = df_sample['has_raw_measurement'].mean() * 100
            summary['avg_filtered_data_availability'] = df_sample['has_filtered_measurement'].mean() * 100
            
            # Divergence stats (where both measurements exist)
            divergence_data = df_sample['divergence_pct'].dropna()
            if not divergence_data.empty:
                summary['avg_divergence_pct'] = float(divergence_data.mean())
                summary['max_divergence_pct'] = float(divergence_data.abs().max())
                summary['median_divergence_pct'] = float(divergence_data.median())
                
                # Count days where raw/filtered disagree on direction
                summary['days_with_conflicting_direction'] = int((divergence_data.abs() > 0.5).sum())
            
            # Key milestone stats (day 30, 60, 90)
            for milestone in [30, 60, 90]:
                milestone_data = df_sample[df_sample['day_number'] == milestone]
                if not milestone_data.empty:
                    summary[f'day_{milestone}_data_availability'] = {
                        'raw': milestone_data['has_raw_measurement'].mean() * 100,
                        'filtered': milestone_data['has_filtered_measurement'].mean() * 100
                    }
            
        except Exception as e:
            logging.warning(f"Could not generate full statistics: {e}")
    
    return summary

def main(user_start_dates: Dict[str, datetime] = None, 
         output_path: Path = None,
         max_days: int = 180) -> Dict:
    """
    Main entry point for daily analysis generation.
    
    Args:
        user_start_dates: Optional dict of user_id to start_date
        output_path: Optional output directory
        max_days: Maximum days to analyze
        
    Returns:
        Summary statistics dictionary
    """
    if output_path is None:
        output_path = Path(".")
    
    # If no user_start_dates provided, load from 90-day analysis
    if user_start_dates is None:
        from analyze_90_day import load_eligible_users
        logging.info("Loading eligible users...")
        user_start_dates = load_eligible_users()
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run the analysis
    return generate_daily_report(user_start_dates, output_path, max_days=max_days)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate daily weight analysis report")
    parser.add_argument('--max-days', type=int, default=180,
                       help='Maximum days to analyze (default: 180)')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Users per batch (default: 50)')
    parser.add_argument('--output', type=str, default=".",
                       help='Output directory (default: current directory)')
    
    args = parser.parse_args()
    
    main(output_path=Path(args.output), max_days=args.max_days)
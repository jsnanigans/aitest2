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

        # Step 4: Load raw data only for selected users
        raw_df = load_raw_data_for_users(raw_file, selected_users)

        # Step 5: Filter dataframes to only selected users
        filtered_df = filtered_df[filtered_df['user_id'].isin(selected_users)]

        # Step 6: Create user data structure
        user_data = create_user_data_structure(
            selected_users, raw_df, filtered_df, employer_filtered
        )

        # Step 7: Analyze weights near start dates
        user_data = find_closest_weight_to_start(user_data)

        # Step 8: Export analysis to markdown
        report_path = export_analysis_to_markdown(user_data, args.employer)

        print("\n" + "="*60)
        print(f"Ready to analyze {len(user_data):,} users")

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
            print(f"    Measurements: {data['raw_count']} raw, {data['filtered_count']} filtered")

            if data.get('closest_raw_weight') is not None:
                print(f"    Closest raw weight: {data['closest_raw_weight']:.1f} lbs")
                print(f"      Date: {data['closest_raw_date']}")
                print(f"      Days from start: {data['days_from_start_raw']:.1f}")

            if data.get('closest_filtered_weight') is not None:
                print(f"    Closest filtered weight: {data['closest_filtered_weight']:.1f} lbs")
                print(f"      Date: {data['closest_filtered_date']}")
                print(f"      Days from start: {data['days_from_start_filtered']:.1f}")
                if data.get('closest_filtered_quality') is not None:
                    print(f"      Quality score: {data['closest_filtered_quality']:.3f}")

        return user_data

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    user_data = main()
#!/usr/bin/env python3
"""Compare user value counts between filtered.csv and reference CSV"""

import pandas as pd
from pathlib import Path
import sys

def compare_user_counts(filtered_path="filtered.csv", reference_path="data/2025-09-05_nocon.csv"):
    """Compare user measurement counts between two CSV files"""

    # Read CSVs
    print(f"Reading {filtered_path}...")
    filtered_df = pd.read_csv(filtered_path)

    print(f"Reading {reference_path}...")
    reference_df = pd.read_csv(reference_path)

    # Count values per user
    filtered_counts = filtered_df.groupby('user_id').size().to_dict()
    reference_counts = reference_df.groupby('user_id').size().to_dict()

    # Get only users that exist in both files
    common_users = set(filtered_counts.keys()) & set(reference_counts.keys())

    print(f"\nTotal users in filtered.csv: {len(filtered_counts)}")
    print(f"Total users in reference CSV: {len(reference_counts)}")
    print(f"Users present in both files: {len(common_users)}")

    # Compare counts for common users only
    comparison_results = []

    for user_id in sorted(common_users):
        filtered_count = filtered_counts.get(user_id, 0)
        reference_count = reference_counts.get(user_id, 0)
        difference = filtered_count - reference_count

        comparison_results.append({
            'user_id': user_id,
            'filtered_count': filtered_count,
            'reference_count': reference_count,
            'difference': difference,
            'percent_change': (difference / reference_count * 100) if reference_count > 0 else 0 if filtered_count == 0 else float('inf')
        })

    # Create DataFrame for results
    results_df = pd.DataFrame(comparison_results)

    # Summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Analyzing {len(results_df)} users that exist in both files")

    # All results are for users in both files
    if not results_df.empty:
        print(f"  Total measurements in filtered.csv: {results_df['filtered_count'].sum()}")
        print(f"  Total measurements in reference CSV: {results_df['reference_count'].sum()}")
        print(f"  Average difference: {results_df['difference'].mean():.2f}")
        print(f"  Users with fewer values in filtered: {len(results_df[results_df['difference'] < 0])}")
        print(f"  Users with more values in filtered: {len(results_df[results_df['difference'] > 0])}")
        print(f"  Users with same count: {len(results_df[results_df['difference'] == 0])}")

    # Show largest differences
    print("\n=== TOP 10 USERS WITH MOST VALUES REMOVED (filtered < reference) ===")
    top_removed = results_df[results_df['difference'] < 0].nsmallest(10, 'difference')
    if not top_removed.empty:
        for _, row in top_removed.iterrows():
            print(f"  {row['user_id']}: {row['filtered_count']} vs {row['reference_count']} (removed {abs(row['difference'])} values)")
    else:
        print("  No users had values removed")

    print("\n=== TOP 10 USERS WITH MOST VALUES ADDED (filtered > reference) ===")
    top_added = results_df[results_df['difference'] > 0].nlargest(10, 'difference')
    if not top_added.empty:
        for _, row in top_added.iterrows():
            print(f"  {row['user_id']}: {row['filtered_count']} vs {row['reference_count']} (added {row['difference']} values)")
    else:
        print("  No users had values added")

    # Save full results to CSV
    output_file = "user_count_comparison.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nFull comparison results saved to: {output_file}")

    return results_df

if __name__ == "__main__":
    # Allow custom file paths as arguments
    filtered_path = sys.argv[1] if len(sys.argv) > 1 else "filtered.csv"
    reference_path = sys.argv[2] if len(sys.argv) > 2 else "data/2025-09-05_nocon.csv"

    compare_user_counts(filtered_path, reference_path)
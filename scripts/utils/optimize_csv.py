#!/usr/bin/env python3
"""
CSV Optimizer - Sorts weight data by user_id and then chronologically within each user.
This ensures all data for each user is grouped together and in time order.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict


def parse_timestamp(date_str, return_none_on_invalid=False):
    """Parse various timestamp formats, filtering out impossible dates."""
    if not date_str:
        return None if return_none_on_invalid else datetime.min
    
    try:
        parsed_date = None
        if "T" in date_str:
            parsed_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        elif " " in date_str:
            parsed_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        else:
            parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
        
        # Filter out impossible dates (must be between year 1 and 9999 for matplotlib)
        if parsed_date and (parsed_date.year < 1 or parsed_date.year > 9999):
            return None if return_none_on_invalid else datetime.min
        
        return parsed_date
    except:
        return None if return_none_on_invalid else datetime.min


def optimize_csv(input_file, output_file):
    """
    Read CSV, sort by user_id then by timestamp, and write optimized version.
    """
    print(f"Reading {input_file}...")
    
    # Read all data
    rows_by_user = defaultdict(list)
    header = None
    total_rows = 0
    invalid_dates = 0
    filtered_rows = 0
    
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        
        for row in reader:
            total_rows += 1
            if total_rows % 10000 == 0:
                print(f"  Read {total_rows:,} rows...")
            
            user_id = row.get('user_id', '')
            if user_id:
                # Check if date is valid before adding
                date_str = row.get('effectiveDateTime') or row.get('effectiveDateTime') or row.get('date', '')
                parsed_date = parse_timestamp(date_str, return_none_on_invalid=True)
                
                if parsed_date is None:
                    invalid_dates += 1
                    filtered_rows += 1
                    if invalid_dates <= 5:  # Show first few invalid dates
                        print(f"  WARNING: Filtering out row with invalid date: {date_str} (user: {user_id[:8]}...)")
                    continue
                    
                # Additional validation: check year is reasonable (1900-2100)
                if parsed_date.year < 1900 or parsed_date.year > 2100:
                    filtered_rows += 1
                    if filtered_rows <= 5:
                        print(f"  WARNING: Filtering out row with unreasonable year {parsed_date.year}: {date_str} (user: {user_id[:8]}...)")
                    continue
                
                rows_by_user[user_id].append(row)
    
    print(f"  Total rows read: {total_rows:,}")
    print(f"  Unique users: {len(rows_by_user):,}")
    if filtered_rows > 0:
        print(f"  Rows filtered out due to invalid dates: {filtered_rows:,}")
        if invalid_dates > 5:
            print(f"    (showing first 5 of {invalid_dates:,} invalid dates)")
    
    # Sort each user's data chronologically and filter out users with only invalid dates
    print("\nSorting and validating data...")
    users_to_remove = []
    for user_id in rows_by_user:
        # Remove any remaining rows with unparseable dates during sorting
        valid_rows = []
        for row in rows_by_user[user_id]:
            date_str = row.get('effectiveDateTime') or row.get('effectiveDateTime') or row.get('date', '')
            if parse_timestamp(date_str, return_none_on_invalid=True) is not None:
                valid_rows.append(row)
        
        if valid_rows:
            valid_rows.sort(
                key=lambda r: parse_timestamp(
                    r.get('effectiveDateTime') or 
                    r.get('effectiveDateTime') or 
                    r.get('date', '')
                )
            )
            rows_by_user[user_id] = valid_rows
        else:
            users_to_remove.append(user_id)
    
    # Remove users with no valid data
    for user_id in users_to_remove:
        del rows_by_user[user_id]
    
    if users_to_remove:
        print(f"  Removed {len(users_to_remove)} users with no valid dates")
    
    # Sort users by their first measurement date (for deterministic ordering)
    sorted_users = sorted(
        rows_by_user.keys(),
        key=lambda uid: (
            parse_timestamp(
                rows_by_user[uid][0].get('effectiveDateTime') or
                rows_by_user[uid][0].get('effectiveDateTime') or
                rows_by_user[uid][0].get('date', '')
            ),
            uid  # Secondary sort by user_id for stability
        )
    )
    
    # Write optimized CSV
    print(f"\nWriting optimized CSV to {output_file}...")
    written_rows = 0
    users_written = 0
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        
        for user_id in sorted_users:
            users_written += 1
            if users_written % 1000 == 0:
                print(f"  Written {users_written:,} users ({written_rows:,} rows)...")
            
            for row in rows_by_user[user_id]:
                writer.writerow(row)
                written_rows += 1
    
    print(f"\nOptimization complete!")
    print(f"  Users written: {users_written:,}")
    print(f"  Rows written: {written_rows:,}")
    
    # Analyze data distribution
    print(f"\nData distribution analysis:")
    measurement_counts = defaultdict(int)
    for user_id, rows in rows_by_user.items():
        count = len(rows)
        if count < 10:
            measurement_counts['1-9'] += 1
        elif count < 20:
            measurement_counts['10-19'] += 1
        elif count < 50:
            measurement_counts['20-49'] += 1
        elif count < 100:
            measurement_counts['50-99'] += 1
        else:
            measurement_counts['100+'] += 1
    
    for range_label in ['1-9', '10-19', '20-49', '50-99', '100+']:
        if range_label in measurement_counts:
            count = measurement_counts[range_label]
            pct = (count / len(rows_by_user)) * 100
            print(f"  {range_label:8} measurements: {count:6,} users ({pct:5.1f}%)")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optimize CSV by sorting data by user and chronologically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default file
  %(prog)s
  
  # Process specific input/output files  
  %(prog)s input.csv output_sorted.csv
  
  # Process with custom paths
  %(prog)s data/weights.csv data/weights_optimized.csv
"""
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        default='./data/2025-09-05.csv',
        help='Input CSV file (default: ./data/2025-09-05.csv)'
    )
    parser.add_argument(
        'output_file',
        nargs='?',
        default='data/2025-09-05_optimized.csv',
        help='Output CSV file (default: data/2025-09-05_optimized.csv)'
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run optimization
    optimize_csv(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
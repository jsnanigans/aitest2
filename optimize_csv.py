#!/usr/bin/env python3
"""
CSV Optimizer - Pre-processes large CSV files for efficient streaming.
Groups data by user_id and sorts by timestamp to enable proper chronological processing.
This ensures the Kalman filter processes each user's data in the correct time order.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
from collections import defaultdict
import heapq

def parse_datetime(dt_string):
    """Parse datetime string to datetime object for sorting."""
    try:
        return datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
    except:
        try:
            return datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S')
        except:
            try:
                return datetime.strptime(dt_string.split('.')[0], '%Y-%m-%dT%H:%M:%S')
            except:
                # If all else fails, try just the date part
                return datetime.strptime(dt_string[:10], '%Y-%m-%d')

def detect_date_column(fieldnames):
    """Detect the date column name from fieldnames."""
    for col in ['effectivDateTime', 'effectiveDateTime', 'date', 'timestamp']:
        if col in fieldnames:
            return col
    return None

def optimize_csv(input_file, output_file=None, chunk_size=100000):
    """
    Optimize CSV file by grouping data by user_id and sorting chronologically.
    Uses external sorting for memory efficiency with large files.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (default: input_file_optimized.csv)
        chunk_size: Number of rows to process in memory at once
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)
    
    if output_file is None:
        output_file = input_path.parent / f"{input_path.stem}_optimized{input_path.suffix}"
    
    print(f"Optimizing {input_file}")
    print(f"Output will be saved to: {output_file}")
    
    temp_dir = Path(tempfile.mkdtemp())
    temp_files = []
    
    try:
        print("Phase 1: Reading and sorting chunks...")
        with open(input_file, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            if not fieldnames:
                print("Error: Unable to read CSV headers")
                sys.exit(1)
            
            # Detect the date column name
            date_column = detect_date_column(fieldnames)
            if not date_column:
                print("Error: Could not find date column (tried: effectivDateTime, effectiveDateTime, date, timestamp)")
                sys.exit(1)
            
            print(f"  Using date column: {date_column}")
            print(f"  Columns found: {', '.join(fieldnames)}")
            
            chunk = []
            chunk_count = 0
            row_count = 0
            
            for row in reader:
                chunk.append(row)
                row_count += 1
                
                if len(chunk) >= chunk_size:
                    chunk_count += 1
                    temp_file = temp_dir / f"chunk_{chunk_count:06d}.csv"
                    
                    # Sort by user_id first, then by date
                    chunk.sort(key=lambda r: (r['user_id'], parse_datetime(r[date_column])))
                    
                    with open(temp_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(chunk)
                    
                    temp_files.append(temp_file)
                    print(f"  Processed chunk {chunk_count} ({row_count:,} rows total)")
                    chunk = []
            
            # Process remaining rows
            if chunk:
                chunk_count += 1
                temp_file = temp_dir / f"chunk_{chunk_count:06d}.csv"
                
                chunk.sort(key=lambda r: (r['user_id'], parse_datetime(r[date_column])))
                
                with open(temp_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(chunk)
                
                temp_files.append(temp_file)
                print(f"  Processed chunk {chunk_count} ({row_count:,} rows total)")
        
        print(f"\nPhase 2: Merging {len(temp_files)} sorted chunks...")
        
        # Open all temp files for merging
        file_readers = []
        for temp_file in temp_files:
            f = open(temp_file, 'r', encoding='utf-8')
            reader = csv.DictReader(f)
            file_readers.append((f, reader))
        
        # Initialize heap with first row from each file
        heap = []
        for i, (f, reader) in enumerate(file_readers):
            try:
                row = next(reader)
                # Use the same date column we detected earlier
                heapq.heappush(heap, (
                    (row['user_id'], parse_datetime(row[date_column])),
                    i,
                    row
                ))
            except StopIteration:
                pass
        
        # Merge all files
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            rows_written = 0
            last_user = None
            user_count = 0
            
            while heap:
                sort_key, file_idx, row = heapq.heappop(heap)
                writer.writerow(row)
                rows_written += 1
                
                if row['user_id'] != last_user:
                    last_user = row['user_id']
                    user_count += 1
                    if user_count % 1000 == 0:
                        print(f"  Merged {user_count:,} users, {rows_written:,} rows")
                
                # Try to get next row from the same file
                try:
                    f, reader = file_readers[file_idx]
                    next_row = next(reader)
                    heapq.heappush(heap, (
                        (next_row['user_id'], parse_datetime(next_row[date_column])),
                        file_idx,
                        next_row
                    ))
                except StopIteration:
                    pass
        
        # Close all temp files
        for f, _ in file_readers:
            f.close()
        
        print(f"\nOptimization complete!")
        print(f"  Total rows: {rows_written:,}")
        print(f"  Total users: {user_count:,}")
        print(f"  Output saved to: {output_file}")
        
        file_size = Path(output_file).stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size:.2f} MB")
        
    finally:
        # Clean up temp files
        shutil.rmtree(temp_dir, ignore_errors=True)

def validate_optimized_csv(csv_file):
    """
    Validate that a CSV file is properly optimized (sorted by user_id and date).
    
    Args:
        csv_file: Path to CSV file to validate
    
    Returns:
        True if file is properly sorted, False otherwise
    """
    print(f"\nValidating {csv_file}...")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        # Detect date column
        date_column = detect_date_column(fieldnames)
        if not date_column:
            print("Error: Could not find date column")
            return False
        
        print(f"  Using date column: {date_column}")
        
        prev_user = None
        prev_datetime = None
        row_count = 0
        user_count = 0
        errors = 0
        
        for row in reader:
            row_count += 1
            user_id = row['user_id']
            dt = parse_datetime(row[date_column])
            
            if user_id != prev_user:
                # New user - reset datetime tracking
                user_count += 1
                prev_user = user_id
                prev_datetime = dt
            else:
                # Same user - check chronological order
                if prev_datetime and dt < prev_datetime:
                    errors += 1
                    if errors <= 5:  # Show first 5 errors
                        print(f"  ERROR at row {row_count}: User {user_id[:8]}... has out-of-order dates")
                        print(f"    Previous: {prev_datetime}")
                        print(f"    Current:  {dt}")
                prev_datetime = dt
            
            if row_count % 100000 == 0:
                print(f"  Validated {row_count:,} rows, {user_count:,} users")
        
        print(f"  Validation complete: {row_count:,} rows, {user_count:,} users")
        if errors == 0:
            print(f"  ✓ File is properly optimized")
            return True
        else:
            print(f"  ✗ Found {errors} out-of-order timestamps")
            return False

def get_csv_stats(csv_file):
    """
    Get statistics about a CSV file.
    
    Args:
        csv_file: Path to CSV file
    """
    print(f"\nAnalyzing {csv_file}...")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        # Detect date column
        date_column = detect_date_column(fieldnames)
        if not date_column:
            print("Warning: Could not find date column")
            date_column = None
        else:
            print(f"  Using date column: {date_column}")
        
        users = set()
        user_readings = defaultdict(int)
        row_count = 0
        min_date = None
        max_date = None
        
        for row in reader:
            row_count += 1
            user_id = row['user_id']
            users.add(user_id)
            user_readings[user_id] += 1
            
            if date_column:
                try:
                    dt = parse_datetime(row[date_column])
                    if min_date is None or dt < min_date:
                        min_date = dt
                    if max_date is None or dt > max_date:
                        max_date = dt
                except:
                    pass
            
            if row_count % 100000 == 0:
                print(f"  Processed {row_count:,} rows...")
        
        # Calculate statistics
        readings_list = list(user_readings.values())
        readings_list.sort()
        
        print(f"\nStatistics:")
        print(f"  Total rows: {row_count:,}")
        print(f"  Total users: {len(users):,}")
        print(f"  Average readings per user: {row_count/len(users):.1f}")
        print(f"  Median readings per user: {readings_list[len(readings_list)//2]}")
        print(f"  Min readings per user: {readings_list[0]}")
        print(f"  Max readings per user: {readings_list[-1]}")
        
        # Show distribution of readings
        print(f"\n  Reading distribution:")
        thresholds = [1, 5, 10, 20, 50, 100, 200, 500]
        for threshold in thresholds:
            count = sum(1 for r in readings_list if r >= threshold)
            pct = 100 * count / len(readings_list)
            print(f"    >= {threshold:3d} readings: {count:,} users ({pct:.1f}%)")
        
        if min_date and max_date:
            print(f"\n  Date range: {min_date.date()} to {max_date.date()}")
        
        file_size = Path(csv_file).stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size:.2f} MB")

def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Optimize CSV files for efficient chronological processing'
    )
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument(
        '-o', '--output',
        help='Output CSV file path (default: input_optimized.csv)'
    )
    parser.add_argument(
        '-c', '--chunk-size',
        type=int,
        default=100000,
        help='Number of rows to process in memory at once (default: 100000)'
    )
    parser.add_argument(
        '-v', '--validate',
        action='store_true',
        help='Validate the output file after optimization'
    )
    parser.add_argument(
        '-s', '--stats',
        action='store_true',
        help='Show statistics about the input file'
    )
    parser.add_argument(
        '--validate-only',
        help='Only validate an existing optimized file'
    )
    parser.add_argument(
        '--stats-only',
        help='Only show statistics for a file'
    )
    
    args = parser.parse_args()
    
    if args.validate_only:
        validate_optimized_csv(args.validate_only)
    elif args.stats_only:
        get_csv_stats(args.stats_only)
    else:
        if args.stats:
            get_csv_stats(args.input_file)
        
        optimize_csv(args.input_file, args.output, args.chunk_size)
        
        if args.validate and args.output:
            validate_optimized_csv(args.output)
        elif args.validate:
            output_path = Path(args.input_file)
            output_file = output_path.parent / f"{output_path.stem}_optimized{output_path.suffix}"
            validate_optimized_csv(output_file)

if __name__ == '__main__':
    main()
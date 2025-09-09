#!/usr/bin/env python3
"""
CSV Optimizer - Pre-processes large CSV files for efficient streaming.
Sorts data by user_id and effectivDateTime to enable sequential processing.
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
            return datetime.strptime(dt_string.split('.')[0], '%Y-%m-%dT%H:%M:%S')

def optimize_csv(input_file, output_file=None, chunk_size=100000):
    """
    Optimize CSV file by sorting data by user_id and effectivDateTime.
    Uses external sorting for memory efficiency with large files.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (default: input_file.optimized.csv)
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
            
            chunk = []
            chunk_count = 0
            row_count = 0
            
            for row in reader:
                chunk.append(row)
                row_count += 1
                
                if len(chunk) >= chunk_size:
                    chunk_count += 1
                    temp_file = temp_dir / f"chunk_{chunk_count:06d}.csv"
                    
                    chunk.sort(key=lambda r: (r['user_id'], parse_datetime(r['effectivDateTime'])))
                    
                    with open(temp_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(chunk)
                    
                    temp_files.append(temp_file)
                    print(f"  Processed chunk {chunk_count} ({row_count:,} rows total)")
                    chunk = []
            
            if chunk:
                chunk_count += 1
                temp_file = temp_dir / f"chunk_{chunk_count:06d}.csv"
                
                chunk.sort(key=lambda r: (r['user_id'], parse_datetime(r['effectivDateTime'])))
                
                with open(temp_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(chunk)
                
                temp_files.append(temp_file)
                print(f"  Processed chunk {chunk_count} ({row_count:,} rows total)")
        
        print(f"\nPhase 2: Merging {len(temp_files)} sorted chunks...")
        
        file_readers = []
        for temp_file in temp_files:
            f = open(temp_file, 'r', encoding='utf-8')
            reader = csv.DictReader(f)
            file_readers.append((f, reader))
        
        heap = []
        for i, (f, reader) in enumerate(file_readers):
            try:
                row = next(reader)
                heapq.heappush(heap, (
                    (row['user_id'], parse_datetime(row['effectivDateTime'])),
                    i,
                    row
                ))
            except StopIteration:
                pass
        
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
                
                try:
                    f, reader = file_readers[file_idx]
                    next_row = next(reader)
                    heapq.heappush(heap, (
                        (next_row['user_id'], parse_datetime(next_row['effectivDateTime'])),
                        file_idx,
                        next_row
                    ))
                except StopIteration:
                    pass
        
        for f, _ in file_readers:
            f.close()
        
        print(f"\nOptimization complete!")
        print(f"  Total rows: {rows_written:,}")
        print(f"  Total users: {user_count:,}")
        print(f"  Output saved to: {output_file}")
        
        file_size = Path(output_file).stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size:.2f} MB")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def validate_optimized_csv(csv_file):
    """
    Validate that a CSV file is properly optimized (sorted by user_id and effectivDateTime).
    
    Args:
        csv_file: Path to CSV file to validate
    
    Returns:
        True if file is properly sorted, False otherwise
    """
    print(f"\nValidating {csv_file}...")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        prev_user = None
        prev_datetime = None
        row_count = 0
        user_count = 0
        
        for row in reader:
            row_count += 1
            user_id = row['user_id']
            dt = parse_datetime(row['effectivDateTime'])
            
            if user_id != prev_user:
                user_count += 1
                prev_user = user_id
                prev_datetime = dt
            else:
                if prev_datetime and dt < prev_datetime:
                    print(f"  ERROR: Data not sorted at row {row_count}")
                    print(f"    User: {user_id}")
                    print(f"    Previous datetime: {prev_datetime}")
                    print(f"    Current datetime: {dt}")
                    return False
                prev_datetime = dt
            
            if row_count % 100000 == 0:
                print(f"  Validated {row_count:,} rows, {user_count:,} users")
        
        print(f"  Validation complete: {row_count:,} rows, {user_count:,} users")
        print(f"  ✓ File is properly optimized")
        return True

def get_csv_stats(csv_file):
    """
    Get statistics about a CSV file.
    
    Args:
        csv_file: Path to CSV file
    """
    print(f"\nAnalyzing {csv_file}...")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        users = set()
        row_count = 0
        min_date = None
        max_date = None
        
        for row in reader:
            row_count += 1
            users.add(row['user_id'])
            
            dt = parse_datetime(row['effectivDateTime'])
            if min_date is None or dt < min_date:
                min_date = dt
            if max_date is None or dt > max_date:
                max_date = dt
            
            if row_count % 100000 == 0:
                print(f"  Processed {row_count:,} rows...")
        
        print(f"\nStatistics:")
        print(f"  Total rows: {row_count:,}")
        print(f"  Total users: {len(users):,}")
        print(f"  Average readings per user: {row_count/len(users):.1f}")
        if min_date and max_date:
            print(f"  Date range: {min_date.date()} to {max_date.date()}")
        
        file_size = Path(csv_file).stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size:.2f} MB")

def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Optimize CSV files for efficient streaming processing'
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
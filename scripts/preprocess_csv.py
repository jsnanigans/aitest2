#!/usr/bin/env python3

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

CONFIG = {
    'ignore_sources': [
        'test-source',
        'manual-entry',
    ],
    'min_datetime': '2010-01-01',
    'max_datetime': '2025-09-05',
    'max_buffer_size': 10000,
    'output_suffix': '_optimized',
    'verbose': True,
    'skip_duplicates': True,
    'date_formats': [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y/%m/%d %H:%M:%S',
        '%m/%d/%Y %H:%M:%S',
        '%d/%m/%Y %H:%M:%S',
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
    ]
}


def parse_datetime(date_str: str, formats: List[str] = None) -> Optional[datetime]:
    if not date_str or date_str.upper() == 'NULL':
        return None

    formats = formats or CONFIG['date_formats']
    date_str = date_str.strip()

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    try:
        from dateutil import parser
        return parser.parse(date_str)
    except:
        pass

    return None


def should_filter_source(source: str, ignore_list: List[str]) -> bool:
    if not ignore_list:
        return False

    source_lower = source.lower().strip()
    return any(ignored.lower().strip() in source_lower for ignored in ignore_list)


def should_filter_datetime(dt: Optional[datetime], min_dt: Optional[datetime], max_dt: Optional[datetime]) -> bool:
    if dt is None:
        return True

    if min_dt and dt < min_dt:
        return True

    if max_dt and dt > max_dt:
        return True

    return False


def process_user_buffer(user_data: List[Dict], skip_duplicates: bool = True) -> List[Dict]:
    if not user_data:
        return []

    user_data.sort(key=lambda x: x['parsed_datetime'])

    if not skip_duplicates:
        return user_data

    cleaned = []
    seen = set()

    for row in user_data:
        key = (
            row['effectiveDateTime'],
            row['weight'],
            row['source_type'],
            row.get('unit', 'kg')
        )

        if key not in seen:
            seen.add(key)
            cleaned.append(row)

    return cleaned


def write_user_data(writer: csv.DictWriter, user_data: List[Dict], fieldnames: List[str]) -> int:
    rows_written = 0

    for row in user_data:
        output_row = {field: row.get(field, '') for field in fieldnames}
        writer.writerow(output_row)
        rows_written += 1

    return rows_written


def preprocess_csv(
    input_path: str,
    output_path: Optional[str] = None,
    config: Dict = None
) -> Dict:
    config = config or CONFIG

    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        stem = input_file.stem
        suffix = config.get('output_suffix', '_optimized')
        output_path = input_file.parent / f"{stem}{suffix}.csv"
    else:
        output_path = Path(output_path)

    ignore_sources = config.get('ignore_sources', [])
    max_buffer_size = config.get('max_buffer_size', 10000)
    verbose = config.get('verbose', True)
    skip_duplicates = config.get('skip_duplicates', True)

    min_dt = None
    max_dt = None

    if config.get('min_datetime'):
        min_dt = parse_datetime(config['min_datetime'])
        if min_dt and verbose:
            print(f"Filtering dates >= {min_dt}")

    if config.get('max_datetime'):
        max_dt = parse_datetime(config['max_datetime'])
        if max_dt and verbose:
            print(f"Filtering dates <= {max_dt}")

    if ignore_sources and verbose:
        print(f"Ignoring sources: {', '.join(ignore_sources)}")

    stats = {
        'total_rows': 0,
        'filtered_rows': 0,
        'filtered_by_source': 0,
        'filtered_by_date': 0,
        'filtered_by_invalid': 0,
        'output_rows': 0,
        'users_processed': 0,
        'duplicates_removed': 0,
        'source_types': defaultdict(int),
        'date_range': {'min': None, 'max': None},
    }

    user_buffers: Dict[str, List[Dict]] = {}
    completed_users: Set[str] = set()
    fieldnames = None

    if verbose:
        print(f"\nProcessing: {input_path}")
        print(f"Output: {output_path}\n")

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        if not fieldnames:
            raise ValueError("CSV file has no headers")

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row_num, row in enumerate(reader, 1):
            stats['total_rows'] += 1

            if verbose and stats['total_rows'] % 10000 == 0:
                print(f"Processed {stats['total_rows']:,} rows, "
                      f"{len(completed_users):,} users completed...")

            user_id = row.get('user_id', '').strip()
            if not user_id:
                stats['filtered_rows'] += 1
                stats['filtered_by_invalid'] += 1
                continue

            weight_str = row.get('weight', '').strip()
            if not weight_str or weight_str.upper() == 'NULL':
                stats['filtered_rows'] += 1
                stats['filtered_by_invalid'] += 1
                continue

            try:
                weight = float(weight_str)
                if weight <= 0 or weight > 1000:
                    stats['filtered_rows'] += 1
                    stats['filtered_by_invalid'] += 1
                    continue
            except ValueError:
                stats['filtered_rows'] += 1
                stats['filtered_by_invalid'] += 1
                continue

            source_type = row.get('source_type', '').strip()
            if should_filter_source(source_type, ignore_sources):
                stats['filtered_rows'] += 1
                stats['filtered_by_source'] += 1
                continue

            date_str = row.get('effectiveDateTime', '').strip()
            parsed_dt = parse_datetime(date_str)

            if parsed_dt is None:
                stats['filtered_rows'] += 1
                stats['filtered_by_invalid'] += 1
                if verbose and stats['filtered_by_invalid'] <= 10:
                    print(f"Warning: Could not parse date '{date_str}' at row {row_num}")
                continue

            if should_filter_datetime(parsed_dt, min_dt, max_dt):
                stats['filtered_rows'] += 1
                stats['filtered_by_date'] += 1
                continue

            if stats['date_range']['min'] is None or parsed_dt < stats['date_range']['min']:
                stats['date_range']['min'] = parsed_dt
            if stats['date_range']['max'] is None or parsed_dt > stats['date_range']['max']:
                stats['date_range']['max'] = parsed_dt

            stats['source_types'][source_type] += 1

            row['parsed_datetime'] = parsed_dt

            if user_id not in user_buffers:
                user_buffers[user_id] = []

            user_buffers[user_id].append(row)

            if len(user_buffers[user_id]) >= max_buffer_size:
                processed = process_user_buffer(user_buffers[user_id], skip_duplicates)
                before_count = len(user_buffers[user_id])
                after_count = len(processed)
                stats['duplicates_removed'] += (before_count - after_count)

                rows_written = write_user_data(writer, processed, fieldnames)
                stats['output_rows'] += rows_written

                completed_users.add(user_id)
                del user_buffers[user_id]

        if verbose:
            print(f"\nFlushing remaining {len(user_buffers)} users...")

        for user_id in sorted(user_buffers.keys()):
            processed = process_user_buffer(user_buffers[user_id], skip_duplicates)
            before_count = len(user_buffers[user_id])
            after_count = len(processed)
            stats['duplicates_removed'] += (before_count - after_count)

            rows_written = write_user_data(writer, processed, fieldnames)
            stats['output_rows'] += rows_written
            completed_users.add(user_id)

    stats['users_processed'] = len(completed_users)

    return stats


def print_summary(stats: Dict, verbose: bool = True):
    if not verbose:
        return

    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)

    print(f"\nRows processed: {stats['total_rows']:,}")
    print(f"Rows filtered:  {stats['filtered_rows']:,}")
    print(f"  - Invalid data:    {stats['filtered_by_invalid']:,}")
    print(f"  - Source filtered: {stats['filtered_by_source']:,}")
    print(f"  - Date filtered:   {stats['filtered_by_date']:,}")

    if stats['duplicates_removed'] > 0:
        print(f"Duplicates removed: {stats['duplicates_removed']:,}")

    print(f"\nRows output:    {stats['output_rows']:,}")
    print(f"Users processed: {stats['users_processed']:,}")

    if stats['output_rows'] > 0:
        avg_per_user = stats['output_rows'] / stats['users_processed']
        print(f"Avg rows/user:   {avg_per_user:.1f}")

    if stats['date_range']['min'] and stats['date_range']['max']:
        print(f"\nDate range in output:")
        print(f"  From: {stats['date_range']['min']}")
        print(f"  To:   {stats['date_range']['max']}")

    if stats['source_types']:
        print(f"\nSource types in output:")
        for source, count in sorted(stats['source_types'].items(),
                                   key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {source:40s} {count:8,}")

        if len(stats['source_types']) > 10:
            print(f"  ... and {len(stats['source_types']) - 10} more")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess CSV weight data for stream processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.csv
  %(prog)s data.csv -o processed.csv
  %(prog)s data.csv --ignore-source "test-source" --ignore-source "manual"
  %(prog)s data.csv --min-date "2020-01-01" --max-date "2025-12-31"
  %(prog)s data.csv --no-verbose --keep-duplicates
        """
    )

    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('-o', '--output', help='Output CSV file path (default: input_optimized.csv)')

    parser.add_argument('--ignore-source', action='append', dest='ignore_sources',
                       help='Source type to ignore (can be used multiple times)')

    parser.add_argument('--min-date', help='Minimum date to include (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--max-date', help='Maximum date to include (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')

    parser.add_argument('--max-buffer', type=int, default=10000,
                       help='Maximum records to buffer per user (default: 10000)')

    parser.add_argument('--keep-duplicates', action='store_true',
                       help='Keep duplicate measurements (default: remove)')

    parser.add_argument('--no-verbose', action='store_true',
                       help='Suppress progress output')

    parser.add_argument('--output-suffix', default='_optimized',
                       help='Suffix for output filename when not specified (default: _optimized)')

    args = parser.parse_args()

    config = CONFIG.copy()

    if args.ignore_sources:
        config['ignore_sources'] = args.ignore_sources

    if args.min_date:
        config['min_datetime'] = args.min_date

    if args.max_date:
        config['max_datetime'] = args.max_date

    config['max_buffer_size'] = args.max_buffer
    config['skip_duplicates'] = not args.keep_duplicates
    config['verbose'] = not args.no_verbose
    config['output_suffix'] = args.output_suffix

    try:
        stats = preprocess_csv(args.input_file, args.output, config)
        print_summary(stats, config['verbose'])

        if config['verbose']:
            output_path = args.output or Path(args.input_file).stem + config['output_suffix'] + '.csv'
            print(f"\nOutput written to: {output_path}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

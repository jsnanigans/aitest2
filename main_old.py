#!/usr/bin/env python3
"""
Simplified Weight Stream Processor
Streams CSV data through Kalman filter, then visualizes results
"""

import argparse
import csv
import json
import sys
import tomllib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.reprocessor import WeightReprocessor
from src.database import ProcessorStateDB, get_state_db
from src.processor import process_weight_enhanced, DataQualityPreprocessor
from src.visualization import create_dashboard


def load_config(config_path: str = "config.toml") -> dict:
    """Load configuration from TOML file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def process_daily_cleanup(day, user_measurements, config, db, user_debug_logs):
    """
    Process daily cleanup for users with multiple measurements.

    Args:
        day: The date to process
        user_measurements: Dict of user_id -> list of measurements
        config: Configuration dict
        db: Database instance
        user_debug_logs: Debug logs dictionary to update

    Returns:
        Stats about the cleanup
    """
    stats = {
        'day': day,
        'users_checked': 0,
        'reprocessed_users': 0,
        'total_rejected': 0
    }

    for user_id, measurements in user_measurements.items():
        stats['users_checked'] += 1

        if len(measurements) > 1:
            # Multiple measurements for this user on this day
            result = WeightReprocessor.process_daily_batch(
                user_id=user_id,
                batch_date=day,
                measurements=measurements,
                processing_config=config["processing"],
                kalman_config=config["kalman"],
                db=db
            )

            if result.get('rejected'):
                stats['reprocessed_users'] += 1
                stats['total_rejected'] += len(result['rejected'])

                # Log cleanup event
                user_debug_logs[user_id]['cleanup_events'].append({
                    'date': str(day),
                    'type': 'daily_cleanup',
                    'measurements_processed': len(measurements),
                    'accepted': len(result.get('accepted', [])),
                    'rejected': len(result.get('rejected', [])),
                    'rejected_items': result.get('rejected', [])
                })

    return stats


def stream_process(csv_path: str, output_dir: str, config: dict):
    """
    Stream process weight data from CSV file.

    True streaming: processes each user's data as it arrives,
    keeping only current state. Stores results for visualization.
    Runs daily cleanup when day changes are detected.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    processors = {}
    results = defaultdict(list)
    seen_users = set()
    skipped_users = set()
    processed_users = set()
    insufficient_data_users = set()  # Users skipped due to min_readings
    user_reading_counts = defaultdict(int)  # Track reading count per user

    # Parse date filters from config
    min_date_str = config["data"].get("min_date", "")
    max_date_str = config["data"].get("max_date", "")
    min_date = None
    max_date = None

    if min_date_str:
        try:
            min_date = parse_timestamp(min_date_str)
        except:
            print(f"Warning: Invalid min_date '{min_date_str}', ignoring date filter")
            min_date = None

    if max_date_str:
        try:
            max_date = parse_timestamp(max_date_str)
        except:
            print(f"Warning: Invalid max_date '{max_date_str}', ignoring date filter")
            max_date = None

    # Detailed debug data for each user
    user_debug_logs = defaultdict(lambda: {
        'user_id': None,
        'first_seen': None,
        'last_seen': None,
        'total_measurements': 0,
        'accepted_count': 0,
        'rejected_count': 0,

        'state_resets': 0,
        'cleanup_events': [],
        'processing_logs': [],
        'state_snapshots': [],
        'adapted_parameters': None,
        'kalman_initialized': False,
        'initialization_timestamp': None,
        'raw_measurements': [],
        'processed_results': [],
        'rejection_reasons': defaultdict(int),
        'source_distribution': defaultdict(int),
        'daily_stats': defaultdict(lambda: {'accepted': 0, 'rejected': 0, 'total': 0})
    })

    # Track daily data for cleanup
    current_day = None
    daily_measurements = defaultdict(lambda: defaultdict(list))
    db = get_state_db()

    stats = {
        "total_rows": 0,
        "total_users": 0,
        "skipped_users": 0,
        "processed_users": 0,
        "accepted": 0,
        "rejected": 0,
        "date_filtered": 0,
        "start_time": datetime.now(),
    }

    max_users = config["data"].get("max_users", 0)
    user_offset = config["data"].get("user_offset", 0)
    test_users = config["data"].get("test_users", [])
    min_readings = config["data"].get("min_readings", 0)

    test_mode = bool(test_users)
    if test_mode:
        test_users_set = set(test_users)
        max_users = 0
        user_offset = 0

    # If min_readings is set, do a first pass to count readings per user
    if min_readings > 0 and not test_mode:
        print(f"Counting readings per user (min_readings={min_readings})...")
        if min_date or max_date:
            date_range_str = []
            if min_date:
                date_range_str.append(f"after {min_date_str}")
            if max_date:
                date_range_str.append(f"before {max_date_str}")
            print(f"  Date filter: {' and '.join(date_range_str)}")

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_id = row.get("user_id")
                if not user_id:
                    continue

                weight_str = row.get("weight", "").strip()
                if not weight_str or weight_str.upper() == "NULL":
                    continue

                # Parse and check date
                date_str = row.get("effectiveDateTime")
                try:
                    timestamp = parse_timestamp(date_str)
                except:
                    continue

                # Apply date filters
                if min_date and timestamp < min_date:
                    continue
                if max_date and timestamp > max_date:
                    continue

                try:
                    float(weight_str)
                    user_reading_counts[user_id] += 1
                except (ValueError, TypeError):
                    continue

        # Identify users with insufficient data
        for user_id, count in user_reading_counts.items():
            if count < min_readings:
                insufficient_data_users.add(user_id)

        if insufficient_data_users:
            print(f"  Found {len(insufficient_data_users)} users with < {min_readings} readings (will skip)")

    print(f"Processing {csv_path}...")
    if test_mode:
        print(f"  Test mode: Processing only {len(test_users)} specific users")
        for user_id in test_users[:5]:
            print(f"    - {user_id}")
        if len(test_users) > 5:
            print(f"    ... and {len(test_users) - 5} more")
    else:
        if user_offset > 0:
            print(f"  Skipping first {user_offset} users")
        if max_users > 0:
            print(f"  Processing up to {max_users} users after offset")
        if min_readings > 0:
            print(f"  Minimum readings required: {min_readings}")

    # Show date filter info
    if min_date or max_date:
        date_range_str = []
        if min_date:
            date_range_str.append(f"after {min_date_str}")
        if max_date:
            date_range_str.append(f"before {max_date_str}")
        print(f"  Date filter: Only processing measurements {' and '.join(date_range_str)}")

    print()

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        current_user = None
        last_progress_time = datetime.now()
        stop_after_user = None  # User to stop after processing all their data

        for row in reader:
            stats["total_rows"] += 1

            if stats["total_rows"] % config["logging"]["progress_interval"] == 0:
                elapsed = (datetime.now() - stats["start_time"]).total_seconds()
                rate = stats["total_rows"] / elapsed if elapsed > 0 else 0
                eta_str = ""

                if test_mode:
                    found_count = len(processed_users)
                    remaining = len(test_users) - found_count
                    if remaining > 0:
                        eta_str = f", {remaining} test users remaining"
                elif max_users > 0 and stats["processed_users"] > 0:
                    remaining = max_users - stats["processed_users"]
                    eta_str = f", ~{remaining} users remaining"

                print(
                    f"  Row {stats['total_rows']:,} | "
                    f"Users: {stats['processed_users']:,} processed"
                    f"{f', {stats["skipped_users"]:,} skipped' if stats['skipped_users'] > 0 else ''} | "
                    f"Rate: {rate:.0f} rows/sec{eta_str}"
                )

            user_id = row.get("user_id")
            if not user_id:
                continue

            # Skip users with insufficient data entirely (doesn't count towards max_users)
            if user_id in insufficient_data_users:
                continue

            # Check if we should stop after finishing the previous user
            if stop_after_user and user_id != stop_after_user:
                # We've finished processing stop_after_user, time to stop
                break

            if test_mode:
                if user_id not in test_users_set:
                    continue

                if user_id not in seen_users:
                    seen_users.add(user_id)
                    processed_users.add(user_id)
                    stats["processed_users"] = len(processed_users)
            else:
                if user_id not in seen_users:
                    seen_users.add(user_id)

                    if len(seen_users) <= user_offset:
                        skipped_users.add(user_id)
                        stats["skipped_users"] = len(skipped_users)
                        continue

                    # Check if we would exceed max_users
                    # We'll tentatively mark for processing but may stop later
                    if max_users > 0 and len(processed_users) >= max_users:
                        # If we've already found enough users with results, skip new users
                        continue

                if user_id in skipped_users:
                    continue

                # Skip users we haven't decided to process yet
                if max_users > 0 and user_id not in processed_users and len(processed_users) >= max_users:
                    continue

            weight_str = row.get("weight", "").strip()
            if not weight_str or weight_str.upper() == "NULL":
                continue

            date_str = row.get("effectiveDateTime")
            source = row.get("source_type") or row.get("source", "unknown")

            # Skip BSA measurements based on source
            if source and 'BSA' in source.upper():
                continue

            try:
                weight_raw = float(weight_str)
                unit = (row.get("unit") or "kg").lower().strip()

                # Skip BSA measurements based on unit
                if 'm2' in unit or 'm²' in unit or 'bsa' in unit:
                    continue

                # Skip values that are clearly BSA (typical range 1.5-2.5 m²)
                if weight_raw < 3.0 and source and 'BSA' not in source.upper():
                    # Could be BSA value even without proper labeling
                    # But only skip if we're not explicitly told it's weight
                    if 'weight' not in source.lower() and 'scale' not in source.lower():
                        continue

                # Convert pounds to kg
                if 'lb' in unit or 'pound' in unit:
                    weight = weight_raw * 0.453592
                # Handle pounds based on typical ranges (if unit missing but value > 130)
                elif weight_raw > 130 and weight_raw < 400 and not ('kg' in unit):
                    # Likely pounds based on typical human weight ranges
                    # This is a heuristic for data with missing/incorrect units
                    weight = weight_raw * 0.453592
                else:
                    weight = weight_raw

            except (ValueError, TypeError):
                continue

            try:
                timestamp = parse_timestamp(date_str)
            except:
                timestamp = datetime.now()

            # Apply date filters
            if min_date and timestamp < min_date:
                stats["date_filtered"] += 1
                continue
            if max_date and timestamp > max_date:
                stats["date_filtered"] += 1
                continue

            # Check for day change and run cleanup if needed
            measurement_day = timestamp.date()

            if current_day is not None and measurement_day != current_day:
                # Day has changed - process previous day's data
                if current_day in daily_measurements:
                    # print(f"\n  Day changed: Processing cleanup for {current_day}")
                    cleanup_stats = process_daily_cleanup(
                        current_day,
                        daily_measurements[current_day],
                        config,
                        db,
                        user_debug_logs
                    )

                    # Clear processed day from memory
                    del daily_measurements[current_day]

            current_day = measurement_day

            # Store measurement for potential cleanup
            daily_measurements[measurement_day][user_id].append({
                'weight': weight,
                'timestamp': timestamp,
                'source': source
            })

            # Initialize user debug log if first time
            if not user_debug_logs[user_id]['user_id']:
                user_debug_logs[user_id]['user_id'] = user_id
                user_debug_logs[user_id]['first_seen'] = str(timestamp)

            user_debug_logs[user_id]['last_seen'] = str(timestamp)
            user_debug_logs[user_id]['total_measurements'] += 1
            user_debug_logs[user_id]['source_distribution'][source] += 1

            # Log raw measurement
            user_debug_logs[user_id]['raw_measurements'].append({
                'timestamp': str(timestamp),
                'weight': weight,
                'source': source,
                'day': str(measurement_day)
            })

            # Load height data once for enhanced processing
            if not hasattr(DataQualityPreprocessor, '_height_data_loaded') or not DataQualityPreprocessor._height_data_loaded:
                DataQualityPreprocessor.load_height_data()

            # Always use enhanced processor with BMI detection and data quality improvements
            # Pass full config in processing_config so adaptive noise settings are available
            processing_config_with_full = config["processing"].copy()
            processing_config_with_full["config"] = config  # Add full config for adaptive noise

            result = process_weight_enhanced(
                user_id=user_id,
                weight=weight,
                timestamp=timestamp,
                source=source,
                processing_config=processing_config_with_full,
                kalman_config=config["kalman"],
                unit=unit  # Pass unit for enhanced processing
            )

            # Log processing event - result is never None anymore
            processing_log = {
                'timestamp': str(timestamp),
                'weight': weight,
                'source': source,
                'result': 'accepted' if result['accepted'] else 'rejected',
                'filtered_weight': result.get('filtered_weight'),
                'confidence': result.get('confidence'),
                'trend': result.get('trend'),
                'rejection_reason': result.get('reason')
            }

            # Track first processed measurement
            if not user_debug_logs[user_id]['kalman_initialized'] and result.get('filtered_weight') is not None:
                user_debug_logs[user_id]['kalman_initialized'] = True
                user_debug_logs[user_id]['initialization_timestamp'] = str(timestamp)

            # Store processing log
            user_debug_logs[user_id]['processing_logs'].append(processing_log)

            # Store processed result
            user_debug_logs[user_id]['processed_results'].append(result)

            # Update daily stats
            day_str = str(measurement_day)
            user_debug_logs[user_id]['daily_stats'][day_str]['total'] += 1

            if result['accepted']:
                user_debug_logs[user_id]['daily_stats'][day_str]['accepted'] += 1
            else:
                user_debug_logs[user_id]['daily_stats'][day_str]['rejected'] += 1
                if result.get('reason'):
                    user_debug_logs[user_id]['rejection_reasons'][result['reason']] += 1

            if result and result['accepted']:
                stats["accepted"] += 1
                user_debug_logs[user_id]['accepted_count'] += 1
            elif result:
                stats["rejected"] += 1
                user_debug_logs[user_id]['rejected_count'] += 1

            if result:
                results[user_id].append(result)

                # Mark user as processed only when they have actual results
                if not test_mode and user_id not in processed_users:
                    processed_users.add(user_id)
                    stats["processed_users"] = len(processed_users)

                    # Check if we've reached the target number of users with results
                    if max_users > 0 and len(processed_users) >= max_users and not stop_after_user:
                        # Set a flag to stop after processing current user's remaining data
                        stop_after_user = user_id

    # Process final day's cleanup if there's data
    if current_day is not None and current_day in daily_measurements:
        print(f"\n  Processing final day cleanup for {current_day}")
        cleanup_stats = process_daily_cleanup(
            current_day,
            daily_measurements[current_day],
            config,
            db,
            user_debug_logs
        )



    elapsed = (datetime.now() - stats["start_time"]).total_seconds()

    print(f"\nProcessing Complete:")
    print(f"  Total rows processed: {stats['total_rows']:,}")
    if test_mode:
        print(f"  Test users found: {len(processed_users)} of {len(test_users)}")
        if len(processed_users) < len(test_users):
            missing = set(test_users) - processed_users
            print(f"  Missing test users: {', '.join(list(missing)[:3])}")
            if len(missing) > 3:
                print(f"    ... and {len(missing) - 3} more")
    else:
        print(f"  Unique users seen: {len(seen_users):,}")
        if insufficient_data_users:
            print(f"  Users skipped (< {min_readings} readings): {len(insufficient_data_users):,}")
        if stats["skipped_users"] > 0:
            print(f"  Users skipped (offset): {stats['skipped_users']:,}")
    print(f"  Users processed: {stats['processed_users']:,}")

    if stats.get("date_filtered", 0) > 0:
        print(f"  Measurements filtered by date: {stats['date_filtered']:,}")
    print(f"  Measurements accepted: {stats['accepted']:,}")
    print(f"  Measurements rejected: {stats['rejected']:,}")
    print(f"  Time: {elapsed:.1f}s ({stats['total_rows']/elapsed:.0f} rows/sec)")
    if stats["accepted"] + stats["rejected"] > 0:
        print(
            f"  Acceptance rate: {stats['accepted']/(stats['accepted']+stats['rejected']):.1%}"
        )

    timestamp = datetime.now().strftime(config["logging"]["timestamp_format"])

    results_file = output_path / f"results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump({"stats": stats, "users": dict(results)}, f, indent=2, default=str)

    print(f"\nResults saved to {results_file}")

    # Save detailed debug reports for each user
    debug_dir = output_path / f"debug_{timestamp}"
    debug_dir.mkdir(exist_ok=True)

    print(f"\nGenerating debug reports for {len(user_debug_logs)} users...")

    # Create summary debug report
    summary_debug = {
        'processing_summary': stats,
        'total_users': len(user_debug_logs),
        'users_with_rejections': sum(1 for log in user_debug_logs.values() if log['rejected_count'] > 0),
        'users_with_cleanups': sum(1 for log in user_debug_logs.values() if log['cleanup_events']),
        'total_cleanup_events': sum(len(log['cleanup_events']) for log in user_debug_logs.values()),
        'aggregate_rejection_reasons': {},
        'aggregate_source_distribution': {}
    }

    # Aggregate rejection reasons and sources across all users
    for user_id, log in user_debug_logs.items():
        for reason, count in log['rejection_reasons'].items():
            summary_debug['aggregate_rejection_reasons'][reason] = \
                summary_debug['aggregate_rejection_reasons'].get(reason, 0) + count
        for source, count in log['source_distribution'].items():
            summary_debug['aggregate_source_distribution'][source] = \
                summary_debug['aggregate_source_distribution'].get(source, 0) + count

    # Save summary debug file
    summary_file = debug_dir / "debug_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_debug, f, indent=2, default=str)

    # Save individual user debug files
    for idx, (user_id, debug_log) in enumerate(user_debug_logs.items(), 1):
        if idx % 100 == 0:
            print(f"  [{idx}/{len(user_debug_logs)}] Writing debug reports...")

        # Add computed summary stats to each user's debug log
        debug_log['summary'] = {
            'acceptance_rate': debug_log['accepted_count'] / debug_log['total_measurements']
                if debug_log['total_measurements'] > 0 else 0,
            'days_active': len(debug_log['daily_stats']),
            'had_cleanup': len(debug_log['cleanup_events']) > 0,
            'unique_sources': len(debug_log['source_distribution']),
            'unique_rejection_reasons': len(debug_log['rejection_reasons'])
        }

        # Get final state from database
        final_state = db.get_state(user_id)
        if final_state:
            debug_log['final_state'] = {
                'has_kalman_params': final_state.get('kalman_params') is not None,
                'has_adapted_params': 'adapted_params' in final_state,
                'last_timestamp': str(final_state.get('last_timestamp', ''))
            }
            if final_state.get('last_state') is not None:
                try:
                    # Handle numpy arrays
                    last_state = final_state['last_state']
                    debug_log['final_state']['last_weight'] = float(last_state[0])
                    debug_log['final_state']['last_trend'] = float(last_state[1])
                except (TypeError, IndexError):
                    # Fallback if conversion fails
                    pass

        user_debug_file = debug_dir / f"user_{user_id}.json"
        with open(user_debug_file, "w") as f:
            json.dump(debug_log, f, indent=2, default=str)

    print(f"Debug reports saved to {debug_dir}")

    # Export database to CSV if enabled
    if config.get("data", {}).get("export_database", True):
        db_dump_file = output_path / f"db_dump_{timestamp}.csv"
        users_exported = db.export_to_csv(str(db_dump_file))
        print(f"\nDatabase dump saved to {db_dump_file} ({users_exported} users)")

    # Generate visualizations if enabled
    if config.get("visualization", {}).get("enabled", True):
        print("\nGenerating visualizations...")
        viz_dir = output_path / f"viz_{timestamp}"
        viz_dir.mkdir(exist_ok=True)

        total_users = len(results)
        viz_count = 0

        for idx, (user_id, user_results) in enumerate(results.items(), 1):
            percentage = (idx / total_users) * 100
            print(f"  [{idx}/{total_users}] {percentage:5.1f}% - Creating dashboard for user {user_id[:8]}...", end="", flush=True)
            try:
                # No longer pass raw_data - only use processed results
                create_dashboard(
                    user_id, user_results, str(viz_dir), config["visualization"]
                )
                viz_count += 1
                print(" ✓")
            except Exception as e:
                print(f" ✗ ({e})")

        print(f"\nVisualization Summary:")
        print(f"  Total users processed: {total_users}")
        print(f"  Visualizations created: {viz_count} in {viz_dir}")

    return results, stats


def parse_timestamp(date_str: str) -> datetime:
    """Parse various timestamp formats."""
    if not date_str:
        return datetime.now()

    if "T" in date_str:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    elif " " in date_str:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    else:
        return datetime.strptime(date_str, "%Y-%m-%d")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Weight Stream Processor",
        epilog="""
Examples:
  # Process all users in file
  %(prog)s data/weights.csv

  # Process specific users only
  %(prog)s data/weights.csv --users user123 user456

  # Process first 100 users
  %(prog)s data/weights.csv --limit 100

  # Skip first 50 users, then process next 20
  %(prog)s data/weights.csv --offset 50 --max-users 20

  # Only process users with at least 10 readings
  %(prog)s data/weights.csv --min-readings 10 --limit 100

  # Process without generating visualizations
  %(prog)s data/weights.csv --no-viz

  # Use custom output directory
  %(prog)s data/weights.csv --output results/custom
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv_file", nargs="?", help="CSV file to process")
    parser.add_argument("--users", nargs="+", help="Specific user IDs to process")
    parser.add_argument("--max-users", type=int, help="Maximum number of users to process")
    parser.add_argument("--offset", type=int, help="Number of users to skip before processing")
    parser.add_argument("--min-readings", type=int, help="Minimum readings required per user")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--config", default="config.toml", help="Configuration file (default: config.toml)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    parser.add_argument("--limit", type=int, help="Alias for --max-users")
    parser.add_argument("--dump-db", action="store_true", help="Export database to CSV and exit")
    parser.add_argument("--db-output", help="Output path for database dump CSV")
    parser.add_argument("--no-db-dump", action="store_true", help="Skip database dump after processing")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Handle standalone database dump
    if args.dump_db:
        from src.database import get_state_db
        db = get_state_db()

        if args.db_output:
            output_file = args.db_output
        else:
            timestamp = datetime.now().strftime(config["logging"]["timestamp_format"])
            output_dir = Path(config["data"]["output_dir"])
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"db_dump_{timestamp}.csv"

        users_exported = db.export_to_csv(str(output_file))
        print(f"Database dump saved to {output_file} ({users_exported} users)")
        sys.exit(0)

    # Override config with CLI arguments
    csv_file = args.csv_file or config["data"]["csv_file"]

    if args.users:
        config["data"]["test_users"] = args.users
        config["data"]["max_users"] = 0  # Disable max_users in test mode
        config["data"]["user_offset"] = 0  # Disable offset in test mode
    else:
        if args.max_users is not None:
            config["data"]["max_users"] = args.max_users
        elif args.limit is not None:  # Support --limit as alias
            config["data"]["max_users"] = args.limit

        if args.offset is not None:
            config["data"]["user_offset"] = args.offset

        if args.min_readings is not None:
            config["data"]["min_readings"] = args.min_readings

    if args.output:
        config["data"]["output_dir"] = args.output

    if args.no_viz:
        config["visualization"]["enabled"] = False

    if args.no_db_dump:
        config.setdefault("data", {})["export_database"] = False

    if not Path(csv_file).exists():
        print(f"Error: File {csv_file} not found")
        sys.exit(1)

    stream_process(csv_file, config["data"]["output_dir"], config)

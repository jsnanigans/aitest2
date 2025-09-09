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

from processor import WeightProcessor
from reprocessor import WeightReprocessor
from processor_database import ProcessorStateDB, get_state_db
from visualization import create_dashboard


def load_config(config_path: str = "config.toml") -> dict:
    """Load configuration from TOML file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def process_daily_cleanup(day, user_measurements, config, db):
    """
    Process daily cleanup for users with multiple measurements.

    Args:
        day: The date to process
        user_measurements: Dict of user_id -> list of measurements
        config: Configuration dict
        db: Database instance

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

    # Track daily data for cleanup
    current_day = None
    daily_measurements = defaultdict(lambda: defaultdict(list))
    db = ProcessorStateDB()

    stats = {
        "total_rows": 0,
        "total_users": 0,
        "skipped_users": 0,
        "processed_users": 0,
        "accepted": 0,
        "rejected": 0,
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
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_id = row.get("user_id")
                if not user_id:
                    continue
                
                weight_str = row.get("weight", "").strip()
                if not weight_str or weight_str.upper() == "NULL":
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
    print()

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        current_user = None
        last_progress_time = datetime.now()

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

                    if max_users > 0 and len(processed_users) >= max_users:
                        continue

                    processed_users.add(user_id)
                    stats["processed_users"] = len(processed_users)

                if user_id in skipped_users:
                    continue

                if max_users > 0 and user_id not in processed_users:
                    continue

            weight_str = row.get("weight", "").strip()
            if not weight_str or weight_str.upper() == "NULL":
                continue

            try:
                weight = float(weight_str)
            except (ValueError, TypeError):
                continue

            date_str = row.get("effectivDateTime") or row.get("effectiveDateTime") or row.get("date")
            source = row.get("source_type") or row.get("source", "unknown")

            try:
                timestamp = parse_timestamp(date_str)
            except:
                timestamp = datetime.now()

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
                        db
                    )
                    if cleanup_stats['reprocessed_users']:
                        print(f"    Cleaned up {cleanup_stats['reprocessed_users']} users, "
                              f"rejected {cleanup_stats['total_rejected']} bad measurements")

                    # Clear processed day from memory
                    del daily_measurements[current_day]

            current_day = measurement_day

            # Store measurement for potential cleanup
            daily_measurements[measurement_day][user_id].append({
                'weight': weight,
                'timestamp': timestamp,
                'source': source
            })

            # Use stateless processor - no need to create instances
            result = WeightProcessor.process_weight(
                user_id=user_id,
                weight=weight,
                timestamp=timestamp,
                source=source,
                processing_config=config["processing"],
                kalman_config=config["kalman"],
                db=db
            )

            # Track unique users
            if user_id not in processors:
                processors[user_id] = True
                stats["total_users"] = len(processors)

            if result:
                # Store ALL processed results (both accepted and rejected)
                if result["accepted"]:
                    stats["accepted"] += 1
                else:
                    stats["rejected"] += 1

                results[user_id].append(result)
                
                # Don't store raw measurements - we only want processed results
                # raw_measurements are no longer needed

            if (
                not test_mode
                and max_users > 0
                and len(processed_users) >= max_users
                and all(
                    uid in processed_users or uid in skipped_users for uid in seen_users
                )
            ):
                print(f"\n  Reached max_users limit ({max_users}), stopping...")
                break

    # Process final day's cleanup if there's data
    if current_day is not None and current_day in daily_measurements:
        print(f"\n  Processing final day cleanup for {current_day}")
        cleanup_stats = process_daily_cleanup(
            current_day,
            daily_measurements[current_day],
            config,
            db
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
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
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
    
    if not Path(csv_file).exists():
        print(f"Error: File {csv_file} not found")
        sys.exit(1)

    stream_process(csv_file, config["data"]["output_dir"], config)

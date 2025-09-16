#!/usr/bin/env python3
"""
Simplified Weight Stream Processor
Streams CSV data through Kalman filter with minimal state management
"""

import argparse
import csv
import json
import sys
import tomllib
from datetime import datetime
from pathlib import Path
import math

from src.database.database import get_state_db
from src.processing.processor import process_measurement
from src.processing.validation import DataQualityPreprocessor

from src.viz.visualization import create_weight_timeline


def load_config(config_path: str = "config.toml") -> dict:
    """Load configuration with profile interpretation."""
    if not Path(config_path).exists():
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {
            "data": {
                "csv_file": "./data/weights.csv",
                "output_dir": "output",
                "max_users": 200,
                "min_readings": 20
            },
            "processing": {
                "extreme_threshold": 0.15
            },
            "kalman": {},
            "visualization": {
                "enabled": True
            }
        }

    # Use the new config loader that interprets profiles
    from src.config_loader import load_config as load_config_with_profiles
    config = load_config_with_profiles(config_path)

    # Print which profile is being used
    with open(config_path, "rb") as f:
        raw_config = tomllib.load(f)
        profile = raw_config.get("profile", "balanced")
        print(f"Using profile: {profile}")

    return config


def load_test_users(filepath: str = "test_users.txt") -> list:
    """Load test user IDs from file."""
    if not Path(filepath).exists():
        return []

    users = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                users.append(line)
    return users


def stream_process(csv_path: str, output_dir: str, config: dict):
    """
    Simplified stream processing - no complex state accumulation.
    Process each measurement and stream results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Set verbosity from config
    from src.utils import set_verbosity
    viz_config = config.get('visualization', {})
    verbosity_str = viz_config.get('verbosity', 'normal')
    verbosity_map = {
        'silent': 0,
        'minimal': 1,
        'normal': 2,
        'verbose': 3
    }
    verbosity_level = verbosity_map.get(verbosity_str, 2)
    set_verbosity(verbosity_level)

    # Load height data once
    DataQualityPreprocessor.load_height_data()

    # Database for Kalman state
    db = get_state_db()

    # Initialize retrospective processing components
    retro_config = config.get("retrospective", {})
    retro_enabled = retro_config.get("enabled", False)
    retro_buffer = None
    outlier_detector = None
    replay_manager = None

    if retro_enabled:
        try:
            from src.retro_buffer import get_retro_buffer
            from src.processing.outlier_detection import OutlierDetector
            from src.replay.replay_manager import ReplayManager

            retro_buffer = get_retro_buffer(retro_config)
            outlier_detector = OutlierDetector(retro_config.get("outlier_detection", {}), db=db)
            replay_manager = ReplayManager(db, retro_config.get("safety", {}))

            print("Retrospective processing enabled")
            print(f"  Buffer window: {retro_config.get('buffer_hours', 72)} hours")
            print(f"  Trigger mode: {retro_config.get('trigger_mode', 'time_based')}")
        except ImportError as e:
            print(f"Warning: Could not initialize retrospective processing: {e}")
            retro_enabled = False

    # Parse configuration
    max_users = config["data"].get("max_users", 0)
    user_offset = config["data"].get("user_offset", 0)
    min_readings = config["data"].get("min_readings", 0)

    # Load test users if specified
    test_users_file = config["data"].get("test_users_file", "")
    test_users = load_test_users(test_users_file) if test_users_file else []
    test_mode = bool(test_users)

    if test_mode:
        test_users_set = set(test_users)
        max_users = len(test_users)
        user_offset = 0
        print(f"Test mode: Processing {len(test_users)} specific users")

    # Date filters
    min_date_str = config["data"].get("min_date", "")
    max_date_str = config["data"].get("max_date", "")
    min_date = parse_timestamp(min_date_str) if min_date_str else None
    max_date = parse_timestamp(max_date_str) if max_date_str else None

    # Tracking
    seen_users = set()
    processed_users = set()
    skipped_users = set()
    user_results = {}  # Only store results, not debug info

    stats = {
        "total_rows": 0,
        "accepted": 0,
        "rejected": 0,
        "date_filtered": 0,
        "start_time": datetime.now(),
    }

    print(f"Processing {csv_path}...")
    if max_users > 0:
        print(f"  Processing up to {max_users} users")
    if min_date or max_date:
        print(f"  Date filter: {min_date_str} to {max_date_str}")
    print()

    # First pass: count readings per user (always do this to properly filter)
    user_reading_counts = {}
    eligible_users = []  # Users that meet min_readings criteria

    if not test_mode:
        print(f"Analyzing user data (min_readings={min_readings}, max_users={max_users})...")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_id = row.get("user_id")
                if not user_id:
                    continue

                # Basic validation
                weight_str = row.get("weight", "").strip()
                if not weight_str or weight_str.upper() == "NULL":
                    continue

                # Date check
                if min_date or max_date:
                    date_str = row.get("effectiveDateTime")
                    try:
                        timestamp = parse_timestamp(date_str)
                        if min_date and timestamp < min_date:
                            continue
                        if max_date and timestamp > max_date:
                            continue
                    except:
                        continue

                user_reading_counts[user_id] = user_reading_counts.get(user_id, 0) + 1

        # Determine eligible users based on min_readings
        for user_id, count in sorted(user_reading_counts.items()):  # Sort for consistent ordering
            if count >= min_readings:
                eligible_users.append(user_id)

        # Apply user_offset and max_users to eligible users
        if user_offset > 0 and user_offset < len(eligible_users):
            eligible_users = eligible_users[user_offset:]
        elif user_offset >= len(eligible_users):
            eligible_users = []  # Offset beyond available users

        if max_users > 0 and len(eligible_users) > max_users:
            eligible_users = eligible_users[:max_users]

        eligible_users_set = set(eligible_users)

        print(f"  Found {len(user_reading_counts)} total users")
        print(f"  {len([u for u, c in user_reading_counts.items() if c >= min_readings])} users with >= {min_readings} readings")
        print(f"  Processing {len(eligible_users)} users (after offset={user_offset}, max={max_users})")

    # Main processing loop
    with open(csv_path) as f:
        reader = csv.DictReader(f)

        for row in reader:
            stats["total_rows"] += 1

            # Progress update
            if stats["total_rows"] % config["logging"]["progress_interval"] == 0:
                elapsed = (datetime.now() - stats["start_time"]).total_seconds()
                rate = stats["total_rows"] / elapsed if elapsed > 0 else 0
                print(f"  Row {stats['total_rows']:,} | "
                      f"Users: {len(processed_users):,} | "
                      f"Rate: {rate:.0f} rows/sec")

            # Parse row
            user_id = row.get("user_id")
            if not user_id:
                continue

            # User filtering
            if test_mode:
                if user_id not in test_users_set:
                    continue
            else:
                # Only process users in the eligible set
                if user_id not in eligible_users_set:
                    continue

            # Parse weight with validation
            weight_str = row.get("weight", "").strip()
            if not weight_str or weight_str.upper() == "NULL":
                continue

            try:
                weight = float(weight_str)
                # Validate weight is reasonable
                if weight <= 0 or weight > 1000:  # Basic sanity check
                    stats["invalid_weight"] = stats.get("invalid_weight", 0) + 1
                    continue
                if math.isnan(weight) or math.isinf(weight):
                    stats["invalid_weight"] = stats.get("invalid_weight", 0) + 1
                    continue
            except (ValueError, TypeError):
                stats["parse_errors"] = stats.get("parse_errors", 0) + 1
                continue

            # Parse metadata
            date_str = row.get("effectiveDateTime")
            source = row.get("source_type") or row.get("source", "unknown")
            unit = (row.get("unit") or "kg").lower().strip()

            # Skip BSA measurements
            if 'BSA' in source.upper() or 'm2' in unit or 'm²' in unit:
                continue

            # Parse timestamp
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

            # Process measurement with error boundary
            try:
                # Merge quality scoring config into main config
                full_config = config.copy()
                if 'quality_scoring' not in full_config:
                    full_config['quality_scoring'] = {}

                result = process_measurement(
                    user_id=user_id,
                    weight=weight,
                    timestamp=timestamp,
                    source=source,
                    config=full_config,
                    unit=unit,
                    db=db
                )
            except Exception as e:
                stats["processing_errors"] = stats.get("processing_errors", 0) + 1
                if config.get("logging", {}).get("verbose", False):
                    print(f"Error processing {user_id}: {e}")
                continue

            # Track results
            if result:
                if user_id not in user_results:
                    user_results[user_id] = []
                    processed_users.add(user_id)

                user_results[user_id].append(result)

                if result.get('accepted'):
                    stats["accepted"] += 1
                else:
                    stats["rejected"] += 1

                # Add to retrospective buffer if enabled
                if retro_enabled and retro_buffer:
                    measurement_data = {
                        'weight': weight,
                        'timestamp': timestamp,
                        'source': source,
                        'unit': unit,
                        'metadata': {
                            'accepted': result.get('accepted', False),
                            'rejection_reason': result.get('rejection_reason', None),
                            'quality_score': result.get('quality_score', None),
                            'quality_components': result.get('quality_components', None)
                        }
                    }

                    buffer_result = retro_buffer.add_measurement(user_id, measurement_data)

                    # Check if buffer is ready for processing
                    if buffer_result.get('buffer_ready', False):
                        # Save state snapshot before buffer analysis
                        db.save_state_snapshot(user_id, timestamp)

                        # Process retrospectively in the background
                        # Note: In production, this might be async/queued
                        try:
                            _process_retrospective_buffer(
                                user_id=user_id,
                                retro_buffer=retro_buffer,
                                outlier_detector=outlier_detector,
                                replay_manager=replay_manager,
                                stats=stats
                            )
                        except Exception as e:
                            stats["retro_errors"] = stats.get("retro_errors", 0) + 1
                            if config.get("logging", {}).get("verbose", False):
                                print(f"Retrospective processing error for {user_id}: {e}")

    # Process any remaining retrospective buffers at the end
    if retro_enabled and retro_buffer:
        ready_buffers = retro_buffer.get_ready_buffers()
        if ready_buffers:
            print(f"\nProcessing {len(ready_buffers)} remaining retrospective buffers...")
            for user_id in ready_buffers:
                try:
                    # Save final snapshot
                    db.save_state_snapshot(user_id, datetime.now())
                    _process_retrospective_buffer(
                        user_id=user_id,
                        retro_buffer=retro_buffer,
                        outlier_detector=outlier_detector,
                        replay_manager=replay_manager,
                        stats=stats
                    )
                except Exception as e:
                    stats["retro_errors"] = stats.get("retro_errors", 0) + 1
                    print(f"Error processing final buffer for {user_id}: {e}")

    # Final statistics
    elapsed = (datetime.now() - stats["start_time"]).total_seconds()

    print(f"\nProcessing Complete:")
    print(f"  Total rows: {stats['total_rows']:,}")
    print(f"  Users processed: {len(processed_users):,}")
    if stats["date_filtered"] > 0:
        print(f"  Measurements filtered by date: {stats['date_filtered']:,}")
    print(f"  Measurements accepted: {stats['accepted']:,}")
    print(f"  Measurements rejected: {stats['rejected']:,}")
    print(f"  Time: {elapsed:.1f}s ({stats['total_rows']/elapsed:.0f} rows/sec)")

    if stats["accepted"] + stats["rejected"] > 0:
        acceptance_rate = stats['accepted'] / (stats['accepted'] + stats['rejected'])
        print(f"  Acceptance rate: {acceptance_rate:.1%}")

    # Retrospective processing statistics
    if retro_enabled and stats.get("retro_processed", 0) > 0:
        print(f"\nRetrospective Processing:")
        print(f"  Buffers processed: {stats.get('retro_processed', 0):,}")
        print(f"  Measurements analyzed: {stats.get('retro_measurements_analyzed', 0):,}")
        print(f"  Outliers found: {stats.get('retro_outliers_found', 0):,}")
        print(f"  Successful replays: {stats.get('retro_replays_successful', 0):,}")
        if stats.get("retro_replays_failed", 0) > 0:
            print(f"  Failed replays: {stats.get('retro_replays_failed', 0):,}")
        if stats.get("retro_errors", 0) > 0:
            print(f"  Processing errors: {stats.get('retro_errors', 0):,}")

    # Save results
    timestamp_str = datetime.now().strftime(config["logging"]["timestamp_format"])
    results_file = output_path / f"results_{timestamp_str}.json"

    with open(results_file, "w") as f:
        json.dump({
            "stats": stats,
            "users": user_results
        }, f, indent=2, default=str)

    print(f"\nResults saved to {results_file}")

    # Export database if requested
    if config["data"].get("export_database", True):
        db_file = output_path / f"db_dump_{timestamp_str}.csv"
        users_exported = db.export_to_csv(str(db_file))
        print(f"Database exported to {db_file} ({users_exported} users)")

    # Generate visualizations if enabled
    if config.get("visualization", {}).get("enabled", True):
        viz_dir = output_path / f"viz_{timestamp_str}"
        viz_dir.mkdir(exist_ok=True)

        total_users = len(user_results)
        if total_users > 0:
            print(f"\nGenerating visualizations for {total_users} user(s)...")

            # Set visualization verbosity based on number of users
            try:
                from src.utils import set_verbosity
                if total_users > 10:
                    set_verbosity(0)  # Silent for many users
                elif total_users > 1:
                    set_verbosity(1)  # Minimal for a few users
                else:
                    set_verbosity(2)  # Normal for single user
            except ImportError:
                pass

            successful = 0
            failed = 0

            for idx, (user_id, results) in enumerate(user_results.items(), 1):
                # Progress indicator for large batches
                if total_users > 10 and idx % 10 == 0:
                    print(f"  Progress: {idx}/{total_users} ({idx*100//total_users}%)")
                elif total_users <= 10:
                    print(f"  [{idx}/{total_users}] User {user_id[:8]}...", end=" ")

                try:
                    from src.viz.visualization import create_weight_timeline
                    dashboard_path = create_weight_timeline(
                        results, user_id, str(viz_dir), config=config
                    )
                    if dashboard_path:
                        successful += 1
                        if total_users <= 10:
                            print("✓")
                    else:
                        failed += 1
                        if total_users <= 10:
                            print("✗")
                except Exception as e:
                    failed += 1
                    if total_users <= 10:
                        print(f"✗ ({str(e)[:30]})")
                    elif config.get("logging", {}).get("verbose", False):
                        print(f"    Error for {user_id}: {e}")

            # Summary
            if total_users > 1:
                print(f"\nVisualization complete: {successful} successful, {failed} failed")

            # Generate index.html for dashboard navigation
            if successful > 0:
                try:
                    from src.viz.viz_index import create_index_from_results
                    index_path = create_index_from_results(
                        all_results=user_results,
                        output_dir=str(viz_dir)
                    )
                    print(f"Dashboard index: {index_path}")
                except Exception as e:
                    print(f"Warning: Could not generate index.html: {e}")

            print(f"Output: {viz_dir}")

    return user_results, stats


def _process_retrospective_buffer(user_id: str, retro_buffer, outlier_detector, replay_manager, stats: dict) -> None:
    """
    Process a retrospective buffer for outlier detection and replay.

    Args:
        user_id: User identifier
        retro_buffer: RetroBuffer instance
        outlier_detector: OutlierDetector instance
        replay_manager: ReplayManager instance
        stats: Statistics dictionary to update
    """
    try:
        # Get buffered measurements
        buffered_measurements = retro_buffer.get_buffer_measurements(user_id)
        if not buffered_measurements:
            return

        buffer_info = retro_buffer.get_buffer_info(user_id)
        buffer_start_time = buffer_info['first_timestamp'] if buffer_info else None

        # Detect outliers (now with quality score awareness and Kalman prediction)
        clean_measurements, outlier_indices = outlier_detector.get_clean_measurements(buffered_measurements, user_id=user_id)

        # Get analysis details
        analysis = outlier_detector.analyze_outliers(buffered_measurements, outlier_indices)

        # Update statistics
        stats["retro_processed"] = stats.get("retro_processed", 0) + 1
        stats["retro_outliers_found"] = stats.get("retro_outliers_found", 0) + len(outlier_indices)
        stats["retro_measurements_analyzed"] = stats.get("retro_measurements_analyzed", 0) + len(buffered_measurements)

        if len(outlier_indices) > 0:
            # Replay clean measurements if we have any and found outliers
            if clean_measurements and buffer_start_time:
                replay_result = replay_manager.replay_clean_measurements(
                    user_id=user_id,
                    clean_measurements=clean_measurements,
                    buffer_start_time=buffer_start_time
                )

                if replay_result['success']:
                    stats["retro_replays_successful"] = stats.get("retro_replays_successful", 0) + 1
                else:
                    stats["retro_replays_failed"] = stats.get("retro_replays_failed", 0) + 1
                    print(f"  Replay failed: {replay_result.get('error', 'unknown error')}")

        # Clear buffer after processing
        retro_buffer.clear_buffer(user_id)

    except Exception as e:
        import traceback
        stats["retro_processing_errors"] = stats.get("retro_processing_errors", 0) + 1
        print(f"Error in retrospective processing for {user_id}: {e}")
        traceback.print_exc()


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
        description="Simplified Weight Stream Processor"
    )
    parser.add_argument("csv_file", nargs="?", help="CSV file to process")
    parser.add_argument("--config", default="config.toml", help="Configuration file")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--max-users", type=int, help="Maximum users to process")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualizations")
    parser.add_argument("--no-db-export", action="store_true", help="Skip database export")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Validate configuration
    from src.utils import validate_config
    is_valid, errors = validate_config(config)
    if not is_valid:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    # Override with command line arguments
    if args.csv_file:
        config["data"]["csv_file"] = args.csv_file
    if args.output:
        config["data"]["output_dir"] = args.output
    if args.max_users:
        config["data"]["max_users"] = args.max_users
    if args.no_viz:
        config["visualization"]["enabled"] = False
    if args.no_db_export:
        config["data"]["export_database"] = False

    csv_file = config["data"]["csv_file"]
    if not Path(csv_file).exists():
        print(f"Error: File {csv_file} not found")
        sys.exit(1)

    stream_process(csv_file, config["data"]["output_dir"], config)

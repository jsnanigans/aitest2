#!/usr/bin/env python3
"""
Simplified Weight Stream Processor
Streams CSV data through Kalman filter, then visualizes results
"""

import csv
import json
import sys
import tomllib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from processor import WeightProcessor
from visualization import create_dashboard


def load_config(config_path: str = "config.toml") -> dict:
    """Load configuration from TOML file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def stream_process(csv_path: str, output_dir: str, config: dict):
    """
    Stream process weight data from CSV file.

    True streaming: processes each user's data as it arrives,
    keeping only current state. Stores results for visualization.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    processors = {}
    results = defaultdict(list)
    seen_users = set()
    skipped_users = set()
    processed_users = set()
    users_with_insufficient_init_data = set()

    stats = {
        "total_rows": 0,
        "total_users": 0,
        "skipped_users": 0,
        "processed_users": 0,
        "users_insufficient_init": 0,
        "accepted": 0,
        "rejected": 0,
        "start_time": datetime.now(),
    }

    max_users = config["data"].get("max_users", 0)
    user_offset = config["data"].get("user_offset", 0)
    test_users = config["data"].get("test_users", [])
    
    test_mode = bool(test_users)
    if test_mode:
        test_users_set = set(test_users)
        max_users = 0
        user_offset = 0

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

            if test_mode:
                if user_id not in test_users_set:
                    continue
                
                if user_id not in seen_users:
                    seen_users.add(user_id)
                    processed_users.add(user_id)
                    stats["processed_users"] = len(processed_users)
                    
                    if len(processed_users) == len(test_users):
                        print(f"\n  Found all {len(test_users)} test users, stopping early...")
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

            date_str = row.get("effectivDateTime") or row.get("date")
            source = row.get("source_type") or row.get("source", "unknown")

            try:
                timestamp = parse_timestamp(date_str)
            except:
                timestamp = datetime.now()

            # Use stateless processor - no need to create instances
            result = WeightProcessor.process_weight(
                user_id=user_id,
                weight=weight,
                timestamp=timestamp,
                source=source,
                processing_config=config["processing"],
                kalman_config=config["kalman"]
            )
            
            # Track unique users
            if user_id not in processors:
                processors[user_id] = True
                stats["total_users"] = len(processors)

            if result:
                if result["accepted"]:
                    stats["accepted"] += 1
                else:
                    stats["rejected"] += 1

                results[user_id].append(result)
            else:
                # Still buffering for initialization
                if user_id not in users_with_insufficient_init_data:
                    users_with_insufficient_init_data.add(user_id)

            if test_mode and len(processed_users) == len(test_users):
                break
            elif (
                not test_mode
                and max_users > 0
                and len(processed_users) >= max_users
                and all(
                    uid in processed_users or uid in skipped_users for uid in seen_users
                )
            ):
                print(f"\n  Reached max_users limit ({max_users}), stopping...")
                break

    # Check for users that never got enough data to initialize
    for user_id in users_with_insufficient_init_data:
        if user_id not in results or len(results[user_id]) == 0:
            stats["users_insufficient_init"] += 1

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
        if stats["skipped_users"] > 0:
            print(f"  Users skipped (offset): {stats['skipped_users']:,}")
    print(f"  Users processed: {stats['processed_users']:,}")
    if stats["users_insufficient_init"] > 0:
        print(f"  Users with insufficient init data (<{config['processing']['min_init_readings']} readings): {stats['users_insufficient_init']:,}")
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

    print("\nGenerating visualizations...")
    viz_dir = output_path / f"viz_{timestamp}"
    viz_dir.mkdir(exist_ok=True)

    min_data_points = config["visualization"].get("min_data_points", 10)
    max_visualizations = config["visualization"].get("max_visualizations", 50)
    
    viz_candidates = [
        (user_id, user_results)
        for user_id, user_results in results.items()
        if len(user_results) >= min_data_points
    ]
    
    users_insufficient_data = len(results) - len(viz_candidates)
    
    viz_count = 0
    for user_id, user_results in viz_candidates[:max_visualizations]:
        try:
            create_dashboard(
                user_id, user_results, str(viz_dir), config["visualization"]
            )
            viz_count += 1
        except Exception as e:
            print(f"  Failed to visualize {user_id}: {e}")

    print(f"\nVisualization Summary:")
    print(f"  Users with sufficient data (>={min_data_points} points): {len(viz_candidates)}")
    if users_insufficient_data > 0:
        print(f"  Users skipped (insufficient data): {users_insufficient_data}")
    if len(viz_candidates) > max_visualizations:
        print(f"  Users skipped (max limit reached): {len(viz_candidates) - max_visualizations}")
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
    config = load_config()

    csv_file = sys.argv[1] if len(sys.argv) > 1 else config["data"]["csv_file"]

    if not Path(csv_file).exists():
        print(f"Error: File {csv_file} not found")
        sys.exit(1)

    stream_process(csv_file, config["data"]["output_dir"], config)

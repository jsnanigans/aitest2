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

from src.database import get_state_db
from src.processor import process_measurement
from src.visualization import create_dashboard
from src.validation import DataQualityPreprocessor


def load_config(config_path: str = "config.toml") -> dict:
    """Load configuration from TOML file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


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
    
    # Load height data once
    DataQualityPreprocessor.load_height_data()
    
    # Database for Kalman state
    db = get_state_db()
    
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
    
    # First pass: count readings if min_readings is set
    user_reading_counts = {}
    if min_readings > 0 and not test_mode:
        print(f"Counting readings per user (min_readings={min_readings})...")
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
            if test_mode and user_id not in test_users_set:
                continue
            
            if not test_mode:
                # Check min_readings
                if min_readings > 0 and user_reading_counts.get(user_id, 0) < min_readings:
                    continue
                
                # Handle offset and max_users
                if user_id not in seen_users:
                    seen_users.add(user_id)
                    
                    if len(seen_users) <= user_offset:
                        skipped_users.add(user_id)
                        continue
                    
                    if max_users > 0 and len(processed_users) >= max_users:
                        break
                
                if user_id in skipped_users:
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
        print("\nGenerating visualizations...")
        viz_dir = output_path / f"viz_{timestamp_str}"
        viz_dir.mkdir(exist_ok=True)
        
        for idx, (user_id, results) in enumerate(user_results.items(), 1):
            if idx % 10 == 0:
                print(f"  [{idx}/{len(user_results)}] Visualizing...")
            
            try:
                create_dashboard(
                    user_id, results, str(viz_dir), config["visualization"]
                )
            except Exception as e:
                print(f"  Error visualizing {user_id}: {e}")
        
        print(f"Visualizations saved to {viz_dir}")
    
    return user_results, stats


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
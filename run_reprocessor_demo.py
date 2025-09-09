#!/usr/bin/env python3
"""
Historical Data Reprocessor
Handles retroactive additions/removals of weight data
This is separate from the main real-time processing flow
"""

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
import tomllib
from collections import defaultdict

from reprocessor import WeightReprocessor
from processor_database import ProcessorDatabase


def load_config(config_path: str = "config.toml") -> dict:
    """Load configuration from TOML file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def reprocess_historical_data(
    user_id: str,
    start_date: datetime,
    csv_path: str,
    config: dict,
    reason: str = "manual"
):
    """
    Reprocess historical data for a specific user from a given date.
    
    Args:
        user_id: User to reprocess
        start_date: Date to start reprocessing from
        csv_path: Path to CSV file with data
        config: Configuration dict
        reason: Reason for reprocessing
    """
    print(f"\nReprocessing Historical Data")
    print(f"User: {user_id}")
    print(f"Start Date: {start_date.date()}")
    print(f"Reason: {reason}")
    print("-" * 50)
    
    db = ProcessorDatabase()
    
    # Create snapshot before reprocessing
    snapshot_id = db.create_snapshot(user_id, datetime.now())
    print(f"Created snapshot: {snapshot_id}")
    
    # Load user's measurements from CSV
    measurements = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('user_id') != user_id:
                continue
                
            weight_str = row.get('weight', '').strip()
            if not weight_str or weight_str.upper() == 'NULL':
                continue
                
            try:
                weight = float(weight_str)
                date_str = row.get('effectivDateTime') or row.get('effectiveDateTime') or row.get('date')
                timestamp = parse_timestamp(date_str)
                source = row.get('source_type') or row.get('source', 'unknown')
                
                measurements.append({
                    'weight': weight,
                    'timestamp': timestamp,
                    'source': source
                })
            except:
                continue
    
    # Sort by timestamp
    measurements.sort(key=lambda x: x['timestamp'])
    
    print(f"Found {len(measurements)} total measurements for user")
    
    # Filter to measurements from start_date onward
    relevant_measurements = [
        m for m in measurements
        if m['timestamp'] >= start_date
    ]
    
    print(f"Reprocessing {len(relevant_measurements)} measurements from {start_date.date()}")
    
    # Perform reprocessing
    result = WeightReprocessor.reprocess_from_date(
        user_id=user_id,
        start_date=start_date,
        measurements=measurements,
        processing_config=config['processing'],
        kalman_config=config['kalman'],
        db=db
    )
    
    # Validate results
    is_valid, validation_msg = WeightReprocessor.validate_reprocessing(
        user_id, snapshot_id, result, db
    )
    
    if is_valid:
        print(f"\n✅ Reprocessing successful: {validation_msg}")
        print(f"  Measurements processed: {result['measurements_processed']}")
        print(f"  Successful: {result['successful']}")
        if result.get('errors'):
            print(f"  Errors: {len(result['errors'])}")
    else:
        print(f"\n❌ Reprocessing failed: {validation_msg}")
        print("Rolling back to snapshot...")
        db.restore_snapshot(user_id, snapshot_id)
        print("Rollback complete")
    
    return result


def detect_and_fix_problematic_days(csv_path: str, config: dict, max_users: int = 10):
    """
    Scan data for problematic days and fix them.
    
    Args:
        csv_path: Path to CSV file
        config: Configuration dict
        max_users: Maximum users to process
    """
    print("\nScanning for Problematic Days")
    print("=" * 50)
    
    db = ProcessorDatabase()
    
    # Group measurements by user and day
    user_day_data = defaultdict(lambda: defaultdict(list))
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        users_seen = set()
        
        for row in reader:
            user_id = row.get('user_id')
            if not user_id:
                continue
                
            # Limit to max_users for demo
            if user_id not in users_seen:
                if len(users_seen) >= max_users:
                    continue
                users_seen.add(user_id)
            elif user_id not in users_seen:
                continue
                
            weight_str = row.get('weight', '').strip()
            if not weight_str or weight_str.upper() == 'NULL':
                continue
                
            try:
                weight = float(weight_str)
                date_str = row.get('effectivDateTime') or row.get('effectiveDateTime') or row.get('date')
                timestamp = parse_timestamp(date_str)
                source = row.get('source_type') or row.get('source', 'unknown')
                
                day = timestamp.date()
                user_day_data[user_id][day].append({
                    'weight': weight,
                    'timestamp': timestamp,
                    'source': source
                })
            except:
                continue
    
    # Find problematic days
    problems_found = []
    
    for user_id, days in user_day_data.items():
        for day, measurements in days.items():
            if len(measurements) > 1:
                weights = [m['weight'] for m in measurements]
                weight_range = max(weights) - min(weights)
                
                if weight_range > 5.0:  # More than 5kg range in a day
                    problems_found.append({
                        'user_id': user_id,
                        'day': day,
                        'measurements': measurements,
                        'range': weight_range
                    })
    
    # Sort by severity (largest range first)
    problems_found.sort(key=lambda x: x['range'], reverse=True)
    
    print(f"Found {len(problems_found)} problematic days")
    
    # Process top problems
    for i, problem in enumerate(problems_found[:5]):
        print(f"\n{i+1}. User {problem['user_id'][:8]}... on {problem['day']}")
        print(f"   Range: {problem['range']:.1f}kg across {len(problem['measurements'])} measurements")
        
        # Use reprocessor to fix
        result = WeightReprocessor.process_daily_batch(
            user_id=problem['user_id'],
            batch_date=problem['day'],
            measurements=problem['measurements'],
            processing_config=config['processing'],
            kalman_config=config['kalman'],
            db=db
        )
        
        if result.get('selected'):
            print(f"   ✅ Selected: {result['selected'][0]['weight']:.1f}kg")
            if result.get('rejected'):
                rejected_weights = [m['weight'] for m in result['rejected']]
                print(f"   ❌ Rejected: {rejected_weights}")


def parse_timestamp(date_str: str) -> datetime:
    """Parse various timestamp formats."""
    if not date_str:
        return datetime.now()
    
    # Handle various formats
    if 'T' in date_str:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    elif ' ' in date_str:
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    else:
        return datetime.strptime(date_str, '%Y-%m-%d')


def main():
    """Main entry point for reprocessor utility."""
    config = load_config()
    
    if len(sys.argv) < 2:
        print("Historical Data Reprocessor")
        print("=" * 50)
        print("\nUsage:")
        print("  python run_reprocessor_demo.py scan [csv_file]")
        print("    - Scan for and fix problematic days")
        print("\n  python run_reprocessor_demo.py reprocess <user_id> <start_date> [csv_file]")
        print("    - Reprocess specific user from date")
        print("\nExamples:")
        print("  python run_reprocessor_demo.py scan")
        print("  python run_reprocessor_demo.py reprocess EB61194E491F40E49064D3D096E0FE64 2025-03-01")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "scan":
        csv_file = sys.argv[2] if len(sys.argv) > 2 else config["data"]["csv_file"]
        detect_and_fix_problematic_days(csv_file, config)
        
    elif command == "reprocess":
        if len(sys.argv) < 4:
            print("Error: reprocess requires user_id and start_date")
            sys.exit(1)
            
        user_id = sys.argv[2]
        start_date = datetime.strptime(sys.argv[3], '%Y-%m-%d')
        csv_file = sys.argv[4] if len(sys.argv) > 4 else config["data"]["csv_file"]
        
        reprocess_historical_data(
            user_id=user_id,
            start_date=start_date,
            csv_path=csv_file,
            config=config,
            reason="Manual reprocessing request"
        )
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
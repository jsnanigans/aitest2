#!/usr/bin/env python3
"""Test processing of specific user with gaps"""

import csv
import sys
import tomllib
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.processor import WeightProcessor
from src.processor_database import ProcessorStateDB

def main():
    target_user = "5e81abae-a19a-4f79-99fe-7165842bbf6a"
    csv_path = "./data/2025-09-05_optimized.csv"
    
    # Load config
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    
    # Create fresh database
    db = ProcessorStateDB()
    
    # Collect user's measurements
    measurements = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("user_id") == target_user:
                weight_str = row.get("weight", "").strip()
                if weight_str and weight_str.upper() != "NULL":
                    try:
                        weight = float(weight_str)
                        date_str = row.get("effectiveDateTime")
                        timestamp = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                        source = row.get("source_type") or row.get("source", "unknown")
                        measurements.append({
                            'weight': weight,
                            'timestamp': timestamp,
                            'source': source
                        })
                    except:
                        pass
    
    print(f"Found {len(measurements)} measurements for {target_user}")
    print()
    
    # Process each measurement
    results = []
    for i, m in enumerate(measurements):
        print(f"Processing measurement {i+1}/{len(measurements)}: {m['timestamp']} - {m['weight']}kg")
        
        result = WeightProcessor.process_weight(
            user_id=target_user,
            weight=m['weight'],
            timestamp=m['timestamp'],
            source=m['source'],
            processing_config=config["processing"],
            kalman_config=config["kalman"],
            db=db
        )
        
        if result:
            print(f"  Result: {'ACCEPTED' if result['accepted'] else 'REJECTED'}")
            print(f"  Filtered: {result.get('filtered_weight', 'N/A')}")
            print(f"  Reason: {result.get('reason', 'N/A')}")
            results.append(result)
        else:
            print(f"  Result: None (shouldn't happen anymore!)")
        
        # Check for gap
        if i > 0:
            gap_days = (m['timestamp'] - measurements[i-1]['timestamp']).days
            if gap_days > 30:
                print(f"  ** GAP DETECTED: {gap_days} days - state should reset **")
        
        print()
    
    print(f"\nSummary:")
    print(f"  Total processed: {len(results)}")
    print(f"  Accepted: {sum(1 for r in results if r and r['accepted'])}")
    print(f"  Rejected: {sum(1 for r in results if r and not r['accepted'])}")
    
    # Check final state
    final_state = db.get_state(target_user)
    if final_state:
        print(f"\nFinal state:")
        print(f"  Has Kalman params: {final_state.get('kalman_params') is not None}")
        print(f"  Last timestamp: {final_state.get('last_timestamp')}")
        if final_state.get('last_state') is not None:
            print(f"  Last weight: {final_state['last_state'][-1][0] if len(final_state['last_state'].shape) > 1 else final_state['last_state'][0]}")

if __name__ == "__main__":
    main()
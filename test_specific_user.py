#!/usr/bin/env python3
"""Debug specific user 1e87d3ab-20b1-479d-ad4d-8986e1af38da to understand volatile Kalman output"""

import csv
import sys
import tomllib
from datetime import datetime
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.processor import WeightProcessor
from src.processor_database import ProcessorStateDB

def main():
    target_user = "1e87d3ab-20b1-479d-ad4d-8986e1af38da"
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
    if measurements:
        print(f"Date range: {measurements[0]['timestamp'].date()} to {measurements[-1]['timestamp'].date()}")
    print("\nProcessing first 30 measurements to analyze volatility:")
    print("-" * 90)
    
    # Process each measurement
    results = []
    for i, m in enumerate(measurements[:30]):
        # Get state before processing
        state_before = db.get_state(target_user)
        
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
            results.append(result)
            
            # Check if accepted (has filtered weight)
            if result.get('accepted', False) and 'filtered_weight' in result:
                # Check for volatility
                if i > 0 and len(results) > 1:
                    # Find previous accepted result
                    prev_result = None
                    for r in reversed(results[:-1]):
                        if r.get('accepted', False) and 'filtered_weight' in r:
                            prev_result = r
                            break
                    
                    if prev_result:
                        prev_filtered = prev_result['filtered_weight']
                        curr_filtered = result['filtered_weight']
                        change = curr_filtered - prev_filtered
                        change_pct = abs(change) / prev_filtered * 100
                        
                        # Calculate gap
                        gap_days = (m['timestamp'] - measurements[i-1]['timestamp']).days
                        
                        # Get current state to check Kalman params
                        state_after = db.get_state(target_user)
                        obs_cov = state_after['kalman_params']['observation_covariance'][0][0] if state_after else None
                        
                        print(f"{i+1:3}. {m['timestamp'].date()} | Raw: {m['weight']:6.1f} | Filtered: {curr_filtered:6.1f} | "
                              f"Δ: {change:+5.2f} ({change_pct:4.1f}%) | Gap: {gap_days:3}d | ObsCov: {obs_cov:.2f}")
                        
                        # Flag large changes
                        if change_pct > 2:
                            print(f"     ^^^ VOLATILE! {change_pct:.1f}% change in filtered output")
                            
                            # Check state details
                            if state_after:
                                last_state = state_after.get('last_state')
                                if last_state is not None:
                                    if len(last_state.shape) > 1:
                                        current = last_state[-1]
                                    else:
                                        current = last_state
                                    print(f"     State: weight={current[0]:.1f}, trend={current[1]:.4f}")
                    else:
                        print(f"{i+1:3}. {m['timestamp'].date()} | Raw: {m['weight']:6.1f} | Filtered: {result['filtered_weight']:6.1f} | (First accepted)")
                else:
                    print(f"{i+1:3}. {m['timestamp'].date()} | Raw: {m['weight']:6.1f} | Filtered: {result['filtered_weight']:6.1f} | (First)")
            else:
                # Rejected
                print(f"{i+1:3}. {m['timestamp'].date()} | Raw: {m['weight']:6.1f} | REJECTED - {result.get('reason', 'unknown')}")
        else:
            print(f"{i+1:3}. {m['timestamp'].date()} | Raw: {m['weight']:6.1f} | No result")
    
    print("\n" + "=" * 90)
    print("Summary of volatility issues:")
    print("-" * 90)
    
    # Analyze patterns
    if results:
        filtered_weights = [r['filtered_weight'] for r in results if r.get('accepted', False) and 'filtered_weight' in r]
        raw_weights = [r['raw_weight'] for r in results if 'raw_weight' in r]
        
        # Calculate volatility metrics
        if len(filtered_weights) > 1:
            filtered_changes = np.diff(filtered_weights)
            raw_changes = np.diff(raw_weights)
            
            print(f"Raw weight stats:")
            print(f"  Mean: {np.mean(raw_weights):.1f} kg")
            print(f"  Std:  {np.std(raw_weights):.2f} kg")
            print(f"  Range: {np.min(raw_weights):.1f} - {np.max(raw_weights):.1f} kg")
            
            print(f"\nFiltered weight stats:")
            print(f"  Mean: {np.mean(filtered_weights):.1f} kg")
            print(f"  Std:  {np.std(filtered_weights):.2f} kg")
            print(f"  Range: {np.min(filtered_weights):.1f} - {np.max(filtered_weights):.1f} kg")
            
            print(f"\nChange statistics:")
            print(f"  Raw changes std:      {np.std(raw_changes):.3f} kg")
            print(f"  Filtered changes std: {np.std(filtered_changes):.3f} kg")
            print(f"  Max filtered change:  {np.max(np.abs(filtered_changes)):.2f} kg")
            
            # Check if filter is actually smoothing or amplifying
            if np.std(filtered_changes) > np.std(raw_changes):
                print("\n⚠️  WARNING: Kalman filter is AMPLIFYING noise, not smoothing!")
                print(f"   Filtered volatility is {np.std(filtered_changes)/np.std(raw_changes):.1f}x higher than raw")
    
    # Check final state
    final_state = db.get_state(target_user)
    if final_state:
        print(f"\nFinal Kalman parameters:")
        kalman_params = final_state.get('kalman_params', {})
        print(f"  Observation covariance: {kalman_params.get('observation_covariance', 'N/A')}")
        print(f"  Transition covariance:  {kalman_params.get('transition_covariance', 'N/A')}")
        print(f"  Initial state mean:     {kalman_params.get('initial_state_mean', 'N/A')}")

if __name__ == "__main__":
    main()
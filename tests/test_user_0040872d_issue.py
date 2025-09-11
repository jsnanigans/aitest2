#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.processor import WeightProcessor
from src.processor_database import ProcessorDatabase

def test_user_0040872d_issue():
    user_id = '0040872d-333a-4ace-8c5a-b2fcd056e65a'
    csv_file = 'data/2025-09-05_optimized.csv'
    
    print("=== Testing User 0040872d Weight Drop Issue ===\n")
    
    df = pd.read_csv(csv_file)
    df.columns = ['user_id', 'timestamp', 'source', 'weight_kg', 'unit']
    
    user_data = df[df['user_id'] == user_id].copy()
    user_data['timestamp'] = pd.to_datetime(user_data['timestamp'])
    user_data = user_data.sort_values('timestamp')
    
    db = ProcessorDatabase()
    processor = WeightProcessor()
    
    accepted_count = 0
    rejected_count = 0
    rejection_reasons = {}
    processed_weights = []
    
    print("Processing measurements chronologically...\n")
    
    for idx, row in user_data.iterrows():
        timestamp = row['timestamp']
        weight = row['weight_kg']
        source = row['source']
        
        state = db.get_user_state(user_id)
        
        if state is None:
            result = processor.process_measurement(
                weight_kg=weight,
                timestamp=timestamp,
                user_state=None,
                source_type=source
            )
        else:
            result = processor.process_measurement(
                weight_kg=weight,
                timestamp=timestamp,
                user_state=state,
                source_type=source
            )
        
        if result['accepted']:
            accepted_count += 1
            db.save_user_state(user_id, result['state'])
            processed_weights.append({
                'timestamp': timestamp,
                'weight': weight,
                'accepted_weight': result['accepted_weight'],
                'source': source,
                'status': 'accepted'
            })
            
            if accepted_count <= 10 or (weight < 60 and accepted_count <= 20):
                print(f"ACCEPTED: {timestamp} - {weight:.1f} kg → {result['accepted_weight']:.1f} kg (source: {source})")
                if result.get('reset_occurred'):
                    print(f"  *** RESET OCCURRED ***")
        else:
            rejected_count += 1
            reason = result.get('rejection_reason', 'Unknown')
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            processed_weights.append({
                'timestamp': timestamp,
                'weight': weight,
                'accepted_weight': None,
                'source': source,
                'status': f'rejected ({reason})'
            })
    
    print(f"\n=== Processing Summary ===")
    print(f"Total measurements: {len(user_data)}")
    print(f"Accepted: {accepted_count}")
    print(f"Rejected: {rejected_count}")
    print(f"\nRejection reasons:")
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count}")
    
    print("\n=== Analyzing Dramatic Drops ===")
    
    processed_df = pd.DataFrame(processed_weights)
    accepted_df = processed_df[processed_df['status'] == 'accepted'].copy()
    
    if not accepted_df.empty:
        accepted_df['weight_diff'] = accepted_df['accepted_weight'].diff()
        large_drops = accepted_df[accepted_df['weight_diff'] < -10]
        
        print(f"\nFound {len(large_drops)} accepted measurements with drops > 10kg:")
        for _, row in large_drops.head(10).iterrows():
            print(f"  {row['timestamp']}: {row['weight']:.1f} kg (drop: {row['weight_diff']:.1f} kg)")
    
    print("\n=== Finding the Critical Drop (87-90kg → ~52kg) ===")
    
    stable_measurements = accepted_df[(accepted_df['accepted_weight'] >= 85) & 
                                      (accepted_df['accepted_weight'] <= 92)]
    low_measurements = accepted_df[(accepted_df['accepted_weight'] >= 45) & 
                                   (accepted_df['accepted_weight'] <= 55)]
    
    if not stable_measurements.empty and not low_measurements.empty:
        last_stable = stable_measurements.iloc[-1]
        first_low = low_measurements.iloc[0]
        
        print(f"\nLast stable (85-92kg): {last_stable['timestamp']} - {last_stable['accepted_weight']:.1f} kg")
        print(f"First low (45-55kg): {first_low['timestamp']} - {first_low['accepted_weight']:.1f} kg")
        
        time_diff = (first_low['timestamp'] - last_stable['timestamp']).total_seconds() / (24*3600)
        weight_diff = first_low['accepted_weight'] - last_stable['accepted_weight']
        pct_change = (weight_diff / last_stable['accepted_weight']) * 100
        
        print(f"\nTime between: {time_diff:.1f} days")
        print(f"Weight change: {weight_diff:.1f} kg ({pct_change:.1f}%)")
        
        print(f"\n=== Why did Kalman accept this drop? ===")
        print("Possible reasons:")
        print("1. Large time gap caused uncertainty to increase")
        print("2. Reset occurred due to gap > 30 days")
        print("3. Kalman filter's process noise allowed for large changes")
        print("4. No physiological limits were enforced")

if __name__ == "__main__":
    test_user_0040872d_issue()
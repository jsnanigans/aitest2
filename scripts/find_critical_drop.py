#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def find_critical_drop():
    user_id = '0040872d-333a-4ace-8c5a-b2fcd056e65a'
    csv_file = 'data/2025-09-05_optimized.csv'
    
    df = pd.read_csv(csv_file)
    df.columns = ['user_id', 'timestamp', 'source', 'weight_kg', 'unit']
    
    user_data = df[df['user_id'] == user_id].copy()
    user_data['timestamp'] = pd.to_datetime(user_data['timestamp'])
    user_data = user_data.sort_values('timestamp')
    
    print("=== Looking for transition from ~87-90kg to ~52kg ===\n")
    
    stable_high = user_data[(user_data['weight_kg'] >= 85) & (user_data['weight_kg'] <= 92)]
    print(f"Found {len(stable_high)} measurements in 85-92kg range")
    print("\nLast few stable measurements around 87-90kg:")
    print(stable_high.tail(10)[['timestamp', 'weight_kg', 'source']])
    
    dropped_low = user_data[(user_data['weight_kg'] >= 50) & (user_data['weight_kg'] <= 55)]
    print(f"\n\nFound {len(dropped_low)} measurements in 50-55kg range")
    print("\nFirst few measurements around 52kg:")
    print(dropped_low.head(10)[['timestamp', 'weight_kg', 'source']])
    
    print("\n=== Timeline around the drop ===")
    
    if not stable_high.empty:
        last_stable_date = stable_high['timestamp'].max()
        print(f"\nLast stable measurement (85-92kg): {last_stable_date}")
        
        transition_period = user_data[(user_data['timestamp'] >= last_stable_date - timedelta(days=7)) & 
                                      (user_data['timestamp'] <= last_stable_date + timedelta(days=30))]
        
        print(f"\nMeasurements from 1 week before to 1 month after last stable:")
        print(transition_period[['timestamp', 'weight_kg', 'source']])
    
    print("\n=== Checking for gaps in measurements ===")
    
    user_data['days_since_last'] = user_data['timestamp'].diff().dt.total_seconds() / (24*3600)
    gaps = user_data[user_data['days_since_last'] > 7]
    
    if not gaps.empty:
        print(f"\nFound {len(gaps)} gaps > 7 days:")
        for idx, row in gaps.iterrows():
            prev_idx = user_data.index[user_data.index.get_loc(idx) - 1]
            prev = user_data.loc[prev_idx]
            print(f"\n  Gap: {row['days_since_last']:.1f} days")
            print(f"    Before: {prev['weight_kg']:.1f} kg at {prev['timestamp']}")
            print(f"    After:  {row['weight_kg']:.1f} kg at {row['timestamp']}")
            print(f"    Change: {row['weight_kg'] - prev['weight_kg']:.1f} kg")

if __name__ == "__main__":
    find_critical_drop()
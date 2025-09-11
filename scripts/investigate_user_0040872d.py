#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

def investigate_user(user_id='0040872d-333a-4ace-8c5a-b2fcd056e65a'):
    csv_file = 'data/2025-09-05_optimized.csv'
    print(f"\n=== Investigating User {user_id} Weight Drop Issue ===\n")
    
    df = pd.read_csv(csv_file)
    
    df.columns = ['user_id', 'timestamp', 'source', 'weight_kg', 'unit']
    
    user_data = df[df['user_id'] == user_id].copy()
    
    if user_data.empty:
        print(f"No data found for user {user_id}")
        return
    
    user_data['timestamp'] = pd.to_datetime(user_data['timestamp'])
    user_data = user_data.sort_values('timestamp')
    
    print(f"Total measurements: {len(user_data)}")
    print(f"Date range: {user_data['timestamp'].min()} to {user_data['timestamp'].max()}")
    print(f"Weight range: {user_data['weight_kg'].min():.1f} to {user_data['weight_kg'].max():.1f} kg")
    
    print("\n=== Finding the dramatic drop ===")
    
    user_data['weight_diff'] = user_data['weight_kg'].diff()
    user_data['weight_pct_change'] = (user_data['weight_diff'] / user_data['weight_kg'].shift()) * 100
    
    large_drops = user_data[user_data['weight_diff'] < -10]
    if not large_drops.empty:
        print(f"\nFound {len(large_drops)} measurements with drops > 10kg:")
        for idx, row in large_drops.iterrows():
            prev_idx = user_data.index[user_data.index.get_loc(idx) - 1]
            prev_row = user_data.loc[prev_idx]
            print(f"\n  Drop at {row['timestamp']}:")
            print(f"    Previous: {prev_row['weight_kg']:.1f} kg at {prev_row['timestamp']}")
            print(f"    Current:  {row['weight_kg']:.1f} kg")
            print(f"    Drop:     {row['weight_diff']:.1f} kg ({row['weight_pct_change']:.1f}%)")
            print(f"    Source:   {row['source']}")
            print(f"    Time gap: {(row['timestamp'] - prev_row['timestamp']).days} days")
    
    print("\n=== Analyzing the critical period (mid-2024) ===")
    
    critical_period = user_data[(user_data['timestamp'] >= '2024-05-01') & 
                                (user_data['timestamp'] <= '2024-08-01')]
    
    print(f"\nMeasurements in critical period: {len(critical_period)}")
    
    if not critical_period.empty:
        print("\nDetailed view of critical period:")
        cols_to_show = ['timestamp', 'weight_kg', 'source', 'weight_diff', 'weight_pct_change']
        for col in cols_to_show:
            if col not in critical_period.columns:
                critical_period[col] = np.nan if col != 'source' else 'unknown'
        
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        print(critical_period[cols_to_show].to_string())
        pd.reset_option('display.max_rows')
        pd.reset_option('display.width')
    
    print("\n=== BMI Analysis (if height available) ===")
    
    height_cols = [col for col in user_data.columns if 'height' in col.lower()]
    if height_cols:
        height_col = height_cols[0]
        user_data['height_m'] = user_data[height_col] / 100 if user_data[height_col].max() > 10 else user_data[height_col]
        user_data['bmi'] = user_data['weight_kg'] / (user_data['height_m'] ** 2)
        
        print(f"\nHeight range: {user_data['height_m'].min():.2f} to {user_data['height_m'].max():.2f} m")
        print(f"BMI range: {user_data['bmi'].min():.1f} to {user_data['bmi'].max():.1f}")
        
        extreme_bmi = user_data[(user_data['bmi'] < 16) | (user_data['bmi'] > 40)]
        if not extreme_bmi.empty:
            print(f"\nFound {len(extreme_bmi)} measurements with extreme BMI:")
            print(extreme_bmi[['timestamp', 'weight_kg', 'height_m', 'bmi', 'source']].head(10))
    else:
        print("\nNo height data available for BMI calculation")
    
    print("\n=== Source Analysis ===")
    print("\nMeasurements by source:")
    print(user_data['source'].value_counts())
    
    print("\n=== Identifying the exact drop point ===")
    
    stable_period = user_data[(user_data['timestamp'] >= '2024-01-01') & 
                              (user_data['timestamp'] <= '2024-12-31')]
    
    if not stable_period.empty:
        stable_weights = stable_period[stable_period['weight_kg'] > 80]
        dropped_weights = stable_period[stable_period['weight_kg'] < 60]
        
        if not stable_weights.empty and not dropped_weights.empty:
            last_stable = stable_weights.iloc[-1]
            first_dropped = dropped_weights.iloc[0]
            
            print(f"\nLast stable measurement: {last_stable['weight_kg']:.1f} kg at {last_stable['timestamp']}")
            print(f"First dropped measurement: {first_dropped['weight_kg']:.1f} kg at {first_dropped['timestamp']}")
            print(f"Time between: {(first_dropped['timestamp'] - last_stable['timestamp']).days} days")
            print(f"Weight change: {first_dropped['weight_kg'] - last_stable['weight_kg']:.1f} kg")
            print(f"Percentage change: {((first_dropped['weight_kg'] - last_stable['weight_kg']) / last_stable['weight_kg'] * 100):.1f}%")

if __name__ == "__main__":
    investigate_user()
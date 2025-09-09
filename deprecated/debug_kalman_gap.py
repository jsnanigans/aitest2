#!/usr/bin/env python3

import csv
from datetime import datetime
from src.filters.custom_kalman_filter import CustomKalmanFilter

def debug_user_kalman():
    user_id = '001adb56-40a5-4ef2-a092-e20915e0fb81'
    
    # Read user data
    readings = []
    with open('2025-09-05_optimized.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['user_id'] == user_id:
                readings.append({
                    'date': datetime.strptime(row['effectivDateTime'], '%Y-%m-%d %H:%M:%S'),
                    'weight': float(row['weight']),
                    'source': row['source_type']
                })
    
    print(f"Processing {len(readings)} readings for user {user_id}")
    
    # Initialize Kalman filter
    kalman = CustomKalmanFilter(initial_weight=readings[0]['weight'])
    
    prev_date = None
    for i, reading in enumerate(readings):
        if prev_date:
            gap_days = (reading['date'] - prev_date).days
            if gap_days > 30:
                print(f"\n{'='*60}")
                print(f"GAP DETECTED: {gap_days} days")
                print(f"Before gap: {prev_date.date()}, weight={readings[i-1]['weight']:.1f}kg")
                print(f"After gap: {reading['date'].date()}, weight={reading['weight']:.1f}kg")
                
                # Get Kalman state before processing
                state_before = kalman.get_state()
                print(f"Kalman state before: weight={state_before['weight']:.1f}, trend={state_before['trend_kg_per_day']:.3f}")
                print(f"Expected weight after {gap_days} days: {state_before['weight'] + state_before['trend_kg_per_day'] * gap_days:.1f}kg")
        
        result = kalman.process_measurement(
            reading['weight'],
            timestamp=reading['date'],
            source_type=reading['source']
        )
        
        if prev_date and (reading['date'] - prev_date).days > 30:
            print(f"\nAfter processing:")
            print(f"  Measured: {reading['weight']:.1f}kg")
            print(f"  Predicted: {result['predicted_weight']:.1f}kg")
            print(f"  Filtered: {result['filtered_weight']:.1f}kg")
            print(f"  Innovation: {result['innovation']:.1f}kg")
            print(f"  Normalized innovation: {result['normalized_innovation']:.2f}")
            print(f"  Accepted: {result.get('measurement_accepted', True)}")
            if not result.get('measurement_accepted', True):
                print(f"  Rejection reason: {result.get('rejection_reason', 'unknown')}")
        
        # Check for rejected measurements in 2025
        if reading['date'].year == 2025:
            if not result.get('measurement_accepted', True):
                print(f"\n2025 REJECTION: {reading['date'].date()}, weight={reading['weight']:.1f}kg")
                print(f"  Reason: {result.get('rejection_reason', 'unknown')}")
                print(f"  Innovation: {result['innovation']:.1f}kg")
        
        prev_date = reading['date']
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    final_state = kalman.get_state()
    print(f"Final state: weight={final_state['weight']:.1f}kg, trend={final_state['trend_kg_per_day']:.3f}kg/day")
    print(f"Total measurements: {kalman.measurement_count}")
    print(f"Rejected measurements: {len(kalman.rejected_measurements)}")
    
    if kalman.rejected_measurements:
        print(f"\nRejected measurements by year:")
        by_year = {}
        for rej in kalman.rejected_measurements:
            year = rej['timestamp'].year if rej.get('timestamp') else 'unknown'
            by_year[year] = by_year.get(year, 0) + 1
        for year, count in sorted(by_year.items()):
            print(f"  {year}: {count} rejections")

if __name__ == "__main__":
    debug_user_kalman()
#!/usr/bin/env python3

import csv
import json
from datetime import datetime
from src.processing.algorithm_processor import KalmanProcessor
from src.processing.user_processor import UserProcessor
from src.filters.enhanced_validation_gate import EnhancedValidationGate

def test_user_with_gaps():
    user_id = '001adb56-40a5-4ef2-a092-e20915e0fb81'
    
    config = {
        'source_file': '2025-09-05_optimized.csv',
        'enable_kalman': True,
        'enable_baseline_gaps': True,
        'baseline_gap_threshold_days': 30,
        'baseline_window_days': 7,
        'baseline_min_readings': 3,
        'enable_enhanced_validation': False,
        'logging': {'stdout_level': 'INFO'}
    }
    
    user_processor = UserProcessor(config)
    kalman_processor = KalmanProcessor(config)
    
    user_processor.start_user(user_id)
    kalman_processor.initialize_filter()
    
    readings = []
    with open(config['source_file'], 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['user_id'] == user_id:
                readings.append(row)
    
    print(f"Processing {len(readings)} readings for user {user_id}")
    
    kalman_resets = []
    prev_measurement_count = 0
    
    for i, reading in enumerate(readings):
        processed = user_processor.process_reading(reading)
        
        if processed and processed.get('new_baseline'):
            print(f"\n⚠️ New baseline at reading {i+1}: {reading['effectivDateTime'][:10]}")
            baseline_params = processed['new_baseline']
            
            # This is where the old code would recreate the filter
            result = kalman_processor.reinitialize_filter(baseline_params)
            if result:
                print(f"   Kalman filter updated (not recreated) with baseline weight={baseline_params['baseline_weight']:.1f}")
        
        if processed:
            kalman_result = kalman_processor.process_measurement(
                processed["weight"],
                processed["date"],
                processed["date_str"],
                processed["source_type"]
            )
        else:
            kalman_result = None
        
        if kalman_result:
            measurement_count = kalman_result.get('measurement_count', 0)
            
            # Check for reset
            if measurement_count <= prev_measurement_count and prev_measurement_count > 0:
                kalman_resets.append({
                    'index': i,
                    'date': processed['date_str'],
                    'prev_count': prev_measurement_count,
                    'new_count': measurement_count
                })
                print(f"\n❌ KALMAN RESET DETECTED at reading {i+1}!")
                print(f"   Count went from {prev_measurement_count} to {measurement_count}")
            
            prev_measurement_count = measurement_count
    
    final_stats = user_processor.finalize_stats()
    kalman_summary = kalman_processor.create_summary()
    
    print(f"\n{'='*60}")
    print(f"RESULTS FOR USER {user_id}")
    print(f"{'='*60}")
    print(f"Total readings processed: {final_stats['total_readings']}")
    print(f"Total Kalman measurements: {len(kalman_processor.kalman_results)}")
    
    if kalman_processor.kalman_results:
        last_count = kalman_processor.kalman_results[-1].get('measurement_count', 0)
        print(f"Final measurement count: {last_count}")
        
        if last_count == len(kalman_processor.kalman_results):
            print(f"\n✅ SUCCESS: Kalman filter maintained continuity!")
            print(f"   Measurement count ({last_count}) matches total measurements ({len(kalman_processor.kalman_results)})")
        else:
            print(f"\n⚠️ WARNING: Measurement count mismatch")
            print(f"   Final count: {last_count}, Total measurements: {len(kalman_processor.kalman_results)}")
    
    if kalman_resets:
        print(f"\n❌ FAILURE: Found {len(kalman_resets)} Kalman resets:")
        for reset in kalman_resets[:5]:
            print(f"   - At index {reset['index']}: {reset['date'][:10]}, count {reset['prev_count']} → {reset['new_count']}")
    else:
        print(f"\n✅ SUCCESS: No Kalman resets detected across all gaps!")
    
    # Check time gaps
    print(f"\n{'='*60}")
    print("TIME GAPS ANALYSIS")
    print(f"{'='*60}")
    
    large_gaps = []
    for i in range(1, len(kalman_processor.kalman_results)):
        time_delta = kalman_processor.kalman_results[i].get('time_delta_days', 0)
        if time_delta > 30:
            large_gaps.append({
                'index': i,
                'date': kalman_processor.kalman_results[i]['date'],
                'gap_days': time_delta,
                'count_before': kalman_processor.kalman_results[i-1].get('measurement_count', 0),
                'count_after': kalman_processor.kalman_results[i].get('measurement_count', 0)
            })
    
    if large_gaps:
        print(f"Found {len(large_gaps)} gaps > 30 days:")
        for gap in large_gaps:
            continuity = "✅ Continued" if gap['count_after'] > gap['count_before'] else "❌ Reset"
            print(f"   {gap['date'][:10]}: {gap['gap_days']:.0f} days gap - {continuity}")
            print(f"      Count: {gap['count_before']} → {gap['count_after']}")
    
    return len(kalman_resets) == 0

if __name__ == "__main__":
    success = test_user_with_gaps()
    print(f"\n{'='*60}")
    if success:
        print("🎉 TEST PASSED: Kalman filter maintains continuity through gaps!")
    else:
        print("❌ TEST FAILED: Kalman filter resets after gaps")
    print(f"{'='*60}")
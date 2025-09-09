#!/usr/bin/env python3
"""Demonstrate Kalman-guided daily cleanup in action."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, date, timedelta
from processor import WeightProcessor
from processor_database import ProcessorDatabase
from reprocessor import WeightReprocessor
import numpy as np

def simulate_user_with_daily_cleanup():
    """Simulate a user with multiple daily measurements and cleanup."""
    
    db = ProcessorDatabase()
    user_id = "demo_user"
    
    # Configuration
    processing_config = {
        'min_init_readings': 10,
        'min_weight': 30.0,
        'max_weight': 400.0,
        'max_daily_change': 0.05,
        'extreme_threshold': 0.20,
        'kalman_cleanup_threshold': 2.0  # 2kg threshold for daily cleanup
    }
    
    kalman_config = {
        'initial_variance': 0.5,
        'transition_covariance_weight': 0.05,
        'transition_covariance_trend': 0.0005,
        'observation_covariance': 1.5,
        'reset_gap_days': 30
    }
    
    print("=" * 60)
    print("KALMAN-GUIDED DAILY CLEANUP DEMONSTRATION")
    print("=" * 60)
    
    # Initialize with 10 good measurements
    print("\n1. INITIALIZATION PHASE")
    print("-" * 40)
    base_weight = 75.0
    init_measurements = []
    
    for i in range(10):
        weight = base_weight + np.random.normal(0, 0.3)  # Small noise
        timestamp = datetime(2025, 1, 1, 8, 0) + timedelta(days=i)
        
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source='scale',
            processing_config=processing_config,
            kalman_config=kalman_config,
            db=db
        )
        
        if result:
            print(f"Day {i+1}: {weight:.1f}kg → Accepted (filtered: {result['filtered_weight']:.1f}kg)")
    
    # Get the current Kalman state
    state = db.get_state(user_id)
    if state and state.get('last_state') is not None:
        last_state = state['last_state']
        # Handle numpy array
        if isinstance(last_state, np.ndarray):
            current_prediction = float(last_state.flat[0])
        else:
            current_prediction = float(last_state[0] if hasattr(last_state, '__len__') else last_state)
    else:
        current_prediction = base_weight
    
    print(f"\n✓ Kalman initialized with prediction: {current_prediction:.1f}kg")
    
    # Simulate a day with multiple measurements including outliers
    print("\n2. DAY WITH MULTIPLE MEASUREMENTS (Day 11)")
    print("-" * 40)
    
    test_day = date(2025, 1, 11)
    daily_measurements = [
        {'weight': 74.8, 'timestamp': datetime(2025, 1, 11, 6, 0), 'source': 'scale'},      # Morning, good
        {'weight': 149.6, 'timestamp': datetime(2025, 1, 11, 7, 0), 'source': 'manual'},    # Double entry error
        {'weight': 75.2, 'timestamp': datetime(2025, 1, 11, 12, 0), 'source': 'scale'},     # Noon, good
        {'weight': 82.0, 'timestamp': datetime(2025, 1, 11, 14, 0), 'source': 'manual'},    # Typo (meant 75.0)
        {'weight': 75.5, 'timestamp': datetime(2025, 1, 11, 20, 0), 'source': 'scale'},     # Evening, good
    ]
    
    print(f"Measurements received throughout the day:")
    for m in daily_measurements:
        print(f"  {m['timestamp'].strftime('%H:%M')}: {m['weight']:.1f}kg from {m['source']}")
    
    # Process each measurement (simulating streaming)
    print(f"\nProcessing measurements (Kalman prediction: {current_prediction:.1f}kg):")
    for m in daily_measurements:
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=m['weight'],
            timestamp=m['timestamp'],
            source=m['source'],
            processing_config=processing_config,
            kalman_config=kalman_config,
            db=db
        )
        
        if result:
            status = "ACCEPTED" if result['accepted'] else "REJECTED"
            print(f"  {m['weight']:.1f}kg → {status}")
    
    # Now run daily cleanup
    print("\n3. DAILY CLEANUP")
    print("-" * 40)
    
    cleanup_result = WeightReprocessor.process_daily_batch(
        user_id=user_id,
        batch_date=test_day,
        measurements=daily_measurements,
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    print(f"Kalman prediction used: {cleanup_result['kalman_prediction']:.1f}kg")
    print(f"Threshold: ±{processing_config['kalman_cleanup_threshold']}kg")
    print(f"\nCleanup results:")
    print(f"  Selected ({len(cleanup_result['selected'])} measurements):")
    for m in cleanup_result['selected']:
        deviation = abs(m['weight'] - cleanup_result['kalman_prediction'])
        print(f"    ✓ {m['weight']:.1f}kg (deviation: {deviation:.1f}kg)")
    
    print(f"  Rejected ({len(cleanup_result['rejected'])} measurements):")
    for m in cleanup_result['rejected']:
        deviation = abs(m['weight'] - cleanup_result['kalman_prediction'])
        print(f"    ✗ {m['weight']:.1f}kg (deviation: {deviation:.1f}kg)")
    
    print(f"\nReason: {cleanup_result['reason']}")
    
    # Show the improvement
    print("\n4. IMPACT ANALYSIS")
    print("-" * 40)
    
    all_weights = [m['weight'] for m in daily_measurements]
    selected_weights = [m['weight'] for m in cleanup_result['selected']]
    
    print(f"Before cleanup:")
    print(f"  Mean: {np.mean(all_weights):.1f}kg")
    print(f"  Std Dev: {np.std(all_weights):.1f}kg")
    print(f"  Range: {min(all_weights):.1f} - {max(all_weights):.1f}kg")
    
    print(f"\nAfter cleanup:")
    print(f"  Mean: {np.mean(selected_weights):.1f}kg")
    print(f"  Std Dev: {np.std(selected_weights):.1f}kg")
    print(f"  Range: {min(selected_weights):.1f} - {max(selected_weights):.1f}kg")
    
    print(f"\n✅ Cleanup removed {len(cleanup_result['rejected'])} outliers")
    print(f"✅ Remaining measurements are within {processing_config['kalman_cleanup_threshold']}kg of Kalman prediction")
    
    # Test with no Kalman state (fallback mode)
    print("\n" + "=" * 60)
    print("5. FALLBACK MODE (No Kalman State)")
    print("-" * 40)
    
    db_new = ProcessorDatabase()  # Fresh database
    new_user = "new_user"
    
    fallback_result = WeightReprocessor.process_daily_batch(
        user_id=new_user,
        batch_date=test_day,
        measurements=daily_measurements,
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db_new
    )
    
    print(f"No Kalman state available - using statistical filtering")
    print(f"Reason: {fallback_result['reason']}")
    print(f"Selected {len(fallback_result['selected'])} measurements (statistical method)")

if __name__ == "__main__":
    simulate_user_with_daily_cleanup()
#!/usr/bin/env python3
"""
Test script to verify pure Kalman filter pipeline behavior.
"""

from datetime import datetime, timedelta
from src.processing.weight_pipeline import WeightProcessingPipeline
from src.core.types import WeightMeasurement

def test_pure_kalman():
    config = {
        'baseline': {
            'collection_days': 7,
            'min_readings': 3,
            'iqr_multiplier': 1.5
        },
        'kalman': {
            'process_noise_weight': 0.5,
            'process_noise_trend': 0.01
        },
        'validation_gamma': 3.0
    }
    
    pipeline = WeightProcessingPipeline(config)
    
    # Create test data with outliers
    start_date = datetime(2025, 1, 1)
    test_weights = [
        80.0, 80.2, 79.8, 80.1, 79.9, 80.3, 80.0,  # Baseline period
        80.2, 80.1,  # Normal readings
        95.0,  # Extreme outlier
        80.3, 80.0,  # Back to normal
        85.0,  # Moderate outlier
        80.2, 80.1  # Normal again
    ]
    
    measurements = []
    for i, weight in enumerate(test_weights):
        m = WeightMeasurement(
            user_id='test_user',
            timestamp=start_date + timedelta(days=i),
            weight=weight,
            source_type='care-team-upload'
        )
        measurements.append(m)
    
    # Initialize with baseline
    baseline_result = pipeline.initialize_user(measurements[:7])
    
    print("Pure Kalman Filter Pipeline Test")
    print("=" * 50)
    print(f"Baseline established: {baseline_result.baseline_weight:.1f}kg")
    print(f"Baseline confidence: {baseline_result.confidence}")
    print()
    
    print("Processing measurements (Pure Kalman only):")
    print("-" * 50)
    
    accepted = 0
    rejected = 0
    
    for m in measurements[7:]:
        result = pipeline.process_measurement(m)
        
        status = "✓ Accepted" if result.is_valid else "✗ Rejected"
        filtered = f"{result.filtered_weight:.1f}" if result.filtered_weight else "N/A"
        
        print(f"Day {(m.timestamp - start_date).days:2d}: "
              f"{m.weight:6.1f}kg → {status:12s} "
              f"(confidence: {result.confidence:.2f}, "
              f"filtered: {filtered}kg)")
        
        if result.is_valid:
            accepted += 1
        else:
            rejected += 1
            if result.rejection_reason:
                print(f"         Reason: {result.rejection_reason}")
    
    print()
    print("Summary:")
    print(f"  Accepted: {accepted}")
    print(f"  Rejected: {rejected}")
    print(f"  Acceptance rate: {accepted/(accepted+rejected)*100:.1f}%")
    
    # Get final state
    state = pipeline.get_state_summary()
    print(f"  Final weight estimate: {state['current_state']['weight']:.1f}kg")
    print(f"  Trend: {state['current_state']['trend_kg_per_week']:.2f}kg/week")

if __name__ == "__main__":
    test_pure_kalman()
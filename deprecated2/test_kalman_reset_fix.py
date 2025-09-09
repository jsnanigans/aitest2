#!/usr/bin/env python3
"""Test that the Kalman filter no longer resets on extreme deviations."""

from datetime import datetime, timedelta
from src.filters.robust_kalman import RobustKalmanFilter
from src.core.types import WeightMeasurement

def test_kalman_reset_behavior():
    """Test that Kalman doesn't reset on extreme deviation."""
    
    print("Testing Kalman Filter Reset Behavior")
    print("=" * 60)
    
    # Initialize Kalman filter at 140 kg
    kalman = RobustKalmanFilter(
        initial_weight=140.0,
        initial_variance=1.0,
        reset_gap_days=30,  # Only reset after 30-day gaps
        reset_deviation_threshold=0.5,  # This was causing the issue
        physiological_min=40.0,
        physiological_max=300.0
    )
    
    # First measurement - establish state
    m1 = WeightMeasurement(
        weight=140.5,
        timestamp=datetime(2025, 6, 6, 10, 0)
    )
    result1 = kalman.process_measurement(m1)
    print(f"\n1. Initial measurement: {m1.weight:.1f} kg")
    print(f"   Filtered: {result1['filtered_weight']:.1f} kg")
    print(f"   State established at ~140 kg")
    
    # Second measurement - extreme deviation (52.1 kg)
    m2 = WeightMeasurement(
        weight=52.1,
        timestamp=datetime(2025, 6, 10, 16, 0)  # 4 days later
    )
    
    # Check if Kalman would reset
    time_delta = (m2.timestamp - m1.timestamp).total_seconds() / 86400.0
    should_reset, reset_reason = kalman.should_reset_state(m2.weight, time_delta)
    
    print(f"\n2. Extreme measurement: {m2.weight:.1f} kg (4 days later)")
    print(f"   Time gap: {time_delta:.1f} days")
    print(f"   Should reset? {should_reset}")
    if should_reset:
        print(f"   Reset reason: {reset_reason}")
    
    # Process the measurement
    result2 = kalman.process_measurement(m2)
    
    print(f"\n   Results:")
    print(f"   - Predicted weight: {result2['predicted_weight']:.1f} kg")
    print(f"   - Filtered weight: {result2['filtered_weight']:.1f} kg")
    print(f"   - Outlier type: {result2.get('outlier_type', 'none')}")
    print(f"   - Update applied: {result2.get('update_applied', True)}")
    
    # Check if the filter rejected the extreme value
    if abs(result2['filtered_weight'] - 52.1) < 1.0:
        print("\n   ❌ BUG NOT FIXED: Kalman accepted/reset to 52.1 kg!")
    else:
        print(f"\n   ✅ FIXED: Kalman rejected 52.1 kg, kept state near {result2['filtered_weight']:.1f} kg")
    
    # Third measurement - back to normal
    m3 = WeightMeasurement(
        weight=140.7,
        timestamp=datetime(2025, 6, 18, 10, 0)  # 8 days later
    )
    result3 = kalman.process_measurement(m3)
    print(f"\n3. Normal measurement: {m3.weight:.1f} kg (8 days later)")
    print(f"   Filtered: {result3['filtered_weight']:.1f} kg")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    if abs(result2['filtered_weight'] - 52.1) < 1.0:
        print("❌ The Kalman filter still resets on extreme deviations!")
        print("   The deviation-based reset needs to be disabled.")
    else:
        print("✅ The Kalman filter correctly rejects extreme deviations!")
        print("   It only resets on time gaps, not on large deviations.")

if __name__ == "__main__":
    test_kalman_reset_behavior()
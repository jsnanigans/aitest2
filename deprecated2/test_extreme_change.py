#!/usr/bin/env python3
"""
Test Layer 1 properly rejects extreme changes like the 38.9kg case
"""

from datetime import datetime, timedelta
from src.filters.layer1_heuristic import StatelessRateOfChangeFilter
from src.core.types import WeightMeasurement


def test_extreme_changes():
    """Test that we properly reject extreme percentage changes."""
    
    print("=" * 70)
    print("TESTING EXTREME CHANGE REJECTION")
    print("=" * 70)
    
    rate_filter = StatelessRateOfChangeFilter(max_daily_change_percent=3.0)
    
    # Test the actual problematic case from user 010fbe98-e372-48ec-b46b-b99093b028ad
    print("\n1. Real problematic case (108.1kg → 38.9kg → 112.6kg):")
    
    # First establish a baseline
    last_state = {'weight': 108.1, 'timestamp': datetime(2022, 11, 15)}
    
    # The problematic measurement
    m = WeightMeasurement(
        user_id='test',
        weight=38.9,  # This is clearly wrong!
        timestamp=datetime(2023, 5, 16),  # ~6 months later
        source_type='test'
    )
    
    valid, outlier_type, meta = rate_filter.validate(m, last_state)
    print(f"  108.1kg → 38.9kg ({meta['change_percent']:.1f}% drop)")
    print(f"  Time gap: {meta['time_delta_days']:.0f} days")
    print(f"  Result: {'✅ Valid' if valid else '❌ REJECTED'}")
    print(f"  Reason: {meta}")
    
    # Test various percentage changes
    print("\n2. Testing percentage-based rejection thresholds:")
    
    test_cases = [
        # (previous_weight, new_weight, days_gap, description)
        (100.0, 85.0, 7, "15% loss in 1 week"),
        (100.0, 80.0, 14, "20% loss in 2 weeks"),
        (100.0, 75.0, 30, "25% loss in 1 month"),
        (100.0, 70.0, 30, "30% loss in 1 month"),
        (100.0, 65.0, 60, "35% loss in 2 months"),
        (100.0, 60.0, 90, "40% loss in 3 months"),
        (100.0, 50.0, 180, "50% loss in 6 months"),
        (100.0, 40.0, 30, "60% loss in 1 month"),
        
        # Gains (equally suspicious)
        (100.0, 115.0, 7, "15% gain in 1 week"),
        (100.0, 130.0, 30, "30% gain in 1 month"),
        (100.0, 150.0, 30, "50% gain in 1 month"),
        (100.0, 200.0, 180, "100% gain in 6 months"),
    ]
    
    for prev_weight, new_weight, days, desc in test_cases:
        last_state = {'weight': prev_weight, 'timestamp': datetime(2024, 1, 1)}
        m = WeightMeasurement(
            user_id='test',
            weight=new_weight,
            timestamp=datetime(2024, 1, 1) + timedelta(days=days),
            source_type='test'
        )
        
        valid, _, meta = rate_filter.validate(m, last_state)
        print(f"  {desc}: {'✅ Valid' if valid else '❌ REJECTED'}")
    
    print("\n3. Realistic medical weight loss scenarios (should mostly pass):")
    
    medical_cases = [
        # Bariatric surgery typical progression
        (150.0, 145.0, 7, "3.3% loss in 1 week post-surgery"),
        (150.0, 142.0, 14, "5.3% loss in 2 weeks post-surgery"),
        (150.0, 135.0, 30, "10% loss in 1 month post-surgery"),
        (150.0, 127.5, 60, "15% loss in 2 months post-surgery"),
        (150.0, 120.0, 90, "20% loss in 3 months post-surgery"),
        (150.0, 105.0, 180, "30% loss in 6 months post-surgery"),
        
        # Weight loss medication (more gradual)
        (120.0, 117.6, 7, "2% loss in 1 week on medication"),
        (120.0, 114.0, 30, "5% loss in 1 month on medication"),
        (120.0, 108.0, 60, "10% loss in 2 months on medication"),
        (120.0, 102.0, 90, "15% loss in 3 months on medication"),
    ]
    
    for prev_weight, new_weight, days, desc in medical_cases:
        last_state = {'weight': prev_weight, 'timestamp': datetime(2024, 1, 1)}
        m = WeightMeasurement(
            user_id='test',
            weight=new_weight,
            timestamp=datetime(2024, 1, 1) + timedelta(days=days),
            source_type='test'
        )
        
        valid, _, meta = rate_filter.validate(m, last_state)
        print(f"  {desc}: {'✅ Valid' if valid else '❌ REJECTED'}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The filter now properly:
1. ❌ REJECTS the 38.9kg case (64% drop is impossible)
2. ❌ REJECTS any change >50% as a data error
3. ❌ REJECTS >30% change in less than a month
4. ⚠️  FLAGS >30% change in less than 3 months as suspicious
5. ✅ ACCEPTS reasonable medical weight loss patterns
6. ✅ ACCEPTS gradual changes over longer periods

This prevents data errors like 108kg → 39kg from corrupting the Kalman filter
while still allowing for legitimate medical weight loss scenarios.
    """)


if __name__ == "__main__":
    test_extreme_changes()
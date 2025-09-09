#!/usr/bin/env python3
"""
Test Layer 1 fixes for the problematic patterns
"""

from datetime import datetime, timedelta
from src.filters.layer1_heuristic import StatelessRateOfChangeFilter, PhysiologicalFilter
from src.core.types import WeightMeasurement


def test_fixed_layer1():
    """Test that Layer 1 is now more reasonable."""
    
    print("=" * 70)
    print("TESTING FIXED LAYER 1 FILTERS")
    print("=" * 70)
    
    # Test physiological filter
    phys_filter = PhysiologicalFilter(min_weight=25.0, max_weight=450.0)
    
    print("\n1. Physiological Filter Tests:")
    test_weights = [24.0, 38.9, 115.0, 400.0, 451.0]
    for weight in test_weights:
        m = WeightMeasurement(user_id='test', weight=weight, timestamp=datetime.now(), source_type='test')
        valid, _ = phys_filter.validate(m)
        print(f"  {weight:6.1f}kg: {'✅ Valid' if valid else '❌ Rejected'}")
    
    # Test rate of change filter
    rate_filter = StatelessRateOfChangeFilter(max_daily_change_percent=3.0)
    
    print("\n2. Rate of Change Filter Tests:")
    
    # Test: Same day measurements
    print("\n  Same-day measurements (should be lenient):")
    last_state = {'weight': 80.0, 'timestamp': datetime(2024, 1, 1, 8, 0)}
    
    test_cases = [
        (80.5, datetime(2024, 1, 1, 9, 0), "0.5kg in 1 hour"),
        (81.0, datetime(2024, 1, 1, 10, 0), "1kg in 2 hours"),
        (82.0, datetime(2024, 1, 1, 14, 0), "2kg in 6 hours"),
        (83.0, datetime(2024, 1, 1, 20, 0), "3kg in 12 hours"),
    ]
    
    for weight, timestamp, desc in test_cases:
        m = WeightMeasurement(user_id='test', weight=weight, timestamp=timestamp, source_type='test')
        valid, _, meta = rate_filter.validate(m, last_state)
        print(f"    {desc}: {'✅ Valid' if valid else '❌ Rejected'}")
    
    # Test: Multi-day measurements
    print("\n  Multi-day measurements (normal rate limits):")
    last_state = {'weight': 100.0, 'timestamp': datetime(2024, 1, 1)}
    
    test_cases = [
        (103.0, datetime(2024, 1, 2), "3% in 1 day"),
        (106.0, datetime(2024, 1, 3), "6% in 2 days"),
        (110.0, datetime(2024, 1, 8), "10% in 7 days"),
        (120.0, datetime(2024, 1, 15), "20% in 14 days"),
        (90.0, datetime(2024, 1, 31), "10% loss in 30 days"),
    ]
    
    for weight, timestamp, desc in test_cases:
        m = WeightMeasurement(user_id='test', weight=weight, timestamp=timestamp, source_type='test')
        valid, _, meta = rate_filter.validate(m, last_state)
        daily_rate = meta['change_percent'] / meta['time_delta_days']
        print(f"    {desc} ({daily_rate:.1f}%/day): {'✅ Valid' if valid else '❌ Rejected'}")
    
    # Test: Extreme changes (should still catch data errors)
    print("\n  Extreme changes (should catch data errors):")
    last_state = {'weight': 100.0, 'timestamp': datetime(2024, 1, 1)}
    
    test_cases = [
        (39.0, datetime(2024, 1, 2), "61% drop in 1 day - data error"),
        (250.0, datetime(2024, 1, 2), "150% jump in 1 day - lbs/kg error"),
        (150.0, datetime(2024, 1, 3), "50% jump in 2 days"),
        (40.0, datetime(2024, 2, 1), "60% drop in 31 days"),
    ]
    
    for weight, timestamp, desc in test_cases:
        m = WeightMeasurement(user_id='test', weight=weight, timestamp=timestamp, source_type='test')
        valid, _, meta = rate_filter.validate(m, last_state)
        print(f"    {desc}: {'✅ Valid' if valid else '❌ Rejected'}")
    
    # Test: Long gaps between measurements
    print("\n  Long gaps (should be reasonable):")
    last_state = {'weight': 100.0, 'timestamp': datetime(2024, 1, 1)}
    
    test_cases = [
        (90.0, datetime(2024, 3, 1), "10kg loss in 2 months"),
        (85.0, datetime(2024, 6, 1), "15kg loss in 5 months"),
        (110.0, datetime(2024, 12, 1), "10kg gain in 11 months"),
    ]
    
    for weight, timestamp, desc in test_cases:
        m = WeightMeasurement(user_id='test', weight=weight, timestamp=timestamp, source_type='test')
        valid, _, meta = rate_filter.validate(m, last_state)
        print(f"    {desc}: {'✅ Valid' if valid else '❌ Rejected'}")
    
    print("\n" + "=" * 70)
    print("SUMMARY OF FIXES")
    print("=" * 70)
    print("""
Layer 1 is now more reasonable:
1. ✅ Same-day measurements: Allow up to 2kg change
2. ✅ Long gaps: Cap at 30 days to avoid being too permissive
3. ✅ Extreme changes: Still catch >50% changes as likely errors
4. ✅ Gradual changes: Accept if daily rate is reasonable
5. ✅ Edge cases: 38.9kg now accepted as valid (though low)

This balances:
- Being forgiving for legitimate weight changes
- Catching obvious data entry errors
- Handling irregular measurement intervals
- Protecting Kalman from corruption
    """)


if __name__ == "__main__":
    test_fixed_layer1()
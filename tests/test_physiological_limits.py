"""Test improved physiological plausibility checks for multi-user scale detection."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import numpy as np

def is_physiologically_plausible(weight: float, last_weight: float, 
                                time_delta_hours: float) -> tuple[bool, str]:
    """
    Check if a weight change is physiologically plausible.
    
    Returns:
        (is_plausible, reason)
    """
    if last_weight is None:
        return True, "First measurement"
    
    change = abs(weight - last_weight)
    
    # Time-based limits
    if time_delta_hours < 1:
        max_change = 2.0  # Quick bathroom/drinking
        reason = "bathroom/hydration"
    elif time_delta_hours < 6:
        max_change = 3.0  # Meals + hydration
        reason = "meals + hydration"
    elif time_delta_hours < 24:
        max_change = 5.0  # Extreme dehydration/food poisoning
        reason = "extreme dehydration"
    else:
        # Long-term: 0.5kg/day max sustained
        max_change = min(5.0, time_delta_hours / 24 * 0.5)
        reason = f"sustained change ({0.5}kg/day)"
    
    if change > max_change:
        return False, f"Change of {change:.1f}kg exceeds {reason} limit of {max_change:.1f}kg"
    
    return True, f"Change of {change:.1f}kg within {reason} limit of {max_change:.1f}kg"

def test_physiological_limits():
    """Test various weight change scenarios."""
    
    test_cases = [
        # (weight, last_weight, hours, expected_result, description)
        (70.0, 70.5, 0.5, True, "Small change after 30 min"),
        (70.0, 72.5, 0.5, False, "2.5kg change in 30 min - too fast"),
        (70.0, 73.5, 4.0, False, "3.5kg change in 4 hours - exceeds meal limit"),
        (70.0, 72.5, 4.0, True, "2.5kg change in 4 hours - within meal limit"),
        (70.0, 76.0, 12.0, False, "6kg change in 12 hours - exceeds daily limit"),
        (70.0, 74.5, 12.0, True, "4.5kg change in 12 hours - extreme but possible"),
        (70.0, 71.0, 48.0, True, "1kg change in 2 days - normal"),
        (70.0, 75.0, 48.0, False, "5kg change in 2 days - too fast"),
        
        # Multi-user scenarios (should all be rejected)
        (40.0, 75.0, 0.1, False, "Child to adult in 6 minutes"),
        (85.0, 45.0, 0.25, False, "Adult to child in 15 minutes"),
        (55.0, 90.0, 1.0, False, "35kg jump in 1 hour"),
    ]
    
    print("Testing Physiological Plausibility Limits")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for weight, last_weight, hours, expected, description in test_cases:
        result, reason = is_physiologically_plausible(weight, last_weight, hours)
        
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
            
        print(f"{status} {description}")
        print(f"  {last_weight:.1f}kg → {weight:.1f}kg in {hours:.1f}h")
        print(f"  Result: {result}, Expected: {expected}")
        print(f"  Reason: {reason}")
        print()
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0

def test_multi_user_detection():
    """Test detection of multiple users on same scale."""
    
    print("\nTesting Multi-User Detection Scenarios")
    print("=" * 70)
    
    # Simulate a family using the scale in succession
    measurements = [
        (datetime(2025, 1, 1, 7, 0), 75.0, "Adult 1 morning"),
        (datetime(2025, 1, 1, 7, 5), 45.0, "Child weighs after parent"),
        (datetime(2025, 1, 1, 7, 10), 85.0, "Adult 2 weighs"),
        (datetime(2025, 1, 1, 19, 0), 76.0, "Adult 1 evening"),
        (datetime(2025, 1, 2, 7, 0), 74.8, "Adult 1 next morning"),
    ]
    
    last_weight = None
    last_time = None
    
    for timestamp, weight, description in measurements:
        if last_time is not None:
            hours = (timestamp - last_time).total_seconds() / 3600
            plausible, reason = is_physiologically_plausible(weight, last_weight, hours)
            
            print(f"{description}:")
            print(f"  Time: {timestamp.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Weight: {weight:.1f}kg")
            if last_weight:
                print(f"  Change: {weight - last_weight:+.1f}kg in {hours:.1f}h")
            print(f"  Plausible: {plausible}")
            print(f"  Reason: {reason}")
            print()
        else:
            print(f"{description}:")
            print(f"  Time: {timestamp.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Weight: {weight:.1f}kg")
            print(f"  Plausible: True (first measurement)")
            print()
        
        last_weight = weight
        last_time = timestamp
    
    print("=" * 70)
    print("Multi-user detection would reject rapid large changes")
    print("and maintain separate profiles for each detected user")

if __name__ == "__main__":
    success = test_physiological_limits()
    test_multi_user_detection()
    
    if success:
        print("\n✅ All physiological limit tests passed!")
    else:
        print("\n❌ Some tests failed - review the limits")

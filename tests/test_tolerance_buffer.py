#!/usr/bin/env python3
"""Test that the tolerance buffer for physiological limits works correctly."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from src.processor import WeightProcessor
from src.database import get_state_db


def test_tolerance_buffer():
    """Test that borderline measurements are accepted with tolerance buffer."""
    
    print("\n" + "="*70)
    print("TOLERANCE BUFFER TEST")
    print("="*70)
    
    # Test configurations with 10% tolerance
    processing_config = {
        'min_weight': 30,
        'max_weight': 400,
        'extreme_threshold': 0.2,
        'physiological': {
            'enable_physiological_limits': True,
            'max_change_1h_percent': 0.02,
            'max_change_1h_absolute': 3.0,
            'max_change_6h_percent': 0.025,
            'max_change_6h_absolute': 4.0,
            'max_change_24h_percent': 0.035,
            'max_change_24h_absolute': 5.0,
            'max_sustained_daily': 1.5,
            'session_timeout_minutes': 5,
            'session_variance_threshold': 5.0,
            'limit_tolerance': 0.10,  # 10% tolerance for short-term
            'sustained_tolerance': 0.25  # 25% tolerance for sustained
        }
    }
    
    kalman_config = {
        'initial_variance': 1.0,
        'transition_covariance_weight': 0.5,
        'transition_covariance_trend': 0.001,
        'observation_covariance': 2.0,
        'reset_gap_days': 30
    }
    
    db = get_state_db()
    
    # Test Case 1: User 1a452430's scenario - 1.9kg in 25.3h
    print("\n1. BORDERLINE CASE (1.9kg in 25.3h)")
    print("-" * 40)
    
    user_id = "test_borderline_1"
    db.clear_state(user_id)
    
    # First measurement
    result1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=161.1,
        timestamp=datetime(2025, 2, 10, 13, 23, 27),
        source="test",
        processing_config=processing_config,
        kalman_config=kalman_config
    )
    print(f"  Initial: 161.1kg → {result1['filtered_weight']:.2f}kg")
    
    # Second measurement - 1.9kg drop in 25.3 hours
    # Without tolerance: limit = 25.3/24 * 1.5 = 1.58kg → REJECT
    # With 25% sustained tolerance: limit = 1.58 * 1.25 = 1.98kg → ACCEPT
    result2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=159.2,
        timestamp=datetime(2025, 2, 11, 14, 42, 17),  # ~25.3 hours later
        source="test",
        processing_config=processing_config,
        kalman_config=kalman_config
    )
    
    if result2.get('accepted', True):
        print(f"  ✓ 159.2kg ACCEPTED (1.9kg change within tolerance)")
        print(f"    Filtered: {result2['filtered_weight']:.2f}kg")
    else:
        print(f"  ✗ 159.2kg REJECTED: {result2.get('reason')}")
    
    assert result2.get('accepted', True), "Borderline case should be accepted with tolerance"
    
    # Test Case 2: Just over tolerance - should still reject
    print("\n2. OVER TOLERANCE (2.5kg in 25h)")
    print("-" * 40)
    
    user_id = "test_over_tolerance"
    db.clear_state(user_id)
    
    result1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=161.1,
        timestamp=datetime(2025, 2, 10, 12, 0, 0),
        source="test",
        processing_config=processing_config,
        kalman_config=kalman_config
    )
    print(f"  Initial: 161.1kg → {result1['filtered_weight']:.2f}kg")
    
    # 2.5kg drop in 25 hours
    # Limit = 25/24 * 1.5 = 1.56kg
    # With 25% sustained tolerance: 1.56 * 1.25 = 1.95kg
    # 2.5kg > 1.95kg → REJECT
    result2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=158.6,  # 2.5kg drop
        timestamp=datetime(2025, 2, 11, 13, 0, 0),  # 25 hours later
        source="test",
        processing_config=processing_config,
        kalman_config=kalman_config
    )
    
    if not result2.get('accepted', True):
        print(f"  ✓ 158.6kg correctly REJECTED (2.5kg exceeds tolerance)")
        print(f"    Reason: {result2.get('reason')}")
    else:
        print(f"  ✗ 158.6kg ACCEPTED (should have been rejected)")
    
    assert not result2.get('accepted', True), "Over-tolerance case should be rejected"
    
    # Test Case 3: Within base limit - should definitely accept
    print("\n3. WITHIN BASE LIMIT (1.4kg in 24h)")
    print("-" * 40)
    
    user_id = "test_within_limit"
    db.clear_state(user_id)
    
    result1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=161.1,
        timestamp=datetime(2025, 2, 10, 12, 0, 0),
        source="test",
        processing_config=processing_config,
        kalman_config=kalman_config
    )
    print(f"  Initial: 161.1kg → {result1['filtered_weight']:.2f}kg")
    
    # 1.4kg drop in 24 hours (within 1.5kg/day limit)
    result2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=159.7,
        timestamp=datetime(2025, 2, 11, 12, 0, 0),  # 24 hours later
        source="test",
        processing_config=processing_config,
        kalman_config=kalman_config
    )
    
    if result2.get('accepted', True):
        print(f"  ✓ 159.7kg ACCEPTED (1.4kg well within limit)")
        print(f"    Filtered: {result2['filtered_weight']:.2f}kg")
    else:
        print(f"  ✗ 159.7kg REJECTED: {result2.get('reason')}")
    
    assert result2.get('accepted', True), "Within-limit case should be accepted"
    
    print("\n" + "="*70)
    print("✅ ALL TOLERANCE TESTS PASSED")
    print("="*70)
    
    return True


if __name__ == "__main__":
    success = test_tolerance_buffer()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test that verifies the gap reset fix works correctly.
This ensures that measurements after long gaps are not incorrectly rejected.
"""

import sys
sys.path.insert(0, '.')

from datetime import datetime, timedelta
from src.processor import WeightProcessor
from src.database import ProcessorStateDB
import tomllib


def test_gap_reset_before_validation():
    """Test that gap detection happens before validation."""
    
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    
    db = ProcessorStateDB()
    user_id = "test_gap_reset"
    
    initial_weight = 56.7
    result1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=initial_weight,
        timestamp=datetime(2013, 3, 30),
        source="test",
        processing_config=config["processing"],
        kalman_config=config["kalman"],
        db=db
    )
    
    assert result1 is not None
    assert result1.get('accepted', False) == True
    assert abs(result1['filtered_weight'] - initial_weight) < 0.1
    
    weight_after_gap = 88.0
    result2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=weight_after_gap,
        timestamp=datetime(2017, 2, 27),  # 1430 days later
        source="test",
        processing_config=config["processing"],
        kalman_config=config["kalman"],
        db=db
    )
    
    assert result2 is not None
    assert result2.get('accepted', False) == True, f"Weight after gap should be accepted, got: {result2.get('reason')}"
    assert abs(result2['filtered_weight'] - weight_after_gap) < 0.1
    
    print("✓ Gap reset triggers before validation")
    print(f"  Initial: {initial_weight}kg accepted")
    print(f"  After 1430-day gap: {weight_after_gap}kg accepted (55% change)")


def test_continuous_validation_still_works():
    """Test that validation still rejects invalid continuous measurements."""
    
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    
    db = ProcessorStateDB()
    user_id = "test_continuous_validation"
    
    weight1 = 70.0
    result1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=weight1,
        timestamp=datetime(2024, 1, 1),
        source="test",
        processing_config=config["processing"],
        kalman_config=config["kalman"],
        db=db
    )
    
    assert result1 is not None
    assert result1.get('accepted', False) == True
    
    weight2 = 150.0  # >50% change in one day
    result2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=weight2,
        timestamp=datetime(2024, 1, 2),  # Only 1 day later
        source="test",
        processing_config=config["processing"],
        kalman_config=config["kalman"],
        db=db
    )
    
    assert result2 is not None
    assert result2.get('accepted', False) == False, "Large change without gap should be rejected"
    assert "validation" in result2.get('reason', '').lower()
    
    print("✓ Continuous validation still working")
    print(f"  Day 1: {weight1}kg accepted")
    print(f"  Day 2: {weight2}kg rejected (>50% change without gap)")


def test_moderate_gap_with_reasonable_change():
    """Test that moderate gaps with reasonable changes work."""
    
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    
    db = ProcessorStateDB()
    user_id = "test_moderate_gap"
    
    weight1 = 70.0
    result1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=weight1,
        timestamp=datetime(2024, 1, 1),
        source="test",
        processing_config=config["processing"],
        kalman_config=config["kalman"],
        db=db
    )
    
    assert result1 is not None
    assert result1.get('accepted', False) == True
    
    weight2 = 75.0  # Reasonable 7% gain
    result2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=weight2,
        timestamp=datetime(2024, 1, 20),  # 19 days later (under reset threshold)
        source="test",
        processing_config=config["processing"],
        kalman_config=config["kalman"],
        db=db
    )
    
    assert result2 is not None
    assert result2.get('accepted', False) == True, "Reasonable change should be accepted"
    
    print("✓ Moderate gap with reasonable change accepted")
    print(f"  Day 1: {weight1}kg accepted")
    print(f"  Day 20: {weight2}kg accepted (7% change, no reset)")


if __name__ == "__main__":
    print("Testing gap reset fix...")
    print("-" * 50)
    
    test_gap_reset_before_validation()
    print()
    
    test_continuous_validation_still_works()
    print()
    
    test_moderate_gap_with_reasonable_change()
    print()
    
    print("-" * 50)
    print("All tests passed! ✓")
    print("\nSummary:")
    print("1. Long gaps (>30 days) trigger state reset, allowing large weight changes")
    print("2. Continuous measurements still validate against 50% change threshold")
    print("3. Moderate gaps with reasonable changes work correctly")
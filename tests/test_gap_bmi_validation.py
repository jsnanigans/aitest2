"""
Test the new gap-based BMI validation logic.
Ensures that after gaps, only BMI validation is used, not deviation checks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from src.processor import WeightProcessor
from src.database import ProcessorStateDB

def test_gap_accepts_valid_bmi_large_change():
    """Test that after a gap, large weight changes are accepted if BMI is valid."""
    print("\n=== Test: Gap Accepts Valid BMI with Large Change ===")
    
    db = ProcessorStateDB()
    user_id = "test_gap_bmi_user"
    
    processing_config = {
        "extreme_threshold": 0.1,
        "user_height_m": 1.7,
        "min_valid_bmi": 10.0,
        "max_valid_bmi": 60.0,
        "skip_deviation_after_gap_days": 30
    }
    
    kalman_config = {
        "reset_gap_days": 30,
        "questionnaire_reset_days": 10
    }
    
    print("\nStep 1: Establish baseline at 60kg")
    result1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=60.0,
        timestamp=datetime(2024, 1, 1),
        source="device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    assert result1["accepted"] == True
    print(f"✓ Baseline: {result1['filtered_weight']:.1f}kg")
    
    print("\nStep 2: After 35-day gap, submit 100kg (BMI 34.6 - valid)")
    result2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=100.0,
        timestamp=datetime(2024, 2, 5),
        source="device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    print(f"Result: accepted={result2['accepted']}, reason={result2.get('reason', 'N/A')}")
    print(f"Gap days: {result2.get('gap_days', 0):.1f}")
    print(f"Was reset: {result2.get('was_reset', False)}")
    
    assert result2["accepted"] == True, f"Should accept 100kg after gap (valid BMI), got: {result2.get('reason')}"
    assert result2.get("was_reset") == True, "Should have reset Kalman filter"
    assert result2["filtered_weight"] == 100.0, "Should accept new weight as baseline"
    
    print("✓ 100kg accepted after gap despite 67% change (BMI valid)")
    
    print("\nStep 3: Without gap, extreme change should be rejected")
    result3 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=150.0,
        timestamp=datetime(2024, 2, 6),
        source="device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    print(f"Result 3: accepted={result3['accepted']}, reason={result3.get('reason', 'N/A')}")
    assert result3["accepted"] == False, f"Should reject 50% change without gap, got: {result3}"
    print(f"✓ 150kg rejected (no gap): {result3['reason']}")


def test_gap_rejects_invalid_bmi():
    """Test that after a gap, weights producing invalid BMI are rejected."""
    print("\n=== Test: Gap Rejects Invalid BMI ===")
    
    db = ProcessorStateDB()
    user_id = "test_gap_invalid_bmi"
    
    processing_config = {
        "extreme_threshold": 0.1,
        "user_height_m": 1.7,
        "min_valid_bmi": 10.0,
        "max_valid_bmi": 60.0
    }
    
    kalman_config = {
        "reset_gap_days": 30
    }
    
    print("\nStep 1: Establish baseline at 70kg")
    result1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=70.0,
        timestamp=datetime(2024, 1, 1),
        source="device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    assert result1["accepted"] == True
    print(f"✓ Baseline: {result1['filtered_weight']:.1f}kg")
    
    print("\nStep 2: After 35-day gap, submit 25kg (BMI 8.7 - too low)")
    result2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=25.0,
        timestamp=datetime(2024, 2, 5),
        source="device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    print(f"Result: accepted={result2['accepted']}, reason={result2.get('reason', 'N/A')}")
    bmi = 25.0 / (1.7 ** 2)
    print(f"BMI would be: {bmi:.1f}")
    
    assert result2["accepted"] == False, "Should reject 25kg (BMI too low)"
    assert "BMI" in result2.get("reason", ""), "Should mention BMI in rejection"
    print(f"✓ 25kg rejected after gap: {result2['reason']}")
    
    print("\nStep 3: After gap, submit 180kg (BMI 62.3 - too high)")
    result3 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=180.0,
        timestamp=datetime(2024, 3, 15),
        source="device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    bmi = 180.0 / (1.7 ** 2)
    print(f"BMI would be: {bmi:.1f}")
    
    assert result3["accepted"] == False, "Should reject 180kg (BMI too high)"
    assert "BMI" in result3.get("reason", ""), "Should mention BMI in rejection"
    print(f"✓ 180kg rejected after gap: {result3['reason']}")


def test_user_0040872d_scenario():
    """Test the specific scenario from user 0040872d."""
    print("\n=== Test: User 0040872d Scenario ===")
    
    db = ProcessorStateDB()
    user_id = "0040872d_test"
    
    processing_config = {
        "extreme_threshold": 0.1,
        "user_height_m": 1.75,
        "min_valid_bmi": 10.0,
        "max_valid_bmi": 60.0
    }
    
    kalman_config = {
        "reset_gap_days": 30
    }
    
    print("\nEstablish baseline around 56.7kg")
    result1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=56.7,
        timestamp=datetime(2024, 1, 1),
        source="connectivehealth",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    assert result1["accepted"] == True
    print(f"✓ Baseline: {result1['filtered_weight']:.1f}kg")
    
    print("\nAfter 40-day gap, submit 100kg (should be accepted if BMI valid)")
    result2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=100.0,
        timestamp=datetime(2024, 2, 10),
        source="iglucose",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    bmi = 100.0 / (1.75 ** 2)
    print(f"BMI: {bmi:.1f} (valid range: 10-60)")
    print(f"Result: accepted={result2['accepted']}")
    print(f"Gap days: {result2.get('gap_days', 0):.1f}")
    
    assert result2["accepted"] == True, f"Should accept 100kg after gap (BMI {bmi:.1f} is valid)"
    assert result2.get("was_reset") == True, "Should have reset"
    print("✓ 100kg accepted and baseline reset!")
    
    print("\nNow subsequent 100kg measurements should be accepted")
    result3 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=102.0,
        timestamp=datetime(2024, 2, 11),
        source="iglucose",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    assert result3["accepted"] == True, "Should accept similar weight"
    print(f"✓ 102kg accepted (close to new baseline)")


def test_user_08f3aee0e_scenario():
    """Test the specific scenario from user 08f3aee0e."""
    print("\n=== Test: User 08f3aee0e Scenario ===")
    
    db = ProcessorStateDB()
    user_id = "08f3aee0e_test"
    
    processing_config = {
        "extreme_threshold": 0.1,
        "user_height_m": 1.8,
        "min_valid_bmi": 10.0,
        "max_valid_bmi": 60.0
    }
    
    kalman_config = {
        "reset_gap_days": 30
    }
    
    print("\nEstablish baseline around 67.6kg")
    result1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=67.6,
        timestamp=datetime(2024, 1, 1),
        source="connectivehealth",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    assert result1["accepted"] == True
    print(f"✓ Baseline: {result1['filtered_weight']:.1f}kg")
    
    print("\nAfter 35-day gap, submit 115kg")
    result2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=115.0,
        timestamp=datetime(2024, 2, 5),
        source="questionnaire",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    bmi = 115.0 / (1.8 ** 2)
    print(f"BMI: {bmi:.1f} (valid range: 10-60)")
    print(f"Result: accepted={result2['accepted']}")
    
    assert result2["accepted"] == True, f"Should accept 115kg after gap (BMI {bmi:.1f} is valid)"
    assert result2.get("was_reset") == True, "Should have reset"
    print("✓ 115kg accepted and baseline reset!")


def test_no_height_fallback():
    """Test behavior when height is not available."""
    print("\n=== Test: No Height Fallback ===")
    
    db = ProcessorStateDB()
    user_id = "test_no_height"
    
    processing_config = {
        "extreme_threshold": 0.1,
    }
    
    kalman_config = {
        "reset_gap_days": 30
    }
    
    print("\nEstablish baseline at 70kg")
    result1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=70.0,
        timestamp=datetime(2024, 1, 1),
        source="device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    assert result1["accepted"] == True
    
    print("\nAfter gap, test absolute bounds (no height for BMI)")
    
    print("Test 1: 250kg (should reject - above 300kg absolute max)")
    result2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=350.0,
        timestamp=datetime(2024, 2, 5),
        source="device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    assert result2["accepted"] == False, "Should reject >300kg even without height"
    print(f"✓ Rejected: {result2.get('reason')}")
    
    print("\nTest 2: 150kg (should accept - within absolute bounds)")
    result3 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=150.0,
        timestamp=datetime(2024, 3, 10),
        source="device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    assert result3["accepted"] == True, "Should accept 150kg without height (within absolute bounds)"
    print("✓ 150kg accepted (no height, but within absolute bounds)")


if __name__ == "__main__":
    test_gap_accepts_valid_bmi_large_change()
    test_gap_rejects_invalid_bmi()
    test_user_0040872d_scenario()
    test_user_08f3aee0e_scenario()
    test_no_height_fallback()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nKey improvements verified:")
    print("1. After gaps, only BMI validation is used (not deviation)")
    print("2. Large but valid weight changes are accepted after gaps")
    print("3. Invalid BMI values are still rejected")
    print("4. Users 0040872d and 08f3aee0e scenarios fixed")
    print("5. Fallback to absolute bounds when height unavailable")
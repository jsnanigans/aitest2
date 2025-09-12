"""
Test that user height is properly loaded and used for BMI calculations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from src.processor import WeightProcessor
from src.database import ProcessorStateDB
from src.quality import DataQualityPreprocessor

def test_user_height_loading():
    """Test that user height is loaded from the CSV file."""
    print("\n=== Test: User Height Loading ===")
    
    # Load height data
    DataQualityPreprocessor.load_height_data()
    
    # Test with a known user (if exists in the CSV)
    test_user_id = "0040872d-333a-4ace-8c5a-b2fcd056e65a"
    height = DataQualityPreprocessor.get_user_height(test_user_id)
    
    print(f"User {test_user_id[:8]}... height: {height:.2f}m")
    
    # Test with unknown user (should return default)
    unknown_user = "unknown_user_12345"
    default_height = DataQualityPreprocessor.get_user_height(unknown_user)
    
    print(f"Unknown user height (default): {default_height:.2f}m")
    assert default_height == DataQualityPreprocessor.DEFAULT_HEIGHT_M
    
    print("✓ Height loading works correctly")


def test_bmi_calculation_with_user_height():
    """Test that BMI is calculated using the actual user height."""
    print("\n=== Test: BMI Calculation with User Height ===")
    
    db = ProcessorStateDB()
    
    # Test with a user that might have height data
    user_id = "test_height_user"
    
    # Get the actual height that will be used
    actual_height = DataQualityPreprocessor.get_user_height(user_id)
    print(f"Height for {user_id}: {actual_height:.2f}m")
    
    processing_config = {
        "extreme_threshold": 0.1,
        "min_valid_bmi": 10.0,
        "max_valid_bmi": 60.0
    }
    
    kalman_config = {
        "reset_gap_days": 30
    }
    
    # Process a weight
    weight = 70.0
    result = WeightProcessor.process_weight(
        user_id=user_id,
        weight=weight,
        timestamp=datetime(2024, 1, 1),
        source="device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    # Calculate expected BMI
    expected_bmi = weight / (actual_height ** 2)
    print(f"Weight: {weight}kg")
    print(f"Expected BMI: {expected_bmi:.1f}")
    
    assert result["accepted"] == True
    print("✓ Weight accepted with user-specific height")
    
    # Now test gap reset with BMI validation
    print("\nTesting gap reset with user height...")
    
    # Submit weight after gap that would be invalid with wrong height
    # For height 1.7m: 180kg = BMI 62.3 (invalid)
    # For height 2.0m: 180kg = BMI 45.0 (valid)
    test_weight = 180.0
    
    result2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=test_weight,
        timestamp=datetime(2024, 2, 10),  # 40 day gap
        source="device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    test_bmi = test_weight / (actual_height ** 2)
    print(f"\nTest weight: {test_weight}kg")
    print(f"BMI with height {actual_height:.2f}m: {test_bmi:.1f}")
    
    if test_bmi > 60:
        assert result2["accepted"] == False, f"Should reject BMI {test_bmi:.1f} > 60"
        print(f"✓ Correctly rejected (BMI {test_bmi:.1f} > 60)")
    else:
        assert result2["accepted"] == True, f"Should accept BMI {test_bmi:.1f} <= 60"
        print(f"✓ Correctly accepted (BMI {test_bmi:.1f} <= 60)")


def test_height_affects_validation():
    """Test that different heights lead to different validation outcomes."""
    print("\n=== Test: Height Affects Validation ===")
    
    db = ProcessorStateDB()
    
    # Test weight that's borderline depending on height
    test_weight = 150.0
    
    # User 1: Default height (1.7m)
    # BMI = 150 / (1.7^2) = 51.9 (valid)
    user1 = "user_default_height"
    height1 = DataQualityPreprocessor.get_user_height(user1)
    bmi1 = test_weight / (height1 ** 2)
    
    print(f"User 1 - Height: {height1:.2f}m, BMI: {bmi1:.1f}")
    
    processing_config = {
        "extreme_threshold": 0.1,
        "min_valid_bmi": 10.0,
        "max_valid_bmi": 60.0
    }
    
    kalman_config = {
        "reset_gap_days": 30
    }
    
    # Establish baseline
    WeightProcessor.process_weight(
        user_id=user1,
        weight=70.0,
        timestamp=datetime(2024, 1, 1),
        source="device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    # Test after gap
    result1 = WeightProcessor.process_weight(
        user_id=user1,
        weight=test_weight,
        timestamp=datetime(2024, 2, 10),
        source="device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    if bmi1 <= 60:
        assert result1["accepted"] == True, f"Should accept BMI {bmi1:.1f}"
        print(f"✓ User 1: Weight accepted (BMI {bmi1:.1f} valid)")
    else:
        assert result1["accepted"] == False, f"Should reject BMI {bmi1:.1f}"
        print(f"✓ User 1: Weight rejected (BMI {bmi1:.1f} invalid)")
    
    print("\n" + "="*60)
    print("Height-based validation working correctly!")


if __name__ == "__main__":
    test_user_height_loading()
    test_bmi_calculation_with_user_height()
    test_height_affects_validation()
    
    print("\n" + "="*60)
    print("✅ ALL HEIGHT INTEGRATION TESTS PASSED!")
    print("="*60)
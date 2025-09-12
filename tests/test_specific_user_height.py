"""
Test with specific user 0040872d who has height data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from src.processor import WeightProcessor
from src.database import ProcessorStateDB
from src.quality import DataQualityPreprocessor

def test_user_0040872d_with_actual_height():
    """Test user 0040872d with their actual height from the CSV."""
    print("\n=== Test: User 0040872d with Actual Height ===")
    
    db = ProcessorStateDB()
    user_id = "0040872d-333a-4ace-8c5a-b2fcd056e65a"
    
    # Load and get actual height
    DataQualityPreprocessor.load_height_data()
    actual_height = DataQualityPreprocessor.get_user_height(user_id)
    print(f"User {user_id[:8]}... actual height: {actual_height:.2f}m")
    
    processing_config = {
        "extreme_threshold": 0.1,
        "min_valid_bmi": 10.0,
        "max_valid_bmi": 60.0
    }
    
    kalman_config = {
        "reset_gap_days": 30
    }
    
    # Establish baseline at 56.7kg (from the images)
    print("\nEstablishing baseline at 56.7kg...")
    result1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=56.7,
        timestamp=datetime(2024, 1, 1),
        source="connectivehealth",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    baseline_bmi = 56.7 / (actual_height ** 2)
    print(f"Baseline BMI: {baseline_bmi:.1f}")
    assert result1["accepted"] == True
    print("✓ Baseline established")
    
    # After gap, test 100kg (from the images)
    print("\nAfter 40-day gap, testing 100kg...")
    result2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=100.0,
        timestamp=datetime(2024, 2, 10),
        source="iglucose",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    test_bmi = 100.0 / (actual_height ** 2)
    print(f"100kg with height {actual_height:.2f}m = BMI {test_bmi:.1f}")
    
    if test_bmi <= 60:
        assert result2["accepted"] == True, f"Should accept BMI {test_bmi:.1f}"
        print(f"✓ 100kg accepted (BMI {test_bmi:.1f} is valid)")
    else:
        assert result2["accepted"] == False, f"Should reject BMI {test_bmi:.1f}"
        print(f"✓ 100kg rejected (BMI {test_bmi:.1f} exceeds limit)")
    
    # Test extreme low weight
    print("\nTesting extreme low weight (25kg)...")
    result3 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=25.0,
        timestamp=datetime(2024, 3, 25),
        source="device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    low_bmi = 25.0 / (actual_height ** 2)
    print(f"25kg with height {actual_height:.2f}m = BMI {low_bmi:.1f}")
    
    assert result3["accepted"] == False, "Should reject 25kg (too low)"
    print(f"✓ 25kg correctly rejected")
    
    print("\n" + "="*60)
    print(f"User {user_id[:8]}... validation with actual height working!")
    print(f"Height: {actual_height:.2f}m")
    print(f"Valid weight range for BMI 10-60: {10 * actual_height**2:.1f}-{60 * actual_height**2:.1f}kg")
    print("="*60)

if __name__ == "__main__":
    test_user_0040872d_with_actual_height()
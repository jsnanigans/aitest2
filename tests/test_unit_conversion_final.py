"""Test that unit conversion works correctly with explicit units."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from datetime import datetime
from processor_enhanced import DataQualityPreprocessor

def test_unit_conversion_with_explicit_units():
    """Test unit conversion with explicit units as in the dataset."""
    
    # Test case from the actual data
    test_cases = [
        # (weight, unit, expected_kg, description)
        (170.097, 'kg', 170.097, "170.097 kg should remain unchanged"),
        (170.097, 'lb', 77.15463842, "170.097 lb should convert to ~77.15 kg"),
        (170.097, 'lbs', 77.15463842, "170.097 lbs should convert to ~77.15 kg"),
        (100.0, 'kg', 100.0, "100 kg should remain unchanged"),
        (220.0, 'lb', 99.79024, "220 lb should convert to ~99.79 kg"),
        (15.0, 'st', 95.25435, "15 stones should convert to ~95.25 kg"),
        (80.5, 'kg', 80.5, "80.5 kg should remain unchanged"),
    ]
    
    source = 'internal-questionnaire'
    timestamp = datetime(2024, 9, 17)
    user_id = '1a452430-7351-4b8c-b921-4fb17f8a29cc'
    
    all_passed = True
    
    for weight, unit, expected, description in test_cases:
        cleaned_weight, metadata = DataQualityPreprocessor.preprocess(
            weight, source, timestamp, user_id, unit
        )
        
        print(f"\nTest: {description}")
        print(f"  Input: {weight} {unit}")
        print(f"  Output: {cleaned_weight} kg")
        print(f"  Expected: {expected} kg")
        print(f"  Corrections: {metadata.get('corrections', [])}")
        
        if cleaned_weight is None:
            print(f"  ❌ FAILED: Weight was rejected!")
            print(f"  Rejection reason: {metadata.get('rejected')}")
            all_passed = False
        elif abs(cleaned_weight - expected) > 0.01:
            print(f"  ❌ FAILED: Expected {expected}, got {cleaned_weight}")
            all_passed = False
        else:
            print(f"  ✓ PASSED")
    
    # Special test: BMI detection should still work for kg values
    bmi_value = 25.0  # Looks like BMI
    cleaned_bmi, bmi_metadata = DataQualityPreprocessor.preprocess(
        bmi_value, source, timestamp, user_id, 'kg'
    )
    
    print(f"\nBMI Detection Test:")
    print(f"  Input: {bmi_value} kg")
    print(f"  Output: {cleaned_bmi} kg")
    print(f"  Warnings: {bmi_metadata.get('warnings', [])}")
    print(f"  Corrections: {bmi_metadata.get('corrections', [])}")
    
    if cleaned_bmi != bmi_value:
        print(f"  Note: BMI detection converted {bmi_value} to {cleaned_bmi} kg")
    
    return all_passed

if __name__ == "__main__":
    success = test_unit_conversion_with_explicit_units()
    if success:
        print("\n✅ All unit conversion tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)

"""Test that the unit conversion bug is fixed."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from datetime import datetime
from processor_enhanced import DataQualityPreprocessor

def test_unit_conversion_fix():
    """Test that 170.097 kg from internal-questionnaire is not converted."""
    
    # The problematic case - with explicit unit
    weight = 170.097
    source = 'internal-questionnaire'
    timestamp = datetime(2024, 9, 17)
    user_id = '1a452430-7351-4b8c-b921-4fb17f8a29cc'
    unit = 'kg'  # Explicitly marked as kg
    
    # Process the weight with unit information
    cleaned_weight, metadata = DataQualityPreprocessor.preprocess(
        weight, source, timestamp, user_id, unit
    )
    
    print(f"Test 1: Explicit kg unit")
    print(f"  Original weight: {weight} {unit}")
    print(f"  Source: {source}")
    print(f"  Cleaned weight: {cleaned_weight} kg")
    print(f"  Corrections: {metadata.get('corrections', [])}")
    print(f"  Warnings: {metadata.get('warnings', [])}")
    
    assert cleaned_weight == weight, f"Weight should not be converted when unit is kg! Got {cleaned_weight}"
    print("  ✓ PASSED: Weight not converted when unit is kg\n")
    
    # Test 2: Same value but marked as pounds
    weight_lb = 170.097
    unit_lb = 'lb'
    
    cleaned_weight_lb, metadata_lb = DataQualityPreprocessor.preprocess(
        weight_lb, source, timestamp, user_id, unit_lb
    )
    
    print(f"Test 2: Explicit lb unit")
    print(f"  Original weight: {weight_lb} {unit_lb}")
    print(f"  Source: {source}")
    print(f"  Cleaned weight: {cleaned_weight_lb} kg")
    print(f"  Corrections: {metadata_lb.get('corrections', [])}")
    
    expected_kg = weight_lb * 0.453592
    assert abs(cleaned_weight_lb - expected_kg) < 0.01, f"Weight should be converted from lb to kg! Got {cleaned_weight_lb}"
    print("  ✓ PASSED: Weight correctly converted from lb to kg\n")
    
    # Test 3: Ambiguous case (no unit specified) - should not convert valid kg values
    weight_no_unit = 170.097
    
    cleaned_weight_no_unit, metadata_no_unit = DataQualityPreprocessor.preprocess(
        weight_no_unit, source, timestamp, user_id, ''  # No unit specified
    )
    
    print(f"Test 3: No unit specified (ambiguous)")
    print(f"  Original weight: {weight_no_unit}")
    print(f"  Source: {source}")
    print(f"  Cleaned weight: {cleaned_weight_no_unit} kg")
    print(f"  Corrections: {metadata_no_unit.get('corrections', [])}")
    print(f"  Warnings: {metadata_no_unit.get('warnings', [])}")
    
    # 170.097 is in both kg range (40-200) and pound range (80-450)
    # Since it's in the typical kg range, it should NOT be converted
    assert cleaned_weight_no_unit == weight_no_unit, f"Ambiguous weight in kg range should not be converted! Got {cleaned_weight_no_unit}"
    print("  ✓ PASSED: Ambiguous weight in kg range not converted\n")
    
    # Test 4: Clear pound value without unit (should be converted)
    weight_clear_lb = 250.0  # Clearly pounds (too high for kg)
    
    cleaned_weight_clear_lb, metadata_clear_lb = DataQualityPreprocessor.preprocess(
        weight_clear_lb, source, timestamp, user_id, ''  # No unit specified
    )
    
    print(f"Test 4: Clear pound value without unit")
    print(f"  Original weight: {weight_clear_lb}")
    print(f"  Source: {source}")
    print(f"  Cleaned weight: {cleaned_weight_clear_lb} kg")
    print(f"  Corrections: {metadata_clear_lb.get('corrections', [])}")
    
    expected_clear_kg = weight_clear_lb * 0.453592
    assert abs(cleaned_weight_clear_lb - expected_clear_kg) < 0.01, f"Clear pound value should be converted! Got {cleaned_weight_clear_lb}"
    print("  ✓ PASSED: Clear pound value correctly converted\n")
    
    print("All tests passed! ✓")
    return True

if __name__ == "__main__":
    test_unit_conversion_fix()

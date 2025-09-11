"""Test to demonstrate and fix the unit conversion bug."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from datetime import datetime
from processor_enhanced import DataQualityPreprocessor

def test_unit_conversion_bug():
    """Test that 170.097 kg from internal-questionnaire is not converted."""
    
    # The problematic case
    weight = 170.097
    source = 'internal-questionnaire'
    timestamp = datetime(2024, 9, 17)
    user_id = '1a452430-7351-4b8c-b921-4fb17f8a29cc'
    
    # Process the weight
    cleaned_weight, metadata = DataQualityPreprocessor.preprocess(
        weight, source, timestamp, user_id
    )
    
    print(f"Original weight: {weight} kg")
    print(f"Source: {source}")
    print(f"Cleaned weight: {cleaned_weight} kg")
    print(f"Corrections: {metadata.get('corrections', [])}")
    print(f"Warnings: {metadata.get('warnings', [])}")
    
    # The bug: 170.097 kg is being treated as pounds
    # because it falls in the range [80, 450]
    # But it's already in kg!
    
    if cleaned_weight != weight:
        print(f"\nBUG DETECTED: Weight was incorrectly converted!")
        print(f"Expected: {weight} kg")
        print(f"Got: {cleaned_weight} kg")
        return False
    
    return True

if __name__ == "__main__":
    test_unit_conversion_bug()

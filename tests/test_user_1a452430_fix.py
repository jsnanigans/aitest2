"""Test that user 1a452430-7351-4b8c-b921-4fb17f8a29cc's data is processed correctly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from datetime import datetime
from processor_enhanced import DataQualityPreprocessor

def test_user_1a452430():
    """Test the specific case mentioned by the user."""
    
    # The exact data from the CSV
    user_id = '1a452430-7351-4b8c-b921-4fb17f8a29cc'
    weight = 170.097
    timestamp = datetime(2024, 9, 17)
    source = 'internal-questionnaire'
    unit = 'kg'
    
    print(f"Testing user {user_id}")
    print(f"Input: {weight} {unit} from {source}")
    
    # Test the preprocessing step directly
    cleaned_weight, metadata = DataQualityPreprocessor.preprocess(
        weight, source, timestamp, user_id, unit
    )
    
    print(f"\nPreprocessing Result:")
    print(f"  Original weight: {metadata.get('original_weight')} {metadata.get('original_unit')}")
    print(f"  Cleaned weight: {cleaned_weight} kg")
    print(f"  Corrections: {metadata.get('corrections', [])}")
    print(f"  Warnings: {metadata.get('warnings', [])}")
    
    # Check that weight was NOT converted
    if cleaned_weight == weight:
        print(f"\n✅ SUCCESS: Weight {weight} kg was correctly preserved!")
        print("The bug has been fixed - kg values are no longer incorrectly converted to pounds.")
        return True
    else:
        print(f"\n❌ FAILURE: Weight was incorrectly changed from {weight} to {cleaned_weight}")
        return False

if __name__ == "__main__":
    success = test_user_1a452430()
    if not success:
        exit(1)

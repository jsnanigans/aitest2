"""
Test BMI detection and conversion functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from datetime import datetime
from processor_enhanced import DataQualityPreprocessor


def test_bmi_detection():
    """Test BMI detection with various inputs."""
    
    print("\n=== Testing BMI Detection ===\n")
    
    # Load height data
    DataQualityPreprocessor.load_height_data()
    
    test_cases = [
        # (weight, source, user_id, description)
        (25.0, 'https://connectivehealth.io', None, "BMI value from connectivehealth"),
        (22.5, 'patient-upload', None, "Possible BMI from patient"),
        (150.0, 'patient-upload', None, "Normal weight in pounds"),
        (68.0, 'patient-device', None, "Normal weight in kg"),
        (28.3, 'https://connectivehealth.io', None, "BMI from connectivehealth"),
        (18.5, 'internal-questionnaire', None, "Low BMI or very low weight"),
        (35.0, 'care-team-upload', None, "High BMI or low weight"),
        (450.0, 'patient-upload', None, "Very high weight (pounds)"),
        (10.0, 'patient-device', None, "Impossibly low weight"),
        (600.0, 'patient-device', None, "Impossibly high weight"),
    ]
    
    timestamp = datetime.now()
    
    for weight, source, user_id, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Input: {weight} from {source}")
        
        cleaned_weight, metadata = DataQualityPreprocessor.preprocess(
            weight, source, timestamp, user_id
        )
        
        if cleaned_weight is None:
            print(f"REJECTED: {metadata.get('rejected')}")
        else:
            print(f"Output: {cleaned_weight:.1f} kg")
            if metadata.get('implied_bmi'):
                print(f"Implied BMI: {metadata['implied_bmi']}")
        
        if metadata.get('corrections'):
            for correction in metadata['corrections']:
                print(f"  - {correction}")
        
        if metadata.get('warnings'):
            for warning in metadata['warnings']:
                print(f"  ⚠ {warning}")
        
        print("-" * 50)


def test_with_actual_user_heights():
    """Test BMI detection with actual user heights from the CSV."""
    
    print("\n=== Testing with Actual User Heights ===\n")
    
    # Load height data
    DataQualityPreprocessor.load_height_data()
    
    # Get a sample of user IDs from the loaded data
    if DataQualityPreprocessor._height_data:
        sample_users = list(DataQualityPreprocessor._height_data.keys())[:5]
        
        for user_id in sample_users:
            height_m = DataQualityPreprocessor.get_user_height(user_id)
            print(f"\nUser: {user_id}")
            print(f"Height: {height_m:.2f}m")
            
            # Test with a BMI value
            bmi_value = 25.0
            cleaned_weight, metadata = DataQualityPreprocessor.preprocess(
                bmi_value, 'https://connectivehealth.io', datetime.now(), user_id
            )
            
            if cleaned_weight:
                print(f"BMI {bmi_value} → Weight {cleaned_weight:.1f}kg")
                print(f"Calculated BMI check: {cleaned_weight / (height_m ** 2):.1f}")
            
            # Test with normal weight
            normal_weight = 70.0
            cleaned_weight, metadata = DataQualityPreprocessor.preprocess(
                normal_weight, 'patient-device', datetime.now(), user_id
            )
            
            if cleaned_weight:
                print(f"Weight {normal_weight}kg → BMI {metadata.get('implied_bmi', 'N/A')}")
    else:
        print("No height data loaded")


def test_edge_cases():
    """Test edge cases for BMI detection."""
    
    print("\n=== Testing Edge Cases ===\n")
    
    timestamp = datetime.now()
    
    edge_cases = [
        (15.0, 'patient-device', "Exactly at BMI lower bound"),
        (50.0, 'patient-device', "Exactly at BMI upper bound"),
        (14.9, 'patient-device', "Just below BMI range"),
        (50.1, 'patient-device', "Just above BMI range"),
        (9.9, 'patient-device', "Below impossible BMI"),
        (100.1, 'patient-device', "Above impossible BMI"),
        (12.9, 'patient-device', "Just below suspicious BMI"),
        (60.1, 'patient-device', "Just above suspicious BMI"),
    ]
    
    for weight, source, description in edge_cases:
        print(f"\n{description}: {weight}")
        cleaned_weight, metadata = DataQualityPreprocessor.preprocess(
            weight, source, timestamp
        )
        
        if cleaned_weight is None:
            print(f"  → REJECTED: {metadata.get('rejected')}")
        else:
            print(f"  → Accepted: {cleaned_weight:.1f}kg")
            if metadata.get('implied_bmi'):
                print(f"  → BMI: {metadata['implied_bmi']}")


if __name__ == "__main__":
    test_bmi_detection()
    test_with_actual_user_heights()
    test_edge_cases()
    
    print("\n=== All BMI Detection Tests Complete ===")
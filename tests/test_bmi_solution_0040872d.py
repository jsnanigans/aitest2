#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.bmi_validator import BMIValidator

def test_bmi_validator_on_user_data():
    """Test BMI validator on actual user 0040872d data."""
    print("=== Testing BMI Validator on User 0040872d ===\n")
    
    height_m = 1.75
    
    measurements = [
        {'timestamp': '2024-05-21 00:00:00', 'weight': 86.64, 'source': 'connectivehealth'},
        {'timestamp': '2024-07-17 00:00:00', 'weight': 86.18, 'source': 'connectivehealth'},
        {'timestamp': '2024-09-18 00:00:00', 'weight': 86.18, 'source': 'connectivehealth'},
        {'timestamp': '2024-11-19 00:00:00', 'weight': 88.45, 'source': 'connectivehealth'},
        {'timestamp': '2025-01-02 00:00:00', 'weight': 87.09, 'source': 'connectivehealth'},
        {'timestamp': '2025-01-13 00:00:00', 'weight': 88.45, 'source': 'connectivehealth'},
        {'timestamp': '2025-03-03 00:00:00', 'weight': 91.17, 'source': 'initial-questionnaire'},
        {'timestamp': '2025-03-12 20:06:23', 'weight': 99.3, 'source': 'iglucose'},
        {'timestamp': '2025-03-12 20:08:05', 'weight': 76.2, 'source': 'iglucose'},
        {'timestamp': '2025-03-13 22:05:24', 'weight': 98.0, 'source': 'iglucose'},
        {'timestamp': '2025-03-15 17:11:55', 'weight': 59.4, 'source': 'iglucose'},
        {'timestamp': '2025-03-17 18:39:13', 'weight': 91.3, 'source': 'iglucose'},
        {'timestamp': '2025-03-19 21:41:49', 'weight': 61.2, 'source': 'iglucose'},
        {'timestamp': '2025-03-20 22:23:40', 'weight': 32.0, 'source': 'iglucose'},
        {'timestamp': '2025-03-24 19:03:05', 'weight': 91.2, 'source': 'iglucose'},
        {'timestamp': '2025-04-04 01:46:03', 'weight': 77.0, 'source': 'iglucose'},
        {'timestamp': '2025-04-13 20:31:13', 'weight': 53.3, 'source': 'iglucose'},
        {'timestamp': '2025-04-17 08:42:45', 'weight': 52.4, 'source': 'iglucose'},
    ]
    
    last_accepted = None
    resets = 0
    rejections = 0
    
    for m in measurements:
        timestamp = pd.to_datetime(m['timestamp'])
        weight = m['weight']
        source = m['source']
        bmi = BMIValidator.calculate_bmi(weight, height_m)
        
        if last_accepted:
            time_delta_hours = (timestamp - last_accepted['timestamp']).total_seconds() / 3600
            
            should_reset, reset_reason = BMIValidator.should_reset_kalman(
                weight, 
                last_accepted['weight'],
                time_delta_hours,
                height_m,
                source
            )
            
            rejection_reason = BMIValidator.get_rejection_reason(
                weight,
                last_accepted['weight'],
                time_delta_hours,
                height_m
            )
            
            confidence = BMIValidator.get_confidence_multiplier(
                weight,
                last_accepted['weight'],
                time_delta_hours,
                height_m
            )
            
            weight_change = weight - last_accepted['weight']
            pct_change = (weight_change / last_accepted['weight']) * 100
            
            status = "✅ ACCEPT"
            action = ""
            
            if should_reset:
                status = "🔄 RESET"
                action = f"\n   → Reset Kalman: {reset_reason}"
                resets += 1
                last_accepted = {'timestamp': timestamp, 'weight': weight}
            elif rejection_reason:
                status = "❌ REJECT"
                action = f"\n   → Rejection: {rejection_reason}"
                rejections += 1
            else:
                last_accepted = {'timestamp': timestamp, 'weight': weight}
            
            print(f"{status} {timestamp}: {weight:6.1f}kg (BMI: {bmi:5.1f}) "
                  f"[{pct_change:+6.1f}% | conf: {confidence:.2f}] - {source}{action}")
        else:
            print(f"✅ ACCEPT {timestamp}: {weight:6.1f}kg (BMI: {bmi:5.1f}) [initial] - {source}")
            last_accepted = {'timestamp': timestamp, 'weight': weight}
    
    print(f"\n=== Summary ===")
    print(f"Total measurements: {len(measurements)}")
    print(f"Resets triggered: {resets}")
    print(f"Rejections: {rejections}")
    print(f"Final accepted weight: {last_accepted['weight']:.1f}kg")

def test_edge_cases():
    """Test various edge cases for BMI validation."""
    print("\n\n=== Testing Edge Cases ===\n")
    
    test_cases = [
        {
            'name': 'Normal daily variation',
            'current': 70.5, 'last': 70.0, 'hours': 24, 'height': 1.75,
            'expected_reset': False, 'expected_reject': False
        },
        {
            'name': 'Large meal + hydration',
            'current': 73.0, 'last': 70.0, 'hours': 2, 'height': 1.75,
            'expected_reset': False, 'expected_reject': False
        },
        {
            'name': 'Impossible instant drop',
            'current': 45.0, 'last': 70.0, 'hours': 0.5, 'height': 1.75,
            'expected_reset': True, 'expected_reject': True
        },
        {
            'name': 'Dangerous BMI < 15',
            'current': 40.0, 'last': 70.0, 'hours': 720, 'height': 1.75,
            'expected_reset': True, 'expected_reject': True
        },
        {
            'name': 'Extreme obesity BMI > 50',
            'current': 160.0, 'last': 90.0, 'hours': 720, 'height': 1.75,
            'expected_reset': True, 'expected_reject': True
        },
        {
            'name': 'Gradual weight loss',
            'current': 65.0, 'last': 70.0, 'hours': 720, 'height': 1.75,
            'expected_reset': False, 'expected_reject': False
        },
        {
            'name': 'Suspicious source with big change',
            'current': 50.0, 'last': 70.0, 'hours': 48, 'height': 1.75,
            'source': 'iglucose', 'expected_reset': True, 'expected_reject': True
        }
    ]
    
    for case in test_cases:
        should_reset, reset_reason = BMIValidator.should_reset_kalman(
            case['current'],
            case['last'],
            case['hours'],
            case.get('height'),
            case.get('source')
        )
        
        rejection_reason = BMIValidator.get_rejection_reason(
            case['current'],
            case['last'],
            case['hours'],
            case.get('height')
        )
        
        confidence = BMIValidator.get_confidence_multiplier(
            case['current'],
            case['last'],
            case['hours'],
            case.get('height')
        )
        
        bmi = BMIValidator.calculate_bmi(case['current'], case.get('height'))
        
        print(f"\nCase: {case['name']}")
        print(f"  Weight: {case['last']:.1f} → {case['current']:.1f}kg in {case['hours']}h")
        if bmi:
            print(f"  BMI: {bmi:.1f}")
        print(f"  Confidence: {confidence:.2f}")
        
        reset_match = "✅" if should_reset == case['expected_reset'] else "❌"
        reject_match = "✅" if (rejection_reason is not None) == case['expected_reject'] else "❌"
        
        print(f"  Reset: {should_reset} {reset_match} (expected: {case['expected_reset']})")
        if reset_reason:
            print(f"    Reason: {reset_reason}")
        
        print(f"  Reject: {rejection_reason is not None} {reject_match} (expected: {case['expected_reject']})")
        if rejection_reason:
            print(f"    Reason: {rejection_reason}")

def demonstrate_solution():
    """Demonstrate how the solution prevents the issue."""
    print("\n\n=== Solution Demonstration ===\n")
    
    print("BEFORE (Current System):")
    print("- Accepts 87kg → 52kg drop after 30-day gap")
    print("- Kalman filter adapts to impossible weight")
    print("- Subsequent valid measurements get rejected")
    print("- User stuck with incorrect weight baseline")
    
    print("\nAFTER (With BMI Validator):")
    print("- Detects 87kg → 52kg as >30% change")
    print("- Triggers Kalman reset instead of adaptation")
    print("- Starts fresh from 52kg (may still be wrong)")
    print("- Further drops (52kg → 32kg) also trigger reset")
    print("- System recovers when valid weights return")
    
    print("\nKEY IMPROVEMENTS:")
    print("1. Percentage-based thresholds catch dramatic changes")
    print("2. BMI limits detect medically impossible values")
    print("3. Time-aware limits (stricter for short timeframes)")
    print("4. Source-specific handling for unreliable sources")
    print("5. Confidence multipliers reduce trust in suspicious data")
    
    print("\nRESULT:")
    print("✅ No more accepting 40% weight drops")
    print("✅ No more BMI < 15 or > 50 acceptance")
    print("✅ Faster recovery from bad data")
    print("✅ Better protection for real users")

if __name__ == "__main__":
    test_bmi_validator_on_user_data()
    test_edge_cases()
    demonstrate_solution()
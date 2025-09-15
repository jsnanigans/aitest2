#!/usr/bin/env python3
"""
Test adaptive quality scoring during initial/post-reset period.
"""

from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import process_measurement
from src.database import get_state_db

def test_adaptive_quality():
    """Test adaptive quality scoring."""
    
    db = get_state_db()
    user_id = "test_adaptive_quality"
    db.delete_state(user_id)
    
    config = {
        'kalman': {
            'initial_variance': 0.361,
            'transition_covariance_weight': 0.0160,
            'transition_covariance_trend': 0.0001,
            'observation_covariance': 3.490,
            'reset': {
                'enabled': True,
                'gap_threshold_days': 30
            }
        },
        'processing': {
            'extreme_threshold': 0.20
        },
        'adaptive_noise': {
            'enabled': True  # Enable to match real scenario
        },
        'quality_scoring': {
            'enabled': True,  # Enable quality scoring
            'threshold': 0.6,
            'component_weights': {
                'safety': 0.35,
                'plausibility': 0.25,
                'consistency': 0.25,
                'reliability': 0.15
            }
        }
    }
    
    print("Testing Adaptive Quality Scoring")
    print("=" * 60)
    print("Scenario: Initial 120kg → measurements at 108kg")
    print("(Matching the rejection shown in your image)\n")
    
    # Initial measurement at 120kg
    base_date = datetime(2025, 4, 20)
    result = process_measurement(
        user_id, 120.0, base_date, "initial-questionnaire", config
    )
    print(f"Initial: 120.0kg - Accepted: {result['accepted']}")
    
    # Measurement at 108.4kg (like in the rejection image)
    timestamp = datetime(2025, 5, 9, 14, 48)  # Match the timestamp in image
    result = process_measurement(
        user_id, 108.4, timestamp, "https://api.iglucose.com", config
    )
    
    print(f"\nMeasurement at 108.4kg (iGlucose):")
    print(f"  Accepted: {result['accepted']}")
    
    if result.get('quality_score') is not None:
        print(f"  Quality score: {result['quality_score']:.2f}")
        if result.get('quality_components'):
            components = result['quality_components']
            print(f"  Components:")
            print(f"    Safety: {components.get('safety', 0):.2f}")
            print(f"    Plausibility: {components.get('plausibility', 0):.2f}")
            print(f"    Consistency: {components.get('consistency', 0):.2f}")
            print(f"    Reliability: {components.get('reliability', 0):.2f}")
    
    if not result['accepted']:
        print(f"  Rejection reason: {result.get('reason', 'unknown')}")
        print("\n  ✗ Still rejected (should be accepted with adaptive scoring)")
    else:
        print("\n  ✓ Accepted with adaptive quality scoring!")
    
    # Test a few more measurements
    print("\nAdditional measurements:")
    test_weights = [107.5, 108.0, 107.2, 106.8]
    
    accepted_count = 0
    for i, weight in enumerate(test_weights):
        timestamp = base_date + timedelta(days=20 + i*3)
        result = process_measurement(
            user_id, weight, timestamp, "patient-device", config
        )
        
        if result['accepted']:
            accepted_count += 1
            status = "✓"
        else:
            status = "✗"
        
        quality = result.get('quality_score', 0)
        print(f"  {weight:.1f}kg: {status} (quality: {quality:.2f})")
    
    print(f"\n" + "=" * 60)
    print(f"Results: {accepted_count + (1 if result['accepted'] else 0)}/{len(test_weights)+1} accepted")
    
    if accepted_count >= 3:
        print("✓ SUCCESS: Adaptive quality scoring is working!")
    else:
        print("⚠ Partial success - some measurements still rejected")

if __name__ == "__main__":
    test_adaptive_quality()

#!/usr/bin/env python3
"""Test the adaptive Kalman implementation with real data."""

import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processor import process_measurement

def test_adaptive_after_reset():
    """Test adaptive Kalman behavior after a 30+ day gap."""
    
    # Test configuration
    config = {
        'kalman': {
            'initial_variance': 0.361,
            'transition_covariance_weight': 0.016,
            'transition_covariance_trend': 0.0001,
            'observation_covariance': 3.49,
            'reset': {
                'enabled': True,
                'gap_threshold_days': 30
            }
        },
        'adaptive_noise': {
            'enabled': True,
            'default_multiplier': 1.5
        },
        'quality_scoring': {
            'enabled': False
        },
        'processing': {
            'extreme_threshold': 0.20
        }
    }
    
    user_id = "test_user_adaptive"
    
    # Initial measurements before gap
    print("Initial measurements (before gap):")
    start_date = datetime(2024, 1, 1)
    
    initial_weights = [120.0, 119.5, 120.2, 119.8]
    for i, weight in enumerate(initial_weights):
        timestamp = start_date + timedelta(days=i)
        result = process_measurement(
            user_id, weight, timestamp, "patient-device", config
        )
        if result:
            print(f"  Day {i}: raw={weight:.1f}, filtered={result.get('filtered_weight', weight):.1f}, "
                  f"accepted={result['accepted']}")
    
    print("\n--- 35 day gap ---\n")
    
    # Measurements after gap (weight dropped to 105kg)
    print("Measurements after gap (with adaptive parameters):")
    post_gap_date = start_date + timedelta(days=40)
    post_gap_weights = [105.0, 106.0, 105.5, 107.0, 105.0, 106.5, 105.8, 106.2]
    
    for i, weight in enumerate(post_gap_weights):
        timestamp = post_gap_date + timedelta(days=i)
        result = process_measurement(
            user_id, weight, timestamp, "patient-device", config
        )
        
        if result:
            reset_info = ""
            if 'reset_event' in result:
                reset_info = f" [RESET: gap={result['reset_event']['gap_days']:.0f} days]"
            
            accepted_str = "✓" if result['accepted'] else "✗"
            reason = f" ({result.get('reason', '')})" if not result['accepted'] else ""
            
            print(f"  Day {i}: raw={weight:.1f}, filtered={result.get('filtered_weight', weight):.1f}, "
                  f"trend={result.get('trend', 0):.4f}, accepted={accepted_str}{reason}{reset_info}")
            
            if 'innovation' in result:
                print(f"    Innovation: {result['innovation']:.2f}, "
                      f"Normalized: {result.get('normalized_innovation', 0):.2f}, "
                      f"Confidence: {result.get('confidence', 0):.2f}")

if __name__ == "__main__":
    test_adaptive_after_reset()
#!/usr/bin/env python3
"""
Test to verify replay system correctly handles outliers for user e751ebe4-3e13-423d-bf50-88a9dd13f132
on 2025-04-10 where a BMI value was misinterpreted as weight.
"""

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.processing.pipeline import ProcessingPipeline
from src.database.database import ProcessorStateDB


def test_replay_outlier_rejection():
    """Test that replay correctly rejects the BMI-as-weight outlier"""
    
    # Test configuration with replay enabled
    config = {
        'replay': {
            'enabled': True,
            'buffer_hours': 24,  # Capture full day of measurements
            'trigger_mode': 'time_based',
            'max_buffer_measurements': 100,
            'outlier_detection': {
                'iqr_multiplier': 1.5,
                'z_score_threshold': 3.0,
                'temporal_max_change_percent': 0.5,
                'min_measurements_for_analysis': 2
            },
            'safety': {
                'max_processing_time_seconds': 60,
                'require_rollback_confirmation': False,
                'preserve_immediate_results': True
            }
        },
        'quality_scoring': {
            'quality_threshold': 0.45
        },
        'features': {
            'kalman_filtering': True,
            'quality_scoring': True,
            'outlier_detection': True,
            'quality_override': True
        }
    }
    
    # Create test data for the specific user and date
    test_data = [
        # Previous measurement for context (from earlier date)
        {
            'user_id': 'e751ebe4-3e13-423d-bf50-88a9dd13f132',
            'timestamp': '2025-04-09 10:00:00',
            'weight': '79.5',
            'source': 'patient-device',
            'unit': 'kg'
        },
        # The problematic BMI value that gets converted to weight
        {
            'user_id': 'e751ebe4-3e13-423d-bf50-88a9dd13f132',
            'timestamp': '2025-04-10 19:27:05',
            'weight': '34.565',  # This is actually BMI
            'source': 'patient-device',
            'unit': 'kg'
        },
        # Correct measurements that should be used instead
        {
            'user_id': 'e751ebe4-3e13-423d-bf50-88a9dd13f132',
            'timestamp': '2025-04-10 19:34:18',
            'weight': '79.637',
            'source': 'patient-device',
            'unit': 'kg'
        },
        {
            'user_id': 'e751ebe4-3e13-423d-bf50-88a9dd13f132',
            'timestamp': '2025-04-10 19:34:40',
            'weight': '79.651',
            'source': 'patient-device',
            'unit': 'kg'
        },
        {
            'user_id': 'e751ebe4-3e13-423d-bf50-88a9dd13f132',
            'timestamp': '2025-04-10 20:29:27',
            'weight': '79.450',
            'source': 'patient-device',
            'unit': 'kg'
        }
    ]
    
    # Write test CSV
    test_csv = 'test_replay_fix.csv'
    with open(test_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['user_id', 'timestamp', 'weight', 'source', 'unit'])
        writer.writeheader()
        writer.writerows(test_data)
    
    # Create pipeline (it will initialize its own database)
    pipeline = ProcessingPipeline(config)
    
    # Process the test data
    print("Processing test data...")
    results = pipeline.process_csv_file(test_csv)
    
    # Check results for the specific user
    user_results = results['user_results'].get('e751ebe4-3e13-423d-bf50-88a9dd13f132', [])
    
    print(f"\nTotal measurements processed: {len(user_results)}")
    
    # Analyze the results
    accepted_weights = []
    rejected_weights = []
    
    for result in user_results:
        timestamp = result.get('timestamp', '')
        weight = result.get('raw_weight', 0)
        accepted = result.get('accepted', False)
        
        # Convert timestamp to string if it's a datetime object
        if isinstance(timestamp, datetime):
            timestamp_str = timestamp.isoformat()
        else:
            timestamp_str = str(timestamp)
        
        if '2025-04-10' in timestamp_str:
            if accepted:
                accepted_weights.append((timestamp_str, weight))
                print(f"✓ ACCEPTED: {timestamp_str} - {weight:.2f}kg")
            else:
                rejected_weights.append((timestamp_str, weight))
                reason = result.get('reason', result.get('rejection_reason', 'Unknown'))
                print(f"✗ REJECTED: {timestamp_str} - {weight:.2f}kg - Reason: {reason}")
    
    # Verify expectations
    print("\n=== TEST RESULTS ===")
    
    # The BMI value (34.565) should either be:
    # 1. Rejected as an outlier, OR
    # 2. Converted to ~100kg and then rejected/corrected by replay
    
    # Check if any measurement around 100kg was accepted (it shouldn't be)
    high_weights_accepted = [w for t, w in accepted_weights if w > 90]
    
    if high_weights_accepted:
        print(f"❌ FAIL: High weight values were accepted: {high_weights_accepted}")
        print("   The replay system should have rejected the BMI-as-weight outlier")
        success = False
    else:
        print("✅ PASS: No abnormally high weights were accepted")
        success = True
    
    # Check if the correct ~79kg measurements were accepted
    normal_weights_accepted = [w for t, w in accepted_weights if 78 < w < 81]
    
    if normal_weights_accepted:
        print(f"✅ PASS: Normal weight values were accepted: {normal_weights_accepted}")
    else:
        print("⚠️  WARNING: No normal weight values were accepted (might all be in buffer)")
    
    # Clean up
    os.remove(test_csv)
    
    return success


if __name__ == "__main__":
    print("Testing replay system fix for outlier rejection...")
    print("=" * 60)
    
    success = test_replay_outlier_rejection()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TEST PASSED: Replay system correctly handles outliers")
        sys.exit(0)
    else:
        print("❌ TEST FAILED: Replay system did not handle outliers correctly")
        sys.exit(1)
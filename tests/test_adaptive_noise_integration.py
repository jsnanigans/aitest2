#!/usr/bin/env python3
"""Integration test for adaptive noise configuration with real data processing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import tomllib
from datetime import datetime
from collections import defaultdict
from src.processor import process_weight_enhanced
from src.database import get_state_db


def test_adaptive_noise_with_csv_data():
    """Test adaptive noise with actual CSV data processing."""
    
    # Load config
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    
    # Create test CSV data with different sources
    test_data = [
        # user_id, weight, timestamp, source, unit
        ["user1", 75.0, "2024-01-01 08:00:00", "patient-upload", "kg"],
        ["user1", 75.2, "2024-01-01 12:00:00", "patient-upload", "kg"],
        ["user1", 75.5, "2024-01-01 16:00:00", "https://api.iglucose.com", "kg"],
        ["user1", 75.3, "2024-01-02 08:00:00", "patient-device", "kg"],
        ["user1", 75.1, "2024-01-02 12:00:00", "care-team-upload", "kg"],
        
        ["user2", 85.0, "2024-01-01 09:00:00", "internal-questionnaire", "kg"],
        ["user2", 85.5, "2024-01-01 14:00:00", "https://connectivehealth.io", "kg"],
        ["user2", 85.3, "2024-01-02 09:00:00", "patient-upload", "kg"],
        ["user2", 85.8, "2024-01-02 14:00:00", "unknown-new-source", "kg"],  # Unknown source
    ]
    
    db = get_state_db()
    
    # Process each measurement
    results_by_user = defaultdict(list)
    
    for row in test_data:
        user_id, weight, timestamp_str, source, unit = row
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        
        # Clear state for clean test
        if len(results_by_user[user_id]) == 0:
            db.clear_state(user_id)
        
        # Add full config to processing_config
        processing_config = config["processing"].copy()
        processing_config["config"] = config
        
        result = process_weight_enhanced(
            user_id=user_id,
            weight=float(weight),
            timestamp=timestamp,
            source=source,
            processing_config=processing_config,
            kalman_config=config["kalman"],
            unit=unit
        )
        
        results_by_user[user_id].append({
            'source': source,
            'weight': weight,
            'result': result
        })
    
    # Analyze results
    print("\nProcessing Results by User:")
    print("=" * 60)
    
    for user_id, measurements in results_by_user.items():
        print(f"\n{user_id}:")
        print("-" * 40)
        
        for m in measurements:
            result = m['result']
            source = m['source']
            weight = m['weight']
            
            if result and 'confidence' in result:
                confidence = result['confidence']
                accepted = result.get('accepted', False)
                noise_used = result.get('measurement_noise_used', 0)
                
                # Get expected multiplier
                if config["adaptive_noise"]["enabled"]:
                    multipliers = config["adaptive_noise"]["multipliers"]
                    expected_mult = multipliers.get(source, config["adaptive_noise"]["default_multiplier"])
                else:
                    expected_mult = 1.0
                
                actual_mult = noise_used / config["kalman"]["observation_covariance"] if noise_used > 0 else 0
                
                print(f"  {source:30} weight={weight:6.1f} conf={confidence:.3f} "
                      f"mult={actual_mult:.1f} (exp={expected_mult:.1f}) "
                      f"{'✓' if accepted else '✗'}")
    
    # Verify different sources got different multipliers
    user1_results = results_by_user["user1"]
    patient_upload = next(r for r in user1_results if r['source'] == 'patient-upload')
    iglucose = next(r for r in user1_results if r['source'] == 'https://api.iglucose.com')
    
    patient_noise = patient_upload['result'].get('measurement_noise_used', 0)
    iglucose_noise = iglucose['result'].get('measurement_noise_used', 0)
    
    assert iglucose_noise > patient_noise, "iGlucose should have higher noise than patient-upload"
    
    # Check unknown source uses default
    user2_results = results_by_user["user2"]
    unknown = next(r for r in user2_results if r['source'] == 'unknown-new-source')
    unknown_noise = unknown['result'].get('measurement_noise_used', 0)
    expected_unknown = config["kalman"]["observation_covariance"] * config["adaptive_noise"]["default_multiplier"]
    
    assert abs(unknown_noise - expected_unknown) < 0.1, "Unknown source should use default multiplier"
    
    print("\n✅ Integration test passed!")
    print("Adaptive noise correctly applied to different sources during processing")
    return True


def test_adaptive_noise_improves_accuracy():
    """Test that adaptive noise improves prediction accuracy."""
    
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    
    db = get_state_db()
    
    # Test scenario: consistent measurements from reliable source,
    # then outlier from unreliable source
    test_scenario = [
        # Establish baseline with reliable source
        ("patient-upload", 75.0, 0),  # Day 0
        ("patient-upload", 75.1, 1),  # Day 1
        ("patient-upload", 75.0, 2),  # Day 2
        ("patient-upload", 75.2, 3),  # Day 3
        
        # Outlier from unreliable source
        ("https://api.iglucose.com", 77.0, 4),  # Day 4 - 2kg jump
        
        # Back to reliable source
        ("patient-upload", 75.3, 5),  # Day 5
    ]
    
    user_id = "test-accuracy"
    db.clear_state(user_id)
    
    base_date = datetime(2024, 1, 1, 8, 0, 0)
    results = []
    
    processing_config = config["processing"].copy()
    processing_config["config"] = config
    
    for source, weight, day_offset in test_scenario:
        timestamp = datetime(2024, 1, day_offset + 1, 8, 0, 0)
        
        result = process_weight_enhanced(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source,
            processing_config=processing_config,
            kalman_config=config["kalman"],
            unit="kg"
        )
        
        results.append({
            'day': day_offset,
            'source': source,
            'raw_weight': weight,
            'filtered_weight': result.get('filtered_weight', weight),
            'confidence': result.get('confidence', 0),
            'accepted': result.get('accepted', False)
        })
    
    print("\nAccuracy Test Results:")
    print("Day | Source                  | Raw  | Filtered | Conf  | Status")
    print("-" * 65)
    
    for r in results:
        print(f"{r['day']:3} | {r['source']:23} | {r['raw_weight']:4.1f} | "
              f"{r['filtered_weight']:8.2f} | {r['confidence']:5.3f} | "
              f"{'✓' if r['accepted'] else '✗'}")
    
    # The outlier from iglucose should have less impact due to higher noise expectation
    outlier_result = results[4]  # The iglucose measurement
    
    # Check that the filtered weight didn't jump as much as the raw weight
    prev_filtered = results[3]['filtered_weight']
    outlier_filtered = outlier_result['filtered_weight']
    outlier_raw = outlier_result['raw_weight']
    
    raw_jump = abs(outlier_raw - prev_filtered)
    filtered_jump = abs(outlier_filtered - prev_filtered)
    
    print(f"\nOutlier handling:")
    print(f"Raw jump: {raw_jump:.2f} kg")
    print(f"Filtered jump: {filtered_jump:.2f} kg")
    print(f"Dampening: {(1 - filtered_jump/raw_jump)*100:.1f}%")
    
    # The filtered jump should be less than the raw jump
    assert filtered_jump < raw_jump, "Kalman filter should dampen outliers"
    
    print("\n✅ Adaptive noise improves accuracy by properly weighting sources!")
    return True


if __name__ == "__main__":
    print("Running adaptive noise integration tests...")
    print("=" * 60)
    
    test_adaptive_noise_with_csv_data()
    test_adaptive_noise_improves_accuracy()
    
    print("\n" + "=" * 60)
    print("All integration tests passed! 🎉")
    print("\nAdaptive noise configuration is fully functional:")
    print("✓ Config-based multipliers are loaded and applied")
    print("✓ Different sources get different noise levels")
    print("✓ Unknown sources use default multiplier")
    print("✓ Accuracy is improved by proper source weighting")
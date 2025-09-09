#!/usr/bin/env python3
"""Test Kalman-guided daily cleanup functionality."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, date
from src.reprocessor import WeightReprocessor
from src.processor_database import ProcessorDatabase
import numpy as np

def test_kalman_guided_selection():
    """Test that measurements are selected based on Kalman prediction."""
    
    measurements = [
        {'weight': 75.2, 'timestamp': datetime(2025, 1, 15, 8, 0), 'source': 'scale'},
        {'weight': 80.5, 'timestamp': datetime(2025, 1, 15, 9, 0), 'source': 'manual'},  # Far from prediction
        {'weight': 75.8, 'timestamp': datetime(2025, 1, 15, 18, 0), 'source': 'scale'},
        {'weight': 76.0, 'timestamp': datetime(2025, 1, 15, 20, 0), 'source': 'manual'}
    ]
    
    kalman_prediction = 75.5  # Kalman thinks weight should be 75.5kg
    threshold = 2.0  # 2kg threshold
    
    selected = WeightReprocessor.select_kalman_guided_measurements(
        measurements,
        kalman_prediction=kalman_prediction,
        threshold=threshold
    )
    
    # Should keep measurements within 2kg of 75.5 (73.5-77.5)
    assert len(selected) == 3, f"Expected 3 measurements within threshold, got {len(selected)}"
    
    weights = [m['weight'] for m in selected]
    assert 80.5 not in weights, "Should have rejected 80.5kg (5kg from prediction)"
    assert all(abs(w - kalman_prediction) <= threshold for w in weights), \
        f"All selected should be within {threshold}kg of prediction"
    
    print(f"✓ Kalman-guided selection (prediction={kalman_prediction}kg, threshold={threshold}kg):")
    for m in selected:
        deviation = abs(m['weight'] - kalman_prediction)
        print(f"  Kept: {m['weight']}kg (deviation: {deviation:.1f}kg)")
    
    # Find rejected
    rejected = [m for m in measurements if m not in selected]
    for m in rejected:
        deviation = abs(m['weight'] - kalman_prediction)
        print(f"  Rejected: {m['weight']}kg (deviation: {deviation:.1f}kg)")

def test_tight_threshold():
    """Test with a tight threshold that rejects more measurements."""
    
    measurements = [
        {'weight': 74.0, 'timestamp': datetime(2025, 1, 15, 8, 0), 'source': 'scale'},
        {'weight': 75.0, 'timestamp': datetime(2025, 1, 15, 12, 0), 'source': 'scale'},
        {'weight': 76.0, 'timestamp': datetime(2025, 1, 15, 18, 0), 'source': 'scale'},
        {'weight': 77.0, 'timestamp': datetime(2025, 1, 15, 20, 0), 'source': 'scale'}
    ]
    
    kalman_prediction = 75.0
    threshold = 0.5  # Very tight 0.5kg threshold
    
    selected = WeightReprocessor.select_kalman_guided_measurements(
        measurements,
        kalman_prediction=kalman_prediction,
        threshold=threshold
    )
    
    # Should only keep 75.0kg (exactly at prediction)
    assert len(selected) == 1, f"Expected 1 measurement within tight threshold, got {len(selected)}"
    assert selected[0]['weight'] == 75.0, "Should keep only the measurement at prediction"
    
    print(f"\n✓ Tight threshold (prediction={kalman_prediction}kg, threshold={threshold}kg):")
    print(f"  Kept only: {selected[0]['weight']}kg (perfect match)")

def test_source_priority_with_kalman():
    """Test that source priority still matters when deviations are equal."""
    
    measurements = [
        {'weight': 75.2, 'timestamp': datetime(2025, 1, 15, 8, 0), 'source': 'patient-upload'},      # priority=4
        {'weight': 75.2, 'timestamp': datetime(2025, 1, 15, 12, 0), 'source': 'internal-questionnaire'}, # priority=1
        {'weight': 75.2, 'timestamp': datetime(2025, 1, 15, 18, 0), 'source': 'patient-device'}      # priority=3
    ]
    
    kalman_prediction = 75.0  # All measurements are 0.2kg away
    threshold = 1.0
    
    selected = WeightReprocessor.select_kalman_guided_measurements(
        measurements,
        kalman_prediction=kalman_prediction,
        threshold=threshold
    )
    
    # All should be kept (all within 1kg)
    assert len(selected) == 3, f"All should be within threshold, got {len(selected)}"
    
    # When deviations are equal, internal-questionnaire should be first due to source priority (priority=1)
    assert selected[0]['source'] == 'internal-questionnaire', "Questionnaire should be prioritized when deviations equal"
    assert selected[1]['source'] == 'patient-device', "Patient device should be second (priority=3)"
    assert selected[2]['source'] == 'patient-upload', "Patient upload should be last (priority=4)"
    
    print(f"\n✓ Source priority preserved when deviations equal:")
    for i, m in enumerate(selected):
        priority = WeightReprocessor.SOURCE_PRIORITY.get(m['source'], 999)
        print(f"  {i+1}. {m['weight']}kg from {m['source']} (priority={priority})")

def test_extreme_outlier_always_rejected():
    """Test that extreme outliers are rejected regardless of Kalman prediction."""
    
    measurements = [
        {'weight': 75.0, 'timestamp': datetime(2025, 1, 15, 8, 0), 'source': 'scale'},
        {'weight': 750.0, 'timestamp': datetime(2025, 1, 15, 9, 0), 'source': 'manual'},  # 10x error
        {'weight': 7.5, 'timestamp': datetime(2025, 1, 15, 10, 0), 'source': 'manual'},   # 0.1x error
    ]
    
    kalman_prediction = 75.0
    threshold = 1000.0  # Even with huge threshold
    
    selected = WeightReprocessor.select_kalman_guided_measurements(
        measurements,
        kalman_prediction=kalman_prediction,
        threshold=threshold
    )
    
    # Should only keep the valid weight
    assert len(selected) == 1, f"Should reject extreme outliers, got {len(selected)}"
    assert selected[0]['weight'] == 75.0, "Should keep only physiologically valid weight"
    
    print(f"\n✓ Extreme outliers rejected (even with threshold={threshold}kg):")
    print(f"  Kept: {selected[0]['weight']}kg")
    print(f"  Rejected: 750.0kg and 7.5kg (outside 30-400kg bounds)")

def test_fallback_when_all_outside_threshold():
    """Test that we keep the closest measurement when all are outside threshold."""
    
    measurements = [
        {'weight': 80.0, 'timestamp': datetime(2025, 1, 15, 8, 0), 'source': 'scale'},
        {'weight': 85.0, 'timestamp': datetime(2025, 1, 15, 12, 0), 'source': 'manual'},
        {'weight': 90.0, 'timestamp': datetime(2025, 1, 15, 18, 0), 'source': 'manual'}
    ]
    
    kalman_prediction = 75.0  # All measurements are far from prediction
    threshold = 2.0
    
    selected = WeightReprocessor.select_kalman_guided_measurements(
        measurements,
        kalman_prediction=kalman_prediction,
        threshold=threshold
    )
    
    # Should keep the closest one (80.0kg)
    assert len(selected) == 1, f"Should keep closest when all outside threshold, got {len(selected)}"
    assert selected[0]['weight'] == 80.0, "Should keep the closest measurement"
    
    print(f"\n✓ Fallback to closest (all outside {threshold}kg threshold):")
    print(f"  Kalman prediction: {kalman_prediction}kg")
    print(f"  Kept closest: {selected[0]['weight']}kg (5kg away)")

def test_process_daily_batch_with_kalman():
    """Test the full daily batch processing with Kalman state."""
    
    db = ProcessorDatabase()
    user_id = "test_user"
    
    # Set up a Kalman state
    state = {
        'initialized': True,
        'last_state': np.array([75.5, 0.01]),  # Weight=75.5kg, trend=0.01
        'last_covariance': np.eye(2),
        'last_timestamp': datetime(2025, 1, 14, 23, 59)
    }
    db.save_state(user_id, state)
    
    measurements = [
        {'weight': 75.2, 'timestamp': datetime(2025, 1, 15, 8, 0), 'source': 'scale'},
        {'weight': 80.5, 'timestamp': datetime(2025, 1, 15, 9, 0), 'source': 'manual'},  # Outlier
        {'weight': 75.8, 'timestamp': datetime(2025, 1, 15, 18, 0), 'source': 'scale'}
    ]
    
    config = {
        'kalman_cleanup_threshold': 2.0,
        'max_daily_change': 0.05
    }
    
    result = WeightReprocessor.process_daily_batch(
        user_id=user_id,
        batch_date=date(2025, 1, 15),
        measurements=measurements,
        processing_config=config,
        kalman_config={},
        db=db
    )
    
    assert result['reason'] == 'kalman_guided_filtering', "Should use Kalman-guided filtering"
    assert result['kalman_prediction'] == 75.5, "Should include Kalman prediction"
    assert len(result['selected']) == 2, f"Should select 2 measurements, got {len(result['selected'])}"
    assert len(result['rejected']) == 1, f"Should reject 1 measurement, got {len(result['rejected'])}"
    
    rejected_weight = result['rejected'][0]['weight']
    assert rejected_weight == 80.5, f"Should reject 80.5kg outlier, rejected {rejected_weight}kg"
    
    print(f"\n✓ Full daily batch with Kalman state:")
    print(f"  Kalman prediction: {result['kalman_prediction']}kg")
    print(f"  Selected: {[m['weight'] for m in result['selected']]}")
    print(f"  Rejected: {[m['weight'] for m in result['rejected']]}")

if __name__ == "__main__":
    test_kalman_guided_selection()
    test_tight_threshold()
    test_source_priority_with_kalman()
    test_extreme_outlier_always_rejected()
    test_fallback_when_all_outside_threshold()
    test_process_daily_batch_with_kalman()
    print("\n✅ All Kalman-guided cleanup tests passed!")
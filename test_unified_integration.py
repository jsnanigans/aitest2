#!/usr/bin/env python
"""Test unified quality scorer integration with processor"""

from datetime import datetime, timedelta
from src.processing.processor import process_measurement
from src.database.database import get_state_db
from src.feature_manager import FeatureManager

def test_unified_scorer_integration():
    """Test that unified scorer is properly integrated with processor"""

    # Setup config
    config = {
        'processing': {
            'extreme_threshold': 0.20,
            'config': {}
        },
        'kalman': {
            'transition_covariance_weight': 0.01,
            'transition_covariance_trend': 0.0001,
            'observation_covariance': 3.49,
            'reset': {'gap_threshold_days': 30}
        },
        'quality_scoring': {
            'threshold': 0.6,
            'component_weights': {
                'kalman_fit': 0.40,
                'temporal_consistency': 0.20,
                'anomaly_detection': 0.20,
                'source_reliability': 0.10,
                'trend_alignment': 0.10
            }
        }
    }

    # Initialize feature manager
    feature_manager = FeatureManager(config)
    config['feature_manager'] = feature_manager

    # Clear any existing state
    db = get_state_db()
    db.delete_state('test_user')

    print("\nTesting unified quality scorer integration:")
    print("=" * 60)

    # Test 1: Initial measurement (should accept)
    result1 = process_measurement(
        user_id='test_user',
        weight=75.0,
        timestamp=datetime(2024, 1, 1, 8, 0),
        source='patient-device',
        config=config
    )
    print(f"\n1. Initial measurement: weight=75.0")
    print(f"   Accepted: {result1['accepted']}")
    print(f"   Stage: {result1.get('stage', 'N/A')}")
    if result1['accepted']:
        print(f"   Filtered weight: {result1.get('filtered_weight', 'N/A'):.1f}")

    # Test 2: Normal variation (should accept)
    result2 = process_measurement(
        user_id='test_user',
        weight=75.5,
        timestamp=datetime(2024, 1, 2, 8, 0),
        source='patient-device',
        config=config
    )
    print(f"\n2. Normal variation: weight=75.5")
    print(f"   Accepted: {result2['accepted']}")
    if result2['accepted']:
        print(f"   Quality score: {result2.get('quality_score', 'N/A'):.2f}")
        if 'quality_components' in result2:
            print(f"   Components: {result2['quality_components']}")

    # Test 3: Large jump - different user scenario (should reject)
    result3 = process_measurement(
        user_id='test_user',
        weight=95.0,  # 20kg jump
        timestamp=datetime(2024, 1, 3, 8, 0),
        source='patient-device',
        config=config
    )
    print(f"\n3. Large jump (different user): weight=95.0")
    print(f"   Accepted: {result3['accepted']}")
    print(f"   Stage: {result3.get('stage', 'N/A')}")
    if not result3['accepted']:
        print(f"   Reason: {result3.get('reason', 'N/A')}")
        score = result3.get('quality_score', 'N/A')
        if score != 'N/A':
            print(f"   Quality score: {score:.2f}")
        else:
            print(f"   Quality score: {score}")

    # Test 4: Return to normal (completing A→B→A pattern)
    result4 = process_measurement(
        user_id='test_user',
        weight=75.3,
        timestamp=datetime(2024, 1, 3, 18, 0),
        source='patient-device',
        config=config
    )
    print(f"\n4. Return to normal: weight=75.3")
    print(f"   Accepted: {result4['accepted']}")
    if result4['accepted']:
        print(f"   Quality score: {result4.get('quality_score', 'N/A'):.2f}")

    # Test 5: Reliable source (care-team)
    result5 = process_measurement(
        user_id='test_user',
        weight=74.8,
        timestamp=datetime(2024, 1, 4, 10, 0),
        source='care-team-upload',
        config=config
    )
    print(f"\n5. Reliable source (care-team): weight=74.8")
    print(f"   Accepted: {result5['accepted']}")
    if result5['accepted']:
        print(f"   Quality score: {result5.get('quality_score', 'N/A'):.2f}")
        if 'quality_components' in result5:
            comp = result5['quality_components']
            print(f"   Source reliability: {comp.get('source_reliability', 'N/A'):.2f}")

    print("\n" + "=" * 60)
    print("Unified scorer integration test complete!")

    # Verify feature is enabled
    if feature_manager.is_enabled('unified_quality_scoring'):
        print("\n✓ Unified quality scoring is ENABLED")
    else:
        print("\n✗ Unified quality scoring is DISABLED")

if __name__ == '__main__':
    test_unified_scorer_integration()
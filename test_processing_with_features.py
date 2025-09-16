#!/usr/bin/env python3
"""
Test that feature toggles actually affect processing behavior.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime
from src.processing.processor import process_measurement
from src.database.database import get_state_db
from src.feature_manager import FeatureManager


def test_processing_without_kalman():
    """Test processing with Kalman filtering disabled."""
    print("\n" + "=" * 60)
    print("TEST: Processing without Kalman filtering")
    print("=" * 60)

    # Create config with Kalman disabled
    config = {
        'features': {
            'kalman_filtering': False,
            'quality_scoring': False,
            'kalman_deviation_check': False
        },
        'feature_manager': FeatureManager({
            'features': {
                'kalman_filtering': False,
                'quality_scoring': False,
                'kalman_deviation_check': False
            }
        })
    }

    db = get_state_db()

    # Process a measurement
    result = process_measurement(
        user_id="test_user_no_kalman",
        weight=70.0,
        timestamp=datetime.now(),
        source="patient-device",
        config=config,
        db=db
    )

    print(f"\nProcessing result:")
    print(f"  Stage: {result.get('stage', 'unknown')}")
    print(f"  Accepted: {result.get('accepted', False)}")
    print(f"  Filtered weight: {result.get('filtered_weight', 0):.1f} kg")
    print(f"  Raw weight: {result.get('cleaned_weight', 0):.1f} kg")

    if result.get('stage') == 'no_filtering':
        print("  ✅ Kalman filtering was correctly skipped!")
    else:
        print(f"  ❌ Expected 'no_filtering' stage but got '{result.get('stage')}'")


def test_processing_without_outlier_detection():
    """Test processing with outlier detection disabled."""
    print("\n" + "=" * 60)
    print("TEST: Processing without outlier detection")
    print("=" * 60)

    # Note: This test is more complex because outlier detection
    # happens in retrospective processing, not in single measurement processing
    config = {
        'features': {
            'outlier_detection': False,
            'quality_override': False
        },
        'feature_manager': FeatureManager({
            'features': {
                'outlier_detection': False,
                'quality_override': False
            }
        }),
        'retrospective': {'outlier_detection': {'enabled': False}}
    }

    # Create outlier detector
    from src.processing.outlier_detection import OutlierDetector
    detector = OutlierDetector(config)

    # Test batch of measurements - including obvious outliers
    measurements = [
        {'weight': 70.0, 'timestamp': datetime(2024, 1, 1, 10, 0), 'metadata': {}},
        {'weight': 70.5, 'timestamp': datetime(2024, 1, 2, 10, 0), 'metadata': {}},
        {'weight': 200.0, 'timestamp': datetime(2024, 1, 3, 10, 0), 'metadata': {}},  # Obvious outlier
        {'weight': 70.2, 'timestamp': datetime(2024, 1, 4, 10, 0), 'metadata': {}},
        {'weight': 69.8, 'timestamp': datetime(2024, 1, 5, 10, 0), 'metadata': {}},
    ]

    outliers = detector.detect_outliers(measurements, user_id="test_user")

    print(f"\nDetected outliers: {outliers}")
    if len(outliers) == 0:
        print("  ✅ No outliers detected (detection disabled)")
    else:
        print(f"  ❌ Outliers detected despite being disabled: {outliers}")


def test_processing_without_quality_scoring():
    """Test processing with quality scoring disabled."""
    print("\n" + "=" * 60)
    print("TEST: Processing without quality scoring")
    print("=" * 60)

    config = {
        'features': {
            'quality_scoring': False,
            'quality_override': False
        },
        'quality_scoring': {'enabled': False},
        'feature_manager': FeatureManager({
            'features': {
                'quality_scoring': False,
                'quality_override': False
            }
        })
    }

    db = get_state_db()

    # Process a measurement with suspicious values
    result = process_measurement(
        user_id="test_user_no_quality",
        weight=25.0,  # Very low weight that quality scorer would flag
        timestamp=datetime.now(),
        source="iglucose-com",  # Unreliable source
        config=config,
        db=db
    )

    print(f"\nProcessing result:")
    print(f"  Accepted: {result.get('accepted', False)}")
    print(f"  Stage: {result.get('stage', 'unknown')}")
    print(f"  Quality score: {result.get('quality_score', 'N/A')}")

    if 'quality_score' not in result or result.get('quality_score') is None:
        print("  ✅ Quality scoring was correctly skipped!")
    else:
        print(f"  ❌ Quality scoring ran despite being disabled")


def test_processing_without_resets():
    """Test processing with reset management disabled."""
    print("\n" + "=" * 60)
    print("TEST: Processing without reset management")
    print("=" * 60)

    config = {
        'features': {
            'resets': {
                'initial': False,
                'hard': False,
                'soft': False
            }
        },
        'feature_manager': FeatureManager({
            'features': {
                'reset_initial': False,
                'reset_hard': False,
                'reset_soft': False
            }
        })
    }

    db = get_state_db()

    # Process first measurement (would normally trigger initial reset)
    result = process_measurement(
        user_id="test_user_no_resets",
        weight=70.0,
        timestamp=datetime.now(),
        source="patient-device",
        config=config,
        db=db
    )

    print(f"\nFirst measurement (no initial reset):")
    print(f"  Accepted: {result.get('accepted', False)}")
    print(f"  Reset event: {result.get('reset_event', 'None')}")

    if 'reset_event' not in result:
        print("  ✅ No reset triggered (resets disabled)")
    else:
        print(f"  ❌ Reset triggered despite being disabled: {result.get('reset_event')}")


def main():
    """Run all processing tests."""
    print("\n🔧 FEATURE TOGGLE PROCESSING TESTS")
    print("Testing that disabled features actually affect processing...")

    test_processing_without_kalman()
    test_processing_without_outlier_detection()
    test_processing_without_quality_scoring()
    test_processing_without_resets()

    print("\n" + "=" * 60)
    print("✅ PROCESSING TESTS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
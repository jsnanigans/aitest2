#!/usr/bin/env python3
"""
Test script to verify quality score override in outlier detection.
"""

from datetime import datetime, timedelta
import sys
import os

# Add src directory to path to import directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import directly to avoid __init__ dependencies
import outlier_detection
OutlierDetector = outlier_detection.OutlierDetector

def test_quality_override():
    """Test that high quality measurements are never marked as outliers."""

    # Create test measurements with some clear outliers
    measurements = [
        {
            'weight': 70.0,
            'timestamp': datetime.now() - timedelta(days=10),
            'metadata': {'quality_score': 0.9, 'accepted': True}  # High quality
        },
        {
            'weight': 71.0,
            'timestamp': datetime.now() - timedelta(days=9),
            'metadata': {'quality_score': 0.85, 'accepted': True}  # High quality
        },
        {
            'weight': 150.0,  # Clear outlier but high quality
            'timestamp': datetime.now() - timedelta(days=8),
            'metadata': {'quality_score': 0.8, 'accepted': True}  # High quality
        },
        {
            'weight': 72.0,
            'timestamp': datetime.now() - timedelta(days=7),
            'metadata': {'quality_score': 0.6, 'accepted': False}  # Low quality
        },
        {
            'weight': 120.0,  # Outlier with low quality
            'timestamp': datetime.now() - timedelta(days=6),
            'metadata': {'quality_score': 0.3, 'accepted': False}  # Low quality
        },
        {
            'weight': 73.0,
            'timestamp': datetime.now() - timedelta(days=5),
            'metadata': {'quality_score': 0.75, 'accepted': True}  # High quality
        },
        {
            'weight': 71.5,
            'timestamp': datetime.now() - timedelta(days=4),
            'metadata': {'quality_score': 0.82, 'accepted': True}  # High quality
        },
        {
            'weight': 200.0,  # Extreme outlier with low quality
            'timestamp': datetime.now() - timedelta(days=3),
            'metadata': {'quality_score': 0.2, 'accepted': False}  # Very low quality
        },
        {
            'weight': 72.5,
            'timestamp': datetime.now() - timedelta(days=2),
            'metadata': {'quality_score': 0.88, 'accepted': True}  # High quality
        },
        {
            'weight': 73.5,
            'timestamp': datetime.now() - timedelta(days=1),
            'metadata': {'quality_score': 0.91, 'accepted': True}  # High quality
        }
    ]

    # Create detector with quality threshold of 0.7
    config = {
        'quality_score_threshold': 0.7,
        'iqr_multiplier': 1.5,
        'z_score_threshold': 2.0
    }
    detector = OutlierDetector(config)

    # Detect outliers
    outlier_indices = detector.detect_outliers(measurements)

    print("Test: Quality Score Override")
    print("=" * 50)
    print(f"Total measurements: {len(measurements)}")
    print(f"Outliers detected: {len(outlier_indices)}")
    print()

    # Check each measurement
    for i, measurement in enumerate(measurements):
        quality_score = measurement['metadata']['quality_score']
        is_outlier = i in outlier_indices
        weight = measurement['weight']

        print(f"Index {i}: Weight={weight:6.1f}, Quality={quality_score:.2f}, Outlier={is_outlier}")

        # Verify high quality measurements are not marked as outliers
        if quality_score > 0.7 and is_outlier:
            print(f"  ERROR: High quality measurement (score={quality_score}) marked as outlier!")
            return False

    print()
    print("Expected outliers (low quality + statistical outliers):")
    print("  Index 4: Weight=120.0 (outlier with quality=0.30)")
    print("  Index 7: Weight=200.0 (extreme outlier with quality=0.20)")
    print()
    print(f"Actual outlier indices: {sorted(outlier_indices)}")

    # Verify that the extreme outliers with low quality are detected
    if 4 not in outlier_indices:
        print("WARNING: Index 4 (120.0kg with quality=0.30) not detected as outlier")
    if 7 not in outlier_indices:
        print("WARNING: Index 7 (200.0kg with quality=0.20) not detected as outlier")

    # Verify that high quality outlier is NOT detected
    if 2 in outlier_indices:
        print("ERROR: Index 2 (150.0kg with quality=0.80) incorrectly marked as outlier")
        return False

    print()
    print("✓ Test PASSED: High quality measurements are protected from outlier detection")
    return True


def test_and_logic():
    """Test that outliers require BOTH low quality AND statistical deviation."""

    measurements = [
        {'weight': 70.0, 'timestamp': datetime.now() - timedelta(days=5), 'metadata': {'quality_score': 0.5}},
        {'weight': 71.0, 'timestamp': datetime.now() - timedelta(days=4), 'metadata': {'quality_score': 0.5}},
        {'weight': 72.0, 'timestamp': datetime.now() - timedelta(days=3), 'metadata': {'quality_score': 0.5}},
        {'weight': 73.0, 'timestamp': datetime.now() - timedelta(days=2), 'metadata': {'quality_score': 0.5}},
        {'weight': 74.0, 'timestamp': datetime.now() - timedelta(days=1), 'metadata': {'quality_score': 0.5}},
        {'weight': 100.0, 'timestamp': datetime.now(), 'metadata': {'quality_score': 0.4}},  # Clear outlier
    ]

    config = {
        'quality_score_threshold': 0.7,
        'min_measurements_for_analysis': 5
    }
    detector = OutlierDetector(config)

    outlier_indices = detector.detect_outliers(measurements)

    print("\nTest: AND Logic for Outlier Detection")
    print("=" * 50)
    print(f"All measurements have low quality scores (0.4-0.5)")
    print(f"Only index 5 (100.0kg) is statistically an outlier")
    print(f"Expected: Only index 5 should be marked as outlier")
    print(f"Actual outlier indices: {sorted(outlier_indices)}")

    if outlier_indices == {5}:
        print("✓ Test PASSED: AND logic working correctly")
        return True
    else:
        print("✗ Test FAILED: Unexpected outlier detection")
        return False


if __name__ == "__main__":
    print("Testing Quality Score Override in Outlier Detection")
    print("=" * 70)

    test1_pass = test_quality_override()
    test2_pass = test_and_logic()

    print()
    print("=" * 70)
    if test1_pass and test2_pass:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
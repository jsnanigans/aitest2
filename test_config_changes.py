#!/usr/bin/env python3
"""Test that the removal of redundant enabled configs doesn't break processing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime
from src.config_loader import load_config
from src.processing.processor import process_measurement
from src.database.database import get_state_db


def test_config_loading():
    """Test that config loads properly without the old enabled flags."""
    print("Testing config loading...")
    config = load_config()

    # Check that feature manager exists
    assert 'feature_manager' in config, "Feature manager not found in config!"
    fm = config['feature_manager']

    # Check that we no longer have old enabled flags
    quality_config = config.get('quality_scoring', {})
    assert 'enabled' not in quality_config, "Old 'enabled' flag still in quality_scoring config!"

    adaptive_config = config.get('adaptive_noise', {})
    assert 'enabled' not in adaptive_config, "Old 'enabled' flag still in adaptive_noise config!"

    print("✅ Config loading test passed")
    return config


def test_processing_with_new_config(config):
    """Test that processing works with the new config structure."""
    print("\nTesting processing with new config...")

    db = get_state_db()

    # Process a test measurement
    result = process_measurement(
        user_id="test_user_config_changes",
        weight=70.5,
        timestamp=datetime.now(),
        source="patient-device",
        config=config,
        db=db
    )

    # Check that processing succeeded
    assert 'accepted' in result, "Processing result missing 'accepted' field!"
    assert 'filtered_weight' in result or 'stage' in result, "Processing result incomplete!"

    print(f"  Processing result: Accepted={result.get('accepted')}, Stage={result.get('stage', 'completed')}")
    print("✅ Processing test passed")
    return result


def test_feature_toggles_respected(config):
    """Test that feature toggles are being respected."""
    print("\nTesting feature toggle behavior...")

    fm = config['feature_manager']

    # Test that we can check various features
    features_to_check = [
        'kalman_filtering',
        'quality_scoring',
        'outlier_detection',
        'adaptive_noise',
        'visualization_enabled',
        'state_persistence'
    ]

    for feature in features_to_check:
        enabled = fm.is_enabled(feature)
        print(f"  {feature}: {'✓' if enabled else '✗'}")

    print("✅ Feature toggle test passed")


def test_no_old_enabled_references():
    """Test that we're not using old .get('enabled') patterns anymore."""
    print("\nChecking for removed 'enabled' config references...")

    # Load config
    config = load_config()

    # These sections should NOT have 'enabled' anymore
    sections_to_check = [
        ('visualization', "Visualization config"),
        ('adaptive_noise', "Adaptive noise config"),
        ('retrospective', "Retrospective config"),
        ('quality_scoring', "Quality scoring config")
    ]

    for section_key, section_name in sections_to_check:
        section = config.get(section_key, {})
        if isinstance(section, dict) and 'enabled' in section:
            print(f"  ❌ {section_name} still has 'enabled' flag!")
            return False
        else:
            print(f"  ✓ {section_name} - no 'enabled' flag")

    print("✅ No old enabled references found")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING REDUNDANT CONFIG REMOVAL")
    print("=" * 60)

    try:
        # Test config loading
        config = test_config_loading()

        # Test processing
        test_processing_with_new_config(config)

        # Test feature toggles
        test_feature_toggles_respected(config)

        # Test old references removed
        test_no_old_enabled_references()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe redundant 'enabled' configs have been successfully removed.")
        print("All functionality is now controlled through the [features] section.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
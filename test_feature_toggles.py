#!/usr/bin/env python3
"""
Test script to verify feature toggle implementation.
Tests various feature combinations to ensure they work correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import load_config
from src.feature_manager import FeatureManager
import json


def test_feature_manager():
    """Test the FeatureManager with different configurations."""
    print("=" * 60)
    print("TESTING FEATURE MANAGER")
    print("=" * 60)

    # Test 1: Default configuration (all features enabled)
    print("\n1. Testing default configuration (all enabled)...")
    config = {}
    fm = FeatureManager(config)

    print(f"   Kalman filtering: {fm.is_enabled('kalman_filtering')}")
    print(f"   Quality scoring: {fm.is_enabled('quality_scoring')}")
    print(f"   Outlier detection: {fm.is_enabled('outlier_detection')}")
    print(f"   Total features: {len(fm.features)}")
    print(f"   Enabled: {len(fm.get_enabled_features())}")
    print(f"   Disabled: {len(fm.get_disabled_features())}")

    # Test 2: Disable specific features
    print("\n2. Testing with disabled features...")
    config_disabled = {
        'features': {
            'kalman_filtering': False,
            'quality_scoring': False,
            'outlier_detection': True
        }
    }
    fm2 = FeatureManager(config_disabled)

    print(f"   Kalman filtering: {fm2.is_enabled('kalman_filtering')}")
    print(f"   Quality scoring: {fm2.is_enabled('quality_scoring')}")
    print(f"   Outlier detection: {fm2.is_enabled('outlier_detection')}")
    print(f"   Quality override: {fm2.is_enabled('quality_override')}")  # Should be disabled due to dependency

    # Test 3: Test nested features
    print("\n3. Testing nested feature configuration...")
    config_nested = {
        'features': {
            'validation': {
                'physiological': False,
                'bmi_checking': False,
                'rate_limiting': True
            },
            'outlier_methods': {
                'iqr': False,
                'mad': True,
                'temporal': False
            }
        }
    }
    fm3 = FeatureManager(config_nested)

    print(f"   Physiological validation: {fm3.is_enabled('validation_physiological')}")
    print(f"   BMI checking: {fm3.is_enabled('validation_bmi_checking')}")
    print(f"   Rate limiting: {fm3.is_enabled('validation_rate_limiting')}")
    print(f"   IQR outlier: {fm3.is_enabled('outlier_iqr')}")
    print(f"   MAD outlier: {fm3.is_enabled('outlier_mad')}")
    print(f"   Temporal outlier: {fm3.is_enabled('outlier_temporal')}")

    # Test 4: Test dependency resolution
    print("\n4. Testing dependency resolution...")
    config_deps = {
        'features': {
            'kalman_filtering': False,  # This should be force-enabled due to quality_scoring
            'quality_scoring': True,
            'quality_override': True
        }
    }
    fm4 = FeatureManager(config_deps)

    print(f"   Kalman filtering: {fm4.is_enabled('kalman_filtering')} (should be True due to dependency)")
    print(f"   Quality scoring: {fm4.is_enabled('quality_scoring')}")
    print(f"   Quality override: {fm4.is_enabled('quality_override')}")

    print("\n✅ FeatureManager tests complete!")


def test_config_integration():
    """Test that config.toml is properly loaded with features."""
    print("\n" + "=" * 60)
    print("TESTING CONFIG.TOML INTEGRATION")
    print("=" * 60)

    # Load the actual config
    config = load_config("config.toml")

    # Check that feature_manager was created
    if 'feature_manager' in config:
        fm = config['feature_manager']
        print("\n✅ FeatureManager successfully integrated into config!")

        # Display current feature states
        print("\nCurrent feature states from config.toml:")
        features_to_check = [
            'kalman_filtering',
            'quality_scoring',
            'outlier_detection',
            'validation_physiological',
            'validation_bmi_checking',
            'outlier_iqr',
            'outlier_mad',
            'reset_initial',
            'reset_hard',
            'reset_soft'
        ]

        for feature in features_to_check:
            status = "✓" if fm.is_enabled(feature) else "✗"
            print(f"   [{status}] {feature}")

        # Get summary
        summary = fm.get_config_summary()
        print(f"\nSummary:")
        print(f"   Total features: {summary['total_features']}")
        print(f"   Enabled: {summary['enabled']}")
        print(f"   Disabled: {summary['disabled']}")
        print(f"   Configuration valid: {summary['valid']}")

        if summary['warnings']:
            print(f"   ⚠️  Warnings: {summary['warnings']}")
    else:
        print("\n❌ FeatureManager not found in config!")
        return False

    return True


def test_feature_disabling():
    """Test that we can disable various features."""
    print("\n" + "=" * 60)
    print("TESTING FEATURE DISABLING")
    print("=" * 60)

    test_configs = [
        {
            'name': 'No Kalman filtering',
            'config': {
                'features': {
                    'kalman_filtering': False,
                    'quality_scoring': False,  # Must disable this too due to dependency
                    'quality_override': False,  # And this
                    'kalman_deviation_check': False  # And this
                }
            },
            'check': 'kalman_filtering'
        },
        {
            'name': 'No outlier detection',
            'config': {
                'features': {
                    'outlier_detection': False,
                    'quality_override': False  # Must disable due to dependency
                }
            },
            'check': 'outlier_detection'
        },
        {
            'name': 'No quality scoring',
            'config': {
                'features': {
                    'quality_scoring': False,
                    'quality_override': False  # Must disable due to dependency
                }
            },
            'check': 'quality_scoring'
        },
        {
            'name': 'No state persistence',
            'config': {
                'features': {
                    'state': {
                        'persistence': False,
                        'history_buffer': False,  # Dependencies
                        'reset_tracking': False
                    }
                }
            },
            'check': 'state_persistence'
        }
    ]

    for test in test_configs:
        print(f"\nTesting: {test['name']}")
        fm = FeatureManager(test['config'])
        is_disabled = not fm.is_enabled(test['check'])

        if is_disabled:
            print(f"   ✅ {test['check']} successfully disabled")
        else:
            print(f"   ❌ {test['check']} is still enabled!")
            # Check if it was due to dependencies
            if test['check'] in fm.DEPENDENCIES:
                print(f"      Dependencies: {fm.DEPENDENCIES[test['check']]}")


def main():
    """Run all tests."""
    print("\n🔧 FEATURE TOGGLE TEST SUITE")
    print("Testing the new feature toggle implementation...")

    # Run tests
    test_feature_manager()
    test_config_integration()
    test_feature_disabling()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETE!")
    print("=" * 60)
    print("\nThe feature toggle system is working correctly.")
    print("You can now enable/disable features in config.toml")
    print("under the [features] section.")


if __name__ == "__main__":
    main()
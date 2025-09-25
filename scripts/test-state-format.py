#!/usr/bin/env python3
"""
Test that Kalman state is properly handled in different formats.
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.replay.enhanced_replay_analyzer import EnhancedReplayAnalyzer


def test_state_formats():
    """Test different state formats that might occur."""

    analyzer = EnhancedReplayAnalyzer({})

    # Test cases with different state formats
    test_cases = [
        # Format 1: Nested list (what was causing the error)
        {"name": "Nested list [[weight]]", "state": [[75.5]], "expected": 75.5},
        # Format 2: Nested list with velocity
        {
            "name": "Nested list [[weight, velocity]]",
            "state": [[75.5, 0.1]],
            "expected": 75.5,
        },
        # Format 3: Flat list
        {"name": "Flat list [weight]", "state": [75.5], "expected": 75.5},
        # Format 4: Numpy 1D array
        {"name": "Numpy 1D array", "state": np.array([75.5]), "expected": 75.5},
        # Format 5: Numpy 2D array (single state)
        {"name": "Numpy 2D array (1x1)", "state": np.array([[75.5]]), "expected": 75.5},
        # Format 6: Numpy 2D array with velocity
        {
            "name": "Numpy 2D array (1x2)",
            "state": np.array([[75.5, 0.1]]),
            "expected": 75.5,
        },
        # Format 7: Multiple states (2D)
        {
            "name": "Numpy 2D array (2x2) - multiple states",
            "state": np.array([[74.0, 0.0], [75.5, 0.1]]),
            "expected": 74.0,  # Should use first row, first element
        },
    ]

    print("Testing state format handling...")
    print("-" * 50)

    all_passed = True

    for test_case in test_cases:
        name = test_case["name"]
        state = test_case["state"]
        expected = test_case["expected"]

        # Create user state with the test state
        user_state = {
            "last_state": state,
            "last_covariance": None,
            "measurement_history": [],
        }

        # Create a dummy measurement
        measurements = [{"timestamp": "2024-01-01T00:00:00", "weight": 75.0}]

        try:
            # This is the method that was failing
            predictions = analyzer._get_kalman_predictions(measurements, user_state)

            # The prediction should be based on the state
            if predictions and len(predictions) > 0:
                actual = predictions[0]
                if (
                    abs(actual - expected) < 0.01
                ):  # Allow small floating point differences
                    print(f"✓ {name}: {actual:.1f}")
                else:
                    print(f"✗ {name}: Expected {expected}, got {actual}")
                    all_passed = False
            else:
                print(f"✗ {name}: No prediction returned")
                all_passed = False

        except Exception as e:
            print(f"✗ {name}: Error - {e}")
            all_passed = False

    print("-" * 50)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")

    return all_passed


if __name__ == "__main__":
    success = test_state_formats()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Integration test for extreme weight change after reset.
Tests that quality scoring correctly rejects impossible changes even after resets.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from src.processing.processor import process_measurement
from src.database.database import get_state_db
import tempfile
import json

def test_extreme_change_after_reset():
    """Test that extreme weight changes are rejected even after resets."""

    try:
        # Use the standard state database
        db = get_state_db()

        # Configuration similar to main.py
        config = {
            'quality_scoring': {
                'enabled': True,
                'threshold': 0.6,
                'use_harmonic_mean': False,
                'component_weights': {
                    'kalman_fit': 0.40,
                    'temporal_consistency': 0.20,
                    'anomaly_detection': 0.20,
                    'source_reliability': 0.10,
                    'trend_alignment': 0.10,
                }
            },
            'kalman': {
                'initial_variance': 1.0,
                'transition_covariance_weight': 0.1,
                'observation_covariance': 1.0,
            }
        }

        user_id = "test-user-d3f6d8e5"

        print("Testing extreme weight change after reset...")
        print("=" * 60)

        # First measurement - will trigger initial reset
        print("\n1. Initial measurement (should trigger reset):")
        result1 = process_measurement(
            user_id=user_id,
            weight=116.573144,
            timestamp=datetime(2025, 2, 13),
            source="internal-questionnaire",
            config=config,
            unit="kg",
            db=db
        )

        print(f"   Weight: 116.57kg")
        print(f"   Accepted: {result1.get('accepted', False)}")
        print(f"   Was reset: {result1.get('was_reset', False)}")
        print(f"   Quality score: {result1.get('quality_score', 'N/A'):.4f}")
        if 'quality_components' in result1:
            components = result1['quality_components']
            print(f"   Components:")
            for comp, score in components.items():
                print(f"     - {comp}: {score:.4f}")

        # Second measurement - extreme drop 13 days later
        print("\n2. Extreme drop 13 days later:")
        result2 = process_measurement(
            user_id=user_id,
            weight=50.160003662109375,
            timestamp=datetime(2025, 2, 26, 15, 59, 18),
            source="patient-device",
            config=config,
            unit="kg",
            db=db
        )

        weight_drop = 116.57 - 50.16
        print(f"   Weight: 50.16kg (drop of {weight_drop:.1f}kg = {weight_drop/116.57*100:.1f}%)")
        print(f"   Accepted: {result2.get('accepted', False)}")
        print(f"   Quality score: {result2.get('quality_score', 'N/A'):.4f}")

        if 'quality_components' in result2:
            components = result2['quality_components']
            print(f"   Components:")
            for comp, score in components.items():
                print(f"     - {comp}: {score:.4f}")

        if not result2.get('accepted'):
            print(f"   Rejection reason: {result2.get('reason', 'N/A')}")

        # Verify the result
        print("\n" + "=" * 60)
        if result2.get('accepted'):
            # Check anomaly detection score specifically
            anomaly_score = result2.get('quality_components', {}).get('anomaly_detection', 0)
            if anomaly_score > 0.1:
                print(f"❌ FAILURE: Anomaly detection score {anomaly_score:.4f} is too high!")
                print("   A 66kg drop in 13 days should be detected as impossible.")
                return False
            else:
                print(f"⚠️  WARNING: Measurement was accepted but anomaly score is low ({anomaly_score:.4f})")
                print("   The overall quality score should have rejected this.")
                return False
        else:
            print("✅ SUCCESS: Extreme weight change was correctly rejected!")
            return True

    finally:
        # Clean up test user from database if needed
        pass  # State DB handles its own cleanup

if __name__ == "__main__":
    success = test_extreme_change_after_reset()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Integration test to verify the enhanced replay mechanism works correctly.

This test simulates the problematic scenario:
1. Reset at 100kg
2. 90kg measurement after 20 days (should be rejected)
3. 98kg measurement 1 hour later (should be accepted)
"""

import tempfile
import csv
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import stream_process


def test_replay_scenario():
    """Test the specific replay scenario with real processing."""

    # Create temp directory for test
    test_dir = tempfile.mkdtemp()
    output_dir = Path(test_dir) / "output"
    output_dir.mkdir()

    base_time = datetime(2024, 1, 1, 12, 0, 0)
    user_id = "test_replay_user"

    # Create test data
    measurements = [
        # Initial measurement to establish state
        {
            'user_id': user_id,
            'effectiveDateTime': base_time.strftime("%Y-%m-%d %H:%M:%S"),
            'source_type': 'patient-device',
            'weight': '100.0',
            'unit': 'kg'
        },
        # Questionnaire triggers soft reset
        {
            'user_id': user_id,
            'effectiveDateTime': (base_time + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            'source_type': 'questionnaire',
            'weight': '100.1',
            'unit': 'kg'
        },
        # 20 days later - 90kg (should be rejected as outlier)
        {
            'user_id': user_id,
            'effectiveDateTime': (base_time + timedelta(days=20, hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            'source_type': 'patient-device',
            'weight': '90.0',
            'unit': 'kg'
        },
        # 1 hour later - 98kg (should be accepted, closer to reset)
        {
            'user_id': user_id,
            'effectiveDateTime': (base_time + timedelta(days=20, hours=3)).strftime("%Y-%m-%d %H:%M:%S"),
            'source_type': 'patient-device',
            'weight': '98.0',
            'unit': 'kg'
        },
        # More measurements to trigger buffer processing
        {
            'user_id': user_id,
            'effectiveDateTime': (base_time + timedelta(days=20, hours=4)).strftime("%Y-%m-%d %H:%M:%S"),
            'source_type': 'patient-device',
            'weight': '98.2',
            'unit': 'kg'
        },
        {
            'user_id': user_id,
            'effectiveDateTime': (base_time + timedelta(days=20, hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
            'source_type': 'patient-device',
            'weight': '98.1',
            'unit': 'kg'
        }
    ]

    # Write CSV
    csv_path = Path(test_dir) / "test_data.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'user_id', 'effectiveDateTime', 'source_type', 'weight', 'unit'
        ])
        writer.writeheader()
        writer.writerows(measurements)

    # Configuration with short buffer for testing
    config = {
        "data": {
            "csv_file": str(csv_path),
            "output_dir": str(output_dir),
            "max_users": 0,
            "min_readings": 0,
            "export_database": False
        },
        "processing": {
            "extreme_threshold": 0.15
        },
        "kalman": {
            "initial_variance": 0.361,
            "transition_covariance_weight": 0.016,
            "observation_covariance": 3.4,
            "reset": {
                "soft": {
                    "enabled": True,
                    "min_weight_change_kg": 5,
                    "trigger_sources": ["questionnaire"],
                    "cooldown_days": 3
                }
            }
        },
        "quality_scoring": {
            "threshold": 0.6
        },
        "replay": {
            "enabled": True,
            "buffer_hours": 2,  # Short buffer for testing
            "trigger_mode": "measurement_count",
            "max_buffer_measurements": 4,  # Trigger after 4 measurements
            "outlier_detection": {
                "min_measurements_for_analysis": 3
            }
        },
        "visualization": {
            "enabled": False
        },
        "logging": {
            "progress_interval": 100,
            "timestamp_format": "%Y%m%d_%H%M%S"
        }
    }

    # Run processing
    print("\n=== Running stream processing with enhanced replay ===")
    user_results, stats = stream_process(
        csv_path=str(csv_path),
        output_dir=str(output_dir),
        config=config
    )

    # Analyze results
    print("\n=== Results Analysis ===")

    if user_id in user_results:
        results = user_results[user_id]
        print(f"Total measurements processed: {len(results)}")

        # Find specific measurements
        for r in results:
            weight = r.get('filtered_weight', r.get('raw_weight'))
            accepted = r.get('accepted', False)
            quality = r.get('quality_score', 0)
            timestamp = r.get('timestamp')

            # Identify which measurement this is
            label = ""
            if abs(weight - 100.0) < 0.2:
                label = "Initial/Reset"
            elif abs(weight - 90.0) < 0.2:
                label = "90kg (should reject)"
            elif abs(weight - 98.0) < 0.2:
                label = "98kg (should accept)"

            if label:
                status = "✓ Accepted" if accepted else "✗ Rejected"
                print(f"  {label}: {weight:.1f}kg - {status} (quality: {quality:.3f})")

    # Check replay stats
    print("\n=== Replay Statistics ===")
    if stats.get('replay_processed', 0) > 0:
        print(f"  Buffers processed: {stats.get('replay_processed', 0)}")
        print(f"  Measurements analyzed: {stats.get('replay_measurements_analyzed', 0)}")
        print(f"  Outliers found: {stats.get('replay_outliers_found', 0)}")
        print(f"  Reset anchors changed: {stats.get('replay_resets_changed', 0)}")
        print(f"  Corrections made: {stats.get('replay_corrections_made', 0)}")
    else:
        print("  No replay processing occurred")

    # Clean up
    import shutil
    shutil.rmtree(test_dir)

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_replay_scenario()
#!/usr/bin/env python3
"""
Test the replay mechanism on problematic real user cases.
"""

import json
import tempfile
import csv
from datetime import datetime, timedelta
from pathlib import Path
import shutil

import sys
sys.path.insert(0, '.')

from main import stream_process

def test_user(user_data_file, user_id, description):
    """Test a specific user's data through the replay mechanism."""

    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"User: {user_id}")
    print(f"{'='*60}")

    # Load user data
    with open(user_data_file, 'r') as f:
        measurements = json.load(f)

    # Create temp directory
    test_dir = tempfile.mkdtemp()
    output_dir = Path(test_dir) / "output"
    output_dir.mkdir()

    # Convert to CSV format
    csv_data = []
    for m in measurements:
        csv_data.append({
            'user_id': user_id,
            'effectiveDateTime': m['timestamp'],
            'source_type': m['source'],
            'weight': str(m['weight']),
            'unit': m['unit']
        })

    # Write CSV
    csv_path = Path(test_dir) / "test_data.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'user_id', 'effectiveDateTime', 'source_type', 'weight', 'unit'
        ])
        writer.writeheader()
        writer.writerows(csv_data)

    # Configuration
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
                    "trigger_sources": ["questionnaire", "internal-questionnaire"],
                    "cooldown_days": 3
                },
                "hard": {
                    "enabled": True,
                    "gap_threshold_days": 30
                }
            }
        },
        "quality_scoring": {
            "threshold": 0.6
        },
        "replay": {
            "enabled": True,
            "buffer_hours": 24,  # 1 day buffer
            "trigger_mode": "time_based",
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
    user_results, stats = stream_process(
        csv_path=str(csv_path),
        output_dir=str(output_dir),
        config=config
    )

    # Analyze results
    print("\n--- Results ---")
    if user_id in user_results:
        results = user_results[user_id]

        # Show all measurements
        for r in results:
            weight = r.get('filtered_weight', r.get('raw_weight'))
            accepted = r.get('accepted', False)
            quality = r.get('quality_score', 0)
            source = r.get('source', 'unknown')

            status = "✓" if accepted else "✗"
            print(f"  {weight:6.1f}kg [{source:25s}] {status} (q={quality:.3f})")

    # Show replay stats
    if stats.get('replay_processed', 0) > 0:
        print(f"\n--- Replay Stats ---")
        print(f"  Buffers: {stats.get('replay_processed', 0)}")
        print(f"  Outliers: {stats.get('replay_outliers_found', 0)}")
        print(f"  Resets changed: {stats.get('replay_resets_changed', 0)}")
        print(f"  Corrections: {stats.get('replay_corrections_made', 0)}")

    # Clean up
    shutil.rmtree(test_dir)


def main():
    """Test problematic cases identified from real data."""

    print("Testing Replay Mechanism on Problematic Real Cases")
    print("="*60)

    # Test Case 1: Extreme drop and recovery (33.5kg error)
    test_user(
        "test_user_44241501.json",
        "44241501-test",
        "Extreme Drop (129kg → 33.5kg → 118kg in 1 minute)"
    )

    # Test Case 2: Massive change after gap
    test_user(
        "test_user_a49f5e62.json",
        "a49f5e62-test",
        "313% Weight Change After 137 Day Gap"
    )

    # Test Case 3: Oscillating pattern
    test_user(
        "test_user_07d08dd8.json",
        "07d08dd8-test",
        "Highly Oscillating Pattern (8 direction changes)"
    )

    # Test Case 4: Rapid resets
    test_user(
        "test_user_05809aa8.json",
        "05809aa8-test",
        "Multiple Resets Same Day"
    )

    print("\n" + "="*60)
    print("Testing Complete")


if __name__ == "__main__":
    main()
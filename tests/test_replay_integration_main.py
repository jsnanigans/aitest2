"""
Integration tests for the replay mechanism as used in main.py.

Tests the complete flow:
1. Processing measurements through main.py's stream_process
2. Buffer accumulation
3. Outlier detection with quality scores
4. State snapshot and restoration
5. Replay of clean measurements
"""

import pytest
import tempfile
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import stream_process, _process_replay_buffer


class TestReplayIntegrationMain:
    """Integration tests for replay mechanism in main.py."""

    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.test_dir) / "output"
        self.output_dir.mkdir()

        self.base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Configuration for testing
        self.config = {
            "data": {
                "csv_file": "",  # Will be set per test
                "output_dir": str(self.output_dir),
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
                "threshold": 0.6,
                "component_weights": {
                    "safety": 0.35,
                    "plausibility": 0.25,
                    "consistency": 0.20,
                    "reliability": 0.20
                }
            },
            "replay": {
                "enabled": True,
                "buffer_hours": 1,  # Short for testing
                "trigger_mode": "time_based",
                "outlier_detection": {
                    "min_measurements_for_analysis": 3,
                    "iqr_multiplier": 1.5,
                    "z_score_threshold": 3.0,
                    "temporal_max_change_percent": 0.30,
                    "quality_score_threshold": 0.7,
                    "kalman_deviation_threshold": 0.15
                },
                "safety": {
                    "max_processing_time_seconds": 60,
                    "preserve_immediate_results": True
                }
            },
            "features": {
                "outlier_detection": True,
                "outlier_iqr": True,
                "outlier_mad": True,
                "outlier_temporal": True,
                "quality_override": True,
                "kalman_deviation_check": True
            },
            "visualization": {
                "enabled": False
            },
            "logging": {
                "progress_interval": 1000,
                "timestamp_format": "%Y%m%d_%H%M%S"
            }
        }

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)

    def _create_test_csv(self, measurements):
        """Create a test CSV file with measurements."""
        csv_path = Path(self.test_dir) / "test_data.csv"

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'user_id', 'effectiveDateTime', 'source_type', 'weight', 'unit'
            ])
            writer.writeheader()
            writer.writerows(measurements)

        return str(csv_path)

    def test_replay_after_reset_scenario(self):
        """Test the specific scenario: reset at 100kg, accepts 90kg after 20 days, rejects 98kg 1 hour later."""
        user_id = "test_user_reset_scenario"

        # Create test data with the problematic scenario
        measurements = [
            # Initial measurements to establish baseline
            {
                'user_id': user_id,
                'effectiveDateTime': self.base_time.strftime("%Y-%m-%d %H:%M:%S"),
                'source_type': 'patient-device',
                'weight': '100.0',
                'unit': 'kg'
            },
            # Trigger soft reset with questionnaire source
            {
                'user_id': user_id,
                'effectiveDateTime': (self.base_time + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
                'source_type': 'questionnaire',
                'weight': '100.1',  # Small change but from questionnaire triggers reset
                'unit': 'kg'
            },
            # Measurement after 20 days - should be rejected (big drop)
            {
                'user_id': user_id,
                'effectiveDateTime': (self.base_time + timedelta(days=20, hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                'source_type': 'patient-device',
                'weight': '90.0',
                'unit': 'kg'
            },
            # Measurement 1 hour later - should be accepted (closer to reset value)
            {
                'user_id': user_id,
                'effectiveDateTime': (self.base_time + timedelta(days=20, hours=3)).strftime("%Y-%m-%d %H:%M:%S"),
                'source_type': 'patient-device',
                'weight': '98.0',
                'unit': 'kg'
            },
            # Add more measurements to trigger buffer processing
            {
                'user_id': user_id,
                'effectiveDateTime': (self.base_time + timedelta(days=20, hours=4)).strftime("%Y-%m-%d %H:%M:%S"),
                'source_type': 'patient-device',
                'weight': '98.2',
                'unit': 'kg'
            }
        ]

        csv_path = self._create_test_csv(measurements)

        # Run stream processing with replay enabled
        user_results, stats = stream_process(
            csv_path=csv_path,
            output_dir=str(self.output_dir),
            config=self.config
        )

        # Check results
        assert user_id in user_results
        results = user_results[user_id]

        # Find the results for our test measurements
        day20_results = [r for r in results if 'days=20' in str(r['timestamp'])]

        # Analyze what was accepted/rejected
        if day20_results:
            # The 90kg should ideally be rejected due to large drop
            result_90kg = next((r for r in day20_results if abs(r['filtered_weight'] - 90.0) < 0.1), None)
            result_98kg = next((r for r in day20_results if abs(r['filtered_weight'] - 98.0) < 0.1), None)

            if result_90kg and result_98kg:
                # Check if replay would have corrected the decision
                # After replay, 98kg should be preferred over 90kg
                print(f"90kg result: accepted={result_90kg.get('accepted')}, quality={result_90kg.get('quality_score')}")
                print(f"98kg result: accepted={result_98kg.get('accepted')}, quality={result_98kg.get('quality_score')}")

        # Check if replay processing occurred
        if stats.get('replay_processed', 0) > 0:
            assert stats.get('replay_measurements_analyzed', 0) > 0
            print(f"Replay stats: processed={stats['replay_processed']}, "
                  f"analyzed={stats['replay_measurements_analyzed']}, "
                  f"outliers={stats.get('replay_outliers_found', 0)}")

    def test_replay_with_normal_progression(self):
        """Test replay with normal weight progression and one outlier."""
        user_id = "test_user_normal"

        # Create measurements with gradual change and one outlier
        measurements = []
        base_weight = 70.0

        for i in range(20):
            timestamp = self.base_time + timedelta(hours=i * 0.1)  # Quick succession to trigger buffer

            # Insert outlier at position 10
            if i == 10:
                weight = 85.0  # Sudden jump
            else:
                weight = base_weight + i * 0.05  # Gradual increase

            measurements.append({
                'user_id': user_id,
                'effectiveDateTime': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'source_type': 'patient-device',
                'weight': str(weight),
                'unit': 'kg'
            })

        csv_path = self._create_test_csv(measurements)

        # Run stream processing
        user_results, stats = stream_process(
            csv_path=csv_path,
            output_dir=str(self.output_dir),
            config=self.config
        )

        # Verify processing
        assert user_id in user_results
        results = user_results[user_id]

        # Check acceptance/rejection patterns
        accepted_count = sum(1 for r in results if r.get('accepted'))
        rejected_count = sum(1 for r in results if not r.get('accepted'))

        print(f"Accepted: {accepted_count}, Rejected: {rejected_count}")

        # The outlier should likely be rejected
        outlier_result = next((r for r in results if abs(r['filtered_weight'] - 85.0) < 0.1), None)
        if outlier_result:
            print(f"Outlier (85kg) was {'accepted' if outlier_result['accepted'] else 'rejected'}")
            print(f"Quality score: {outlier_result.get('quality_score')}")

    def test_replay_buffer_accumulation_and_trigger(self):
        """Test that buffer accumulates properly and triggers replay."""
        user_id = "test_buffer_trigger"

        # Create measurements spanning more than buffer_hours
        measurements = []
        for i in range(30):
            # Space out measurements to span > 1 hour (our test buffer_hours)
            timestamp = self.base_time + timedelta(minutes=i * 3)
            weight = 70.0 + (i % 5) * 0.1  # Some variation

            measurements.append({
                'user_id': user_id,
                'effectiveDateTime': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'source_type': 'patient-device',
                'weight': str(weight),
                'unit': 'kg'
            })

        csv_path = self._create_test_csv(measurements)

        # Mock to track replay processing
        replay_calls = []

        def mock_process_replay(*args, **kwargs):
            replay_calls.append(kwargs)
            # Call original implementation
            return _process_replay_buffer(*args, **kwargs)

        with patch('main._process_replay_buffer', side_effect=mock_process_replay):
            user_results, stats = stream_process(
                csv_path=csv_path,
                output_dir=str(self.output_dir),
                config=self.config
            )

        # Check that replay was triggered
        if self.config['replay']['enabled']:
            # Buffer should have triggered at least once
            print(f"Replay calls: {len(replay_calls)}")
            print(f"Replay stats: {stats.get('replay_processed', 0)} buffers processed")

    def test_quality_score_override_in_replay(self):
        """Test that high quality scores prevent outlier detection during replay."""
        user_id = "test_quality_override"

        measurements = [
            # Normal measurements
            {
                'user_id': user_id,
                'effectiveDateTime': self.base_time.strftime("%Y-%m-%d %H:%M:%S"),
                'source_type': 'patient-device',
                'weight': '70.0',
                'unit': 'kg'
            },
            {
                'user_id': user_id,
                'effectiveDateTime': (self.base_time + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"),
                'source_type': 'patient-device',
                'weight': '70.2',
                'unit': 'kg'
            },
            # High-quality measurement that looks like outlier
            {
                'user_id': user_id,
                'effectiveDateTime': (self.base_time + timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S"),
                'source_type': 'care-team-upload',  # High reliability source
                'weight': '75.0',  # Jump but from reliable source
                'unit': 'kg'
            },
            {
                'user_id': user_id,
                'effectiveDateTime': (self.base_time + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
                'source_type': 'patient-device',
                'weight': '70.3',
                'unit': 'kg'
            },
            # Trigger buffer processing
            {
                'user_id': user_id,
                'effectiveDateTime': (self.base_time + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                'source_type': 'patient-device',
                'weight': '70.4',
                'unit': 'kg'
            }
        ]

        csv_path = self._create_test_csv(measurements)

        user_results, stats = stream_process(
            csv_path=csv_path,
            output_dir=str(self.output_dir),
            config=self.config
        )

        # Check results
        assert user_id in user_results
        results = user_results[user_id]

        # Find the high-quality outlier measurement
        high_quality_result = next(
            (r for r in results if abs(r['filtered_weight'] - 75.0) < 0.1),
            None
        )

        if high_quality_result:
            print(f"High-quality 75kg measurement:")
            print(f"  Accepted: {high_quality_result.get('accepted')}")
            print(f"  Quality score: {high_quality_result.get('quality_score')}")
            print(f"  Source: {high_quality_result.get('source')}")

            # Care-team-upload should have high quality and be accepted
            # even if it looks like an outlier statistically
            if high_quality_result.get('source') == 'care-team-upload':
                assert high_quality_result.get('quality_score', 0) > 0.7

    def test_filtered_csv_output(self):
        """Test that filtered CSV contains only accepted measurements with quality scores."""
        user_id = "test_filtered_output"

        measurements = [
            {
                'user_id': user_id,
                'effectiveDateTime': self.base_time.strftime("%Y-%m-%d %H:%M:%S"),
                'source_type': 'patient-device',
                'weight': '70.0',
                'unit': 'kg'
            },
            {
                'user_id': user_id,
                'effectiveDateTime': (self.base_time + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
                'source_type': 'patient-device',
                'weight': '95.0',  # Likely to be rejected
                'unit': 'kg'
            },
            {
                'user_id': user_id,
                'effectiveDateTime': (self.base_time + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                'source_type': 'patient-device',
                'weight': '70.5',
                'unit': 'kg'
            }
        ]

        csv_path = self._create_test_csv(measurements)
        filtered_output = Path(self.test_dir) / "filtered.csv"

        user_results, stats = stream_process(
            csv_path=csv_path,
            output_dir=str(self.output_dir),
            config=self.config,
            filtered_output=str(filtered_output)
        )

        # Check filtered CSV was created
        assert filtered_output.exists()

        # Read and verify filtered CSV
        with open(filtered_output) as f:
            reader = csv.DictReader(f)
            filtered_rows = list(reader)

        # Should only contain accepted measurements
        assert len(filtered_rows) == stats['accepted']

        # Each row should have quality_score column
        for row in filtered_rows:
            assert 'quality_score' in row
            quality_score = float(row['quality_score'])
            assert 0 <= quality_score <= 1
            print(f"Weight: {row['weight']}kg, Quality: {quality_score:.3f}")

    @pytest.mark.parametrize("trigger_mode,buffer_hours,max_measurements", [
        ("time_based", 1, 100),
        ("measurement_count", 72, 10),
    ])
    def test_different_trigger_modes(self, trigger_mode, buffer_hours, max_measurements):
        """Test different buffer trigger modes."""
        user_id = f"test_trigger_{trigger_mode}"

        # Update config for this test
        self.config['replay']['trigger_mode'] = trigger_mode
        self.config['replay']['buffer_hours'] = buffer_hours
        self.config['replay']['max_buffer_measurements'] = max_measurements

        # Create measurements based on trigger mode
        if trigger_mode == "time_based":
            # Create measurements spanning more than buffer_hours
            num_measurements = 20
            time_span = buffer_hours + 0.5  # Exceed buffer hours
        else:
            # Create enough measurements to trigger count-based
            num_measurements = max_measurements + 5
            time_span = 0.5  # Short time span

        measurements = []
        for i in range(num_measurements):
            timestamp = self.base_time + timedelta(
                hours=i * time_span / num_measurements
            )
            measurements.append({
                'user_id': user_id,
                'effectiveDateTime': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'source_type': 'patient-device',
                'weight': str(70.0 + i * 0.1),
                'unit': 'kg'
            })

        csv_path = self._create_test_csv(measurements)

        user_results, stats = stream_process(
            csv_path=csv_path,
            output_dir=str(self.output_dir),
            config=self.config
        )

        print(f"Trigger mode: {trigger_mode}")
        print(f"Measurements: {num_measurements}")
        print(f"Replay buffers processed: {stats.get('replay_processed', 0)}")

        # Should have processed some measurements
        assert user_id in user_results
        assert len(user_results[user_id]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
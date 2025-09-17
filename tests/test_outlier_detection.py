"""
Comprehensive unit tests for OutlierDetector.
Focus on critical functionality: statistical methods, quality overrides, Kalman deviation.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from src.processing.outlier_detection import OutlierDetector
from src.feature_manager import FeatureManager


@pytest.fixture
def basic_config():
    """Basic configuration for outlier detector."""
    return {
        'iqr_multiplier': 1.5,
        'z_score_threshold': 3.0,
        'temporal_max_change_percent': 0.30,
        'min_measurements_for_analysis': 5,
        'quality_score_threshold': 0.7,
        'kalman_deviation_threshold': 0.15
    }


@pytest.fixture
def normal_measurements():
    """Generate measurements with normal variation."""
    base_time = datetime(2024, 1, 1, 12, 0)
    weights = [70.0, 70.5, 69.8, 70.2, 70.1, 69.9, 70.3, 70.4, 69.7, 70.0]
    return [
        {
            'weight': w,
            'timestamp': base_time + timedelta(days=i),
            'source': 'patient-device',
            'metadata': {'quality_score': 0.6}
        }
        for i, w in enumerate(weights)
    ]


@pytest.fixture
def measurements_with_outliers():
    """Dataset with known statistical outliers."""
    base_time = datetime(2024, 1, 1, 12, 0)
    # Normal weights around 70kg with two obvious outliers
    weights = [70.0, 70.5, 69.8, 85.0, 70.2, 70.1, 55.0, 70.3, 70.4, 69.7]
    return [
        {
            'weight': w,
            'timestamp': base_time + timedelta(days=i),
            'source': 'patient-device',
            'metadata': {'quality_score': 0.5}
        }
        for i, w in enumerate(weights)
    ]


@pytest.fixture
def high_quality_measurements():
    """Measurements with high quality scores that should be protected."""
    base_time = datetime(2024, 1, 1, 12, 0)
    return [
        {'weight': 70.0, 'timestamp': base_time, 'metadata': {'quality_score': 0.5}},
        {'weight': 70.5, 'timestamp': base_time + timedelta(days=1), 'metadata': {'quality_score': 0.6}},
        {'weight': 85.0, 'timestamp': base_time + timedelta(days=2), 'metadata': {'quality_score': 0.9}},  # High quality outlier
        {'weight': 70.2, 'timestamp': base_time + timedelta(days=3), 'metadata': {'quality_score': 0.5}},
        {'weight': 70.1, 'timestamp': base_time + timedelta(days=4), 'metadata': {'quality_score': 0.6}},
        {'weight': 55.0, 'timestamp': base_time + timedelta(days=5), 'metadata': {'accepted': True}},  # Accepted outlier
        {'weight': 70.3, 'timestamp': base_time + timedelta(days=6), 'metadata': {'quality_score': 0.5}},
    ]


@pytest.fixture
def mock_feature_manager():
    """Mock feature manager with all features enabled."""
    manager = Mock(spec=FeatureManager)
    manager.is_enabled.return_value = True
    return manager


@pytest.fixture
def mock_db_with_kalman_state():
    """Mock database with Kalman state for predictions."""
    db = Mock()
    db.get_state.return_value = {
        'last_state': [70.0, 0.1],  # Expected weight around 70kg
        'state_history': [
            {
                'timestamp': datetime(2024, 1, 1),
                'state': [70.0, 0.1]
            }
        ]
    }
    return db


class TestStatisticalMethods:
    """Test core statistical outlier detection methods."""

    def test_iqr_detection_normal_data(self, basic_config, normal_measurements):
        """Test IQR detection on normal distribution data."""
        detector = OutlierDetector(basic_config)
        outliers = detector._detect_iqr_outliers([m['weight'] for m in normal_measurements])

        # Should find no outliers in normal data
        assert len(outliers) == 0

    def test_iqr_detection_with_outliers(self, basic_config, measurements_with_outliers):
        """Test IQR detection correctly identifies outliers."""
        detector = OutlierDetector(basic_config)
        weights = [m['weight'] for m in measurements_with_outliers]
        outliers = detector._detect_iqr_outliers(weights)

        # Should detect the extreme values (85.0 at index 3, 55.0 at index 6)
        assert 3 in outliers  # 85.0 kg
        assert 6 in outliers  # 55.0 kg

    def test_iqr_small_dataset(self, basic_config):
        """Test IQR with insufficient data points."""
        detector = OutlierDetector(basic_config)
        weights = [70.0, 70.5, 69.8]  # Less than 4 points
        outliers = detector._detect_iqr_outliers(weights)

        assert len(outliers) == 0

    def test_mad_detection_normal_data(self, basic_config, normal_measurements):
        """Test Modified Z-score detection on normal data."""
        detector = OutlierDetector(basic_config)
        outliers = detector._detect_zscore_outliers([m['weight'] for m in normal_measurements])

        # Should find no outliers in normal data
        assert len(outliers) == 0

    def test_mad_detection_with_outliers(self, basic_config, measurements_with_outliers):
        """Test Modified Z-score correctly identifies outliers."""
        detector = OutlierDetector(basic_config)
        weights = [m['weight'] for m in measurements_with_outliers]
        outliers = detector._detect_zscore_outliers(weights)

        # Should detect extreme outliers
        assert 3 in outliers  # 85.0 kg
        assert 6 in outliers  # 55.0 kg

    def test_mad_zero_variance(self, basic_config):
        """Test MAD with all identical values."""
        detector = OutlierDetector(basic_config)
        weights = [70.0] * 10  # All identical
        outliers = detector._detect_zscore_outliers(weights)

        assert len(outliers) == 0

    def test_temporal_consistency(self, basic_config):
        """Test temporal consistency detection for rapid weight changes."""
        detector = OutlierDetector(basic_config)
        base_time = datetime(2024, 1, 1, 12, 0)

        # Create measurements with rapid weight change
        measurements = [
            {'weight': 70.0, 'timestamp': base_time},
            {'weight': 70.5, 'timestamp': base_time + timedelta(hours=25)},  # Normal change
            {'weight': 95.0, 'timestamp': base_time + timedelta(hours=50)},  # >30% change
        ]

        outliers = detector._detect_temporal_outliers(measurements)

        assert 2 in outliers  # The 95kg measurement should be flagged

    def test_temporal_short_interval(self, basic_config):
        """Test temporal detection ignores measurements <1hr apart."""
        detector = OutlierDetector(basic_config)
        base_time = datetime(2024, 1, 1, 12, 0)

        measurements = [
            {'weight': 70.0, 'timestamp': base_time},
            {'weight': 95.0, 'timestamp': base_time + timedelta(minutes=30)},  # <1hr, should be ignored
        ]

        outliers = detector._detect_temporal_outliers(measurements)

        assert len(outliers) == 0


class TestQualityOverrides:
    """Test quality score override functionality."""

    def test_high_quality_protection(self, basic_config, high_quality_measurements, mock_feature_manager):
        """Test that high quality measurements are never marked as outliers."""
        config = basic_config.copy()
        config['feature_manager'] = mock_feature_manager
        detector = OutlierDetector(config)

        outliers = detector.detect_outliers(high_quality_measurements)

        # The 85.0kg and 55.0kg measurements should NOT be in outliers
        # because they have high quality score or are accepted
        assert 2 not in outliers  # 85.0kg with quality_score=0.9
        assert 5 not in outliers  # 55.0kg with accepted=True

    def test_quality_override_disabled(self, basic_config, high_quality_measurements):
        """Test outlier detection when quality override is disabled."""
        manager = Mock(spec=FeatureManager)
        manager.is_enabled.side_effect = lambda feature: {
            'outlier_detection': True,
            'quality_override': False,  # Disabled
            'outlier_iqr': True,
            'outlier_mad': True,
            'outlier_temporal': False,
            'kalman_deviation_check': False
        }.get(feature, False)

        config = basic_config.copy()
        config['feature_manager'] = manager
        detector = OutlierDetector(config)

        outliers = detector.detect_outliers(high_quality_measurements)

        # Without quality override, the outliers should be detected
        assert 2 in outliers  # 85.0kg should be an outlier
        assert 5 in outliers  # 55.0kg should be an outlier

    def test_quality_threshold_boundary(self, basic_config):
        """Test quality score threshold boundary conditions."""
        manager = Mock(spec=FeatureManager)
        manager.is_enabled.return_value = True

        config = basic_config.copy()
        config['feature_manager'] = manager
        config['quality_score_threshold'] = 0.7
        detector = OutlierDetector(config)

        base_time = datetime(2024, 1, 1)
        measurements = [
            {'weight': 70.0, 'timestamp': base_time, 'metadata': {'quality_score': 0.5}},
            {'weight': 85.0, 'timestamp': base_time + timedelta(days=1), 'metadata': {'quality_score': 0.7}},  # At threshold
            {'weight': 55.0, 'timestamp': base_time + timedelta(days=2), 'metadata': {'quality_score': 0.71}},  # Above threshold
            {'weight': 70.0, 'timestamp': base_time + timedelta(days=3), 'metadata': {'quality_score': 0.5}},
            {'weight': 70.0, 'timestamp': base_time + timedelta(days=4), 'metadata': {'quality_score': 0.5}},
        ]

        outliers = detector.detect_outliers(measurements)

        # 0.7 is not > 0.7, so should be detected
        assert 1 in outliers  # 85.0kg with quality_score=0.7
        # 0.71 is > 0.7, so should be protected
        assert 2 not in outliers  # 55.0kg with quality_score=0.71


class TestKalmanDeviation:
    """Test Kalman prediction-based outlier detection."""

    def test_kalman_deviation_detection(self, basic_config, mock_db_with_kalman_state):
        """Test detection of measurements deviating from Kalman prediction."""
        config = basic_config.copy()
        detector = OutlierDetector(config, db=mock_db_with_kalman_state)

        base_time = datetime(2024, 1, 2)  # After the state history timestamp
        measurements = [
            {'weight': 70.5, 'timestamp': base_time, 'metadata': {}},  # Within 15%
            {'weight': 75.0, 'timestamp': base_time + timedelta(days=1), 'metadata': {}},  # ~7% deviation
            {'weight': 85.0, 'timestamp': base_time + timedelta(days=2), 'metadata': {}},  # >15% deviation
            {'weight': 70.0, 'timestamp': base_time + timedelta(days=3), 'metadata': {}},  # Perfect match
            {'weight': 55.0, 'timestamp': base_time + timedelta(days=4), 'metadata': {}},  # >15% deviation
        ]

        outliers = detector._detect_kalman_outliers(measurements, 'user123')

        assert 2 in outliers  # 85.0kg is >15% from 70kg
        assert 4 in outliers  # 55.0kg is >15% from 70kg
        assert 0 not in outliers  # 70.5kg is within threshold
        assert 1 not in outliers  # 75.0kg is within threshold

    def test_kalman_no_state(self, basic_config):
        """Test Kalman detection with no user state available."""
        db = Mock()
        db.get_state.return_value = None

        detector = OutlierDetector(basic_config, db=db)
        measurements = [
            {'weight': 85.0, 'timestamp': datetime(2024, 1, 1), 'metadata': {}}
        ]

        outliers = detector._detect_kalman_outliers(measurements, 'user123')

        assert len(outliers) == 0  # No detection without state

    def test_kalman_fifteen_percent_threshold(self, basic_config, mock_db_with_kalman_state):
        """Verify the exact 15% deviation threshold."""
        config = basic_config.copy()
        config['kalman_deviation_threshold'] = 0.15
        detector = OutlierDetector(config, db=mock_db_with_kalman_state)

        base_time = datetime(2024, 1, 2)
        # Kalman predicts 70kg
        measurements = [
            {'weight': 80.5, 'timestamp': base_time, 'metadata': {}},  # Exactly 15% above (70 * 1.15 = 80.5)
            {'weight': 59.5, 'timestamp': base_time + timedelta(days=1), 'metadata': {}},  # Exactly 15% below (70 * 0.85 = 59.5)
            {'weight': 80.6, 'timestamp': base_time + timedelta(days=2), 'metadata': {}},  # Just over 15%
            {'weight': 59.4, 'timestamp': base_time + timedelta(days=3), 'metadata': {}},  # Just over 15%
        ]

        outliers = detector._detect_kalman_outliers(measurements, 'user123')

        # Exactly 15% should not be flagged
        assert 0 not in outliers
        assert 1 not in outliers
        # Over 15% should be flagged
        assert 2 in outliers
        assert 3 in outliers


class TestBatchProcessing:
    """Test batch processing and minimum measurement requirements."""

    def test_minimum_measurements_requirement(self, basic_config):
        """Test that analysis requires minimum 5 measurements."""
        detector = OutlierDetector(basic_config)

        base_time = datetime(2024, 1, 1)
        # Less than 5 measurements
        measurements = [
            {'weight': 70.0, 'timestamp': base_time + timedelta(days=i), 'metadata': {}}
            for i in range(4)
        ]

        outliers = detector.detect_outliers(measurements)

        assert len(outliers) == 0  # No analysis with <5 measurements

    def test_exactly_minimum_measurements(self, basic_config):
        """Test analysis proceeds with exactly 5 measurements."""
        detector = OutlierDetector(basic_config)

        base_time = datetime(2024, 1, 1)
        measurements = [
            {'weight': 70.0, 'timestamp': base_time, 'metadata': {}},
            {'weight': 70.5, 'timestamp': base_time + timedelta(days=1), 'metadata': {}},
            {'weight': 85.0, 'timestamp': base_time + timedelta(days=2), 'metadata': {}},  # Outlier
            {'weight': 70.2, 'timestamp': base_time + timedelta(days=3), 'metadata': {}},
            {'weight': 70.1, 'timestamp': base_time + timedelta(days=4), 'metadata': {}},
        ]

        outliers = detector.detect_outliers(measurements)

        # Analysis should proceed and detect the outlier
        assert 2 in outliers  # 85.0kg

    def test_large_batch_performance(self, basic_config):
        """Test performance with large batch (not timing, just functionality)."""
        detector = OutlierDetector(basic_config)

        base_time = datetime(2024, 1, 1)
        # Generate 1000 measurements
        np.random.seed(42)
        measurements = []
        for i in range(1000):
            weight = np.random.normal(70, 2)  # Normal distribution
            if i in [100, 500, 800]:  # Add some outliers
                weight = 95.0
            measurements.append({
                'weight': weight,
                'timestamp': base_time + timedelta(hours=i),
                'metadata': {}
            })

        outliers = detector.detect_outliers(measurements)

        # Should detect at least some of our injected outliers
        assert len(outliers) > 0
        # Check that at least one of our injected outliers was found
        assert any(idx in outliers for idx in [100, 500, 800])


class TestIntegration:
    """Test integration of multiple detection methods."""

    def test_and_logic_statistical_and_kalman(self, basic_config, mock_db_with_kalman_state):
        """Test AND logic requiring both statistical and Kalman failures."""
        manager = Mock(spec=FeatureManager)
        manager.is_enabled.return_value = True

        config = basic_config.copy()
        config['feature_manager'] = manager
        detector = OutlierDetector(config, db=mock_db_with_kalman_state)

        base_time = datetime(2024, 1, 2)
        measurements = [
            {'weight': 70.0, 'timestamp': base_time, 'metadata': {}},
            {'weight': 70.5, 'timestamp': base_time + timedelta(days=1), 'metadata': {}},
            {'weight': 82.0, 'timestamp': base_time + timedelta(days=2), 'metadata': {}},  # Fails Kalman but not statistical
            {'weight': 85.0, 'timestamp': base_time + timedelta(days=3), 'metadata': {}},  # Fails both
            {'weight': 70.2, 'timestamp': base_time + timedelta(days=4), 'metadata': {}},
        ]

        outliers = detector.detect_outliers(measurements, 'user123')

        # 82.0kg fails Kalman (>15% from 70kg) but might not fail statistical
        # 85.0kg should fail both statistical and Kalman tests
        assert 3 in outliers  # 85.0kg should definitely be an outlier

    def test_feature_toggle_controls(self, basic_config):
        """Test that feature toggles control method activation."""
        manager = Mock(spec=FeatureManager)

        # Configure specific feature states
        manager.is_enabled.side_effect = lambda feature: {
            'outlier_detection': True,
            'outlier_iqr': False,  # IQR disabled
            'outlier_mad': True,   # MAD enabled
            'outlier_temporal': False,  # Temporal disabled
            'kalman_deviation_check': False,  # Kalman disabled
            'quality_override': False
        }.get(feature, False)

        config = basic_config.copy()
        config['feature_manager'] = manager
        detector = OutlierDetector(config)

        base_time = datetime(2024, 1, 1)
        measurements = [
            {'weight': 70.0, 'timestamp': base_time, 'metadata': {}},
            {'weight': 70.5, 'timestamp': base_time + timedelta(days=1), 'metadata': {}},
            {'weight': 85.0, 'timestamp': base_time + timedelta(days=2), 'metadata': {}},  # Outlier
            {'weight': 70.2, 'timestamp': base_time + timedelta(days=3), 'metadata': {}},
            {'weight': 70.1, 'timestamp': base_time + timedelta(days=4), 'metadata': {}},
        ]

        outliers = detector.detect_outliers(measurements)

        # Only MAD detection should be active
        assert 2 in outliers  # Should still detect the extreme outlier

    def test_outlier_detection_completely_disabled(self, basic_config):
        """Test that disabling outlier_detection returns empty set."""
        manager = Mock(spec=FeatureManager)
        manager.is_enabled.side_effect = lambda feature: feature != 'outlier_detection'

        config = basic_config.copy()
        config['feature_manager'] = manager
        detector = OutlierDetector(config)

        measurements = [
            {'weight': 70.0, 'timestamp': datetime(2024, 1, 1), 'metadata': {}},
            {'weight': 200.0, 'timestamp': datetime(2024, 1, 2), 'metadata': {}},  # Extreme outlier
        ]

        outliers = detector.detect_outliers(measurements)

        assert len(outliers) == 0  # No detection when disabled


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_measurements(self, basic_config):
        """Test handling of empty measurements list."""
        detector = OutlierDetector(basic_config)
        outliers = detector.detect_outliers([])

        assert len(outliers) == 0

    def test_single_measurement(self, basic_config):
        """Test handling of single measurement."""
        detector = OutlierDetector(basic_config)
        measurements = [
            {'weight': 70.0, 'timestamp': datetime(2024, 1, 1), 'metadata': {}}
        ]
        outliers = detector.detect_outliers(measurements)

        assert len(outliers) == 0

    def test_all_identical_values(self, basic_config):
        """Test handling of all identical measurements."""
        detector = OutlierDetector(basic_config)

        base_time = datetime(2024, 1, 1)
        measurements = [
            {'weight': 70.0, 'timestamp': base_time + timedelta(days=i), 'metadata': {}}
            for i in range(10)
        ]

        outliers = detector.detect_outliers(measurements)

        assert len(outliers) == 0

    def test_extreme_outlier_detection(self, basic_config):
        """Test detection of extreme outliers (10x median)."""
        detector = OutlierDetector(basic_config)

        base_time = datetime(2024, 1, 1)
        measurements = [
            {'weight': 70.0, 'timestamp': base_time, 'metadata': {}},
            {'weight': 70.5, 'timestamp': base_time + timedelta(days=1), 'metadata': {}},
            {'weight': 700.0, 'timestamp': base_time + timedelta(days=2), 'metadata': {}},  # 10x median
            {'weight': 70.2, 'timestamp': base_time + timedelta(days=3), 'metadata': {}},
            {'weight': 70.1, 'timestamp': base_time + timedelta(days=4), 'metadata': {}},
        ]

        outliers = detector.detect_outliers(measurements)

        assert 2 in outliers  # 700kg should definitely be an outlier

    def test_negative_weights(self, basic_config):
        """Test handling of negative weights."""
        detector = OutlierDetector(basic_config)

        base_time = datetime(2024, 1, 1)
        measurements = [
            {'weight': 70.0, 'timestamp': base_time, 'metadata': {}},
            {'weight': 70.5, 'timestamp': base_time + timedelta(days=1), 'metadata': {}},
            {'weight': -70.0, 'timestamp': base_time + timedelta(days=2), 'metadata': {}},  # Negative
            {'weight': 70.2, 'timestamp': base_time + timedelta(days=3), 'metadata': {}},
            {'weight': 70.1, 'timestamp': base_time + timedelta(days=4), 'metadata': {}},
        ]

        outliers = detector.detect_outliers(measurements)

        assert 2 in outliers  # Negative weight should be an outlier


class TestAnalysisAndUtilities:
    """Test analysis and utility methods."""

    def test_analyze_outliers(self, basic_config, measurements_with_outliers):
        """Test outlier analysis report generation."""
        detector = OutlierDetector(basic_config)
        outlier_indices = {3, 6}  # Known outliers

        analysis = detector.analyze_outliers(measurements_with_outliers, outlier_indices)

        assert analysis['total_measurements'] == 10
        assert analysis['outlier_count'] == 2
        assert analysis['outlier_percentage'] == 20.0
        assert len(analysis['outlier_details']) == 2

        # Check details for first outlier
        detail = analysis['outlier_details'][0]
        assert detail['index'] == 3
        assert detail['weight'] == 85.0
        assert 'deviation_from_median' in detail
        assert 'change_from_previous' in detail

    def test_get_clean_measurements(self, basic_config, measurements_with_outliers):
        """Test removal of outliers from measurements."""
        detector = OutlierDetector(basic_config)

        clean_measurements, outlier_indices = detector.get_clean_measurements(measurements_with_outliers)

        # Should have removed the outliers
        assert len(clean_measurements) < len(measurements_with_outliers)

        # Check that outliers are not in clean measurements
        clean_weights = [m['weight'] for m in clean_measurements]
        assert 85.0 not in clean_weights
        assert 55.0 not in clean_weights

        # Original list should be unchanged
        assert len(measurements_with_outliers) == 10

    def test_config_updates(self, basic_config):
        """Test dynamic configuration updates."""
        detector = OutlierDetector(basic_config)

        # Initial config
        assert detector.iqr_multiplier == 1.5
        assert detector.z_score_threshold == 3.0

        # Update config
        new_config = {
            'iqr_multiplier': 2.0,
            'z_score_threshold': 2.5
        }
        detector.update_config(new_config)

        assert detector.iqr_multiplier == 2.0
        assert detector.z_score_threshold == 2.5

        # Get config
        config = detector.get_config()
        assert config['iqr_multiplier'] == 2.0
        assert config['z_score_threshold'] == 2.5


@pytest.mark.parametrize("weights,expected_outliers", [
    ([70, 70, 70, 70, 70], set()),  # All identical
    ([70, 71, 69, 70, 150], {4}),  # One extreme outlier
    ([70, 71, 69, 70, 20], {4}),  # One extreme low outlier
    ([70, 71, 69, 150, 20], {3, 4}),  # Two outliers
])
def test_parametrized_outlier_detection(basic_config, weights, expected_outliers):
    """Parametrized test for various outlier scenarios."""
    detector = OutlierDetector(basic_config)

    base_time = datetime(2024, 1, 1)
    measurements = [
        {'weight': float(w), 'timestamp': base_time + timedelta(days=i), 'metadata': {}}
        for i, w in enumerate(weights)
    ]

    outliers = detector.detect_outliers(measurements)

    assert outliers == expected_outliers
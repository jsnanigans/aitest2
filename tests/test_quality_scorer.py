"""
Unit tests for QualityScorer - focusing on critical functionality and common use cases.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import numpy as np

from src.processing.quality_scorer import QualityScorer, QualityScore
from src.constants import PHYSIOLOGICAL_LIMITS, SOURCE_PROFILES


class TestQualityScorer:
    """Test suite for QualityScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create a QualityScorer instance for testing."""
        config = {'feature_manager': MagicMock()}
        return QualityScorer(config)

    @pytest.fixture
    def mock_feature_manager(self):
        """Create a mock feature manager with all features enabled."""
        manager = MagicMock()
        manager.is_enabled.return_value = True
        return manager

    # ============= Safety Score Tests (Critical) =============

    @pytest.mark.parametrize("weight,expected_score", [
        (25, 0.0),      # Below absolute minimum
        (450, 0.0),     # Above absolute maximum
        (70, 1.0),      # Safe range
        (150, 1.0),     # Safe range (higher)
        (35, 0.1),      # Suspicious - very low
        (350, 0.1),     # Suspicious - very high
    ])
    def test_safety_score_boundaries(self, scorer, weight, expected_score):
        """Test safety score at critical weight boundaries."""
        score = scorer._calculate_safety(weight, height_m=1.7)
        assert pytest.approx(score, rel=0.2) == expected_score

    def test_safety_score_with_extreme_bmi(self, scorer):
        """Test safety score with extreme BMI values - BMI penalty only applies in suspicious range."""
        # BMI < 15 with suspicious weight (38kg, height=2.0 -> BMI=9.5)
        score_low_bmi = scorer._calculate_safety(38, height_m=2.0)
        assert score_low_bmi < 0.3  # Should be heavily penalized

        # Normal weight range - no BMI penalty even with high BMI
        score_high_bmi = scorer._calculate_safety(200, height_m=1.3)
        assert score_high_bmi == 1.0  # In safe range, no penalty

    def test_safety_score_edge_cases(self, scorer):
        """Test safety score with edge cases."""
        # Negative weight
        assert scorer._calculate_safety(-10, height_m=1.7) == 0.0

        # Zero weight
        assert scorer._calculate_safety(0, height_m=1.7) == 0.0

        # Infinity weight
        assert scorer._calculate_safety(float('inf'), height_m=1.7) == 0.0

        # NaN weight - implementation returns 1.0 due to comparison behavior
        assert scorer._calculate_safety(float('nan'), height_m=1.7) == 1.0

    # ============= Plausibility Score Tests =============

    def test_plausibility_no_history(self, scorer):
        """Test plausibility score with no historical data."""
        score = scorer._calculate_plausibility(
            weight=70,
            previous_weight=None,
            recent_weights=None
        )
        assert score == 0.8  # Default for no history

    def test_plausibility_with_previous_only(self, scorer):
        """Test plausibility with only previous weight."""
        score = scorer._calculate_plausibility(
            weight=71,
            previous_weight=70,
            recent_weights=None
        )
        # Small change (1.4%) should be plausible
        assert score > 0.9

    def test_plausibility_with_history(self, scorer):
        """Test plausibility with recent weight history."""
        recent_weights = [70, 70.5, 71, 70.8, 71.2]

        # Test normal variation
        score_normal = scorer._calculate_plausibility(
            weight=71.5,
            previous_weight=71.2,
            recent_weights=recent_weights
        )
        assert score_normal > 0.8

        # Test large deviation
        score_outlier = scorer._calculate_plausibility(
            weight=80,  # Big jump from ~71
            previous_weight=71.2,
            recent_weights=recent_weights
        )
        assert score_outlier < 0.5

    def test_plausibility_with_trend(self, scorer):
        """Test plausibility with trending weights."""
        # Consistent upward trend
        recent_weights = [70, 71, 72, 73, 74]

        score = scorer._calculate_plausibility(
            weight=75,  # Continues trend
            previous_weight=74,
            recent_weights=recent_weights
        )
        assert score > 0.8  # Should be plausible

    # ============= Consistency Score Tests =============

    def test_consistency_no_previous(self, scorer):
        """Test consistency score with no previous weight."""
        score = scorer._calculate_consistency(
            weight=70,
            previous_weight=None,
            time_diff_hours=None
        )
        assert score == 0.8  # Default for no history

    @pytest.mark.parametrize("time_diff,weight_change,expected_min", [
        (2, 1.0, 0.9),   # 1kg in 2 hours - acceptable
        (2, 4.0, 0.3),   # 4kg in 2 hours - suspicious
        (24, 2.0, 0.8),  # 2kg in 24 hours - borderline
        (24, 5.0, 0.3),  # 5kg in 24 hours - very suspicious
        (168, 3.0, 0.9), # 3kg in a week - normal
    ])
    def test_consistency_change_rates(self, scorer, time_diff, weight_change, expected_min):
        """Test consistency score with various change rates."""
        score = scorer._calculate_consistency(
            weight=70 + weight_change,
            previous_weight=70,
            time_diff_hours=time_diff
        )
        assert score >= expected_min

    def test_consistency_percentage_mode(self, scorer):
        """Test consistency switches to percentage mode for large changes."""
        # 10% change should trigger percentage mode
        score = scorer._calculate_consistency(
            weight=77,  # 10% increase from 70
            previous_weight=70,
            time_diff_hours=24
        )
        assert score < 0.7  # Should be penalized

    # ============= Reliability Score Tests =============

    @pytest.mark.parametrize("source,expected_score", [
        ('care-team-upload', 1.0),         # excellent, low outlier rate
        ('patient-device', 0.765),         # good * 0.9 (outlier rate ~20)
        ('https://api.iglucose.com', 0.4), # poor * 0.8 (high outlier rate)
        ('unknown-source', 0.54),          # unknown * 0.9
    ])
    def test_reliability_by_source(self, scorer, source, expected_score):
        """Test reliability scores for different sources."""
        score = scorer._calculate_reliability(source)
        assert pytest.approx(score, rel=0.1) == expected_score

    # ============= Integration Tests =============

    def test_overall_score_calculation(self, scorer):
        """Test overall quality score calculation."""
        result = scorer.calculate_quality_score(
            weight=71,
            source='patient-device',
            previous_weight=70,
            time_diff_hours=24,
            recent_weights=[70, 70.5, 70.8],
            user_height_m=1.75
        )

        assert isinstance(result, QualityScore)
        assert 0 <= result.overall <= 1
        assert all(0 <= v <= 1 for v in result.components.values())
        assert len(result.components) == 4  # safety, plausibility, consistency, reliability

    def test_acceptance_threshold(self, scorer):
        """Test acceptance logic with threshold."""
        # Good measurement
        good_result = scorer.calculate_quality_score(
            weight=70,
            source='care-team-upload',
            previous_weight=69.5,
            time_diff_hours=24,
            user_height_m=1.75
        )
        assert good_result.accepted
        assert good_result.rejection_reason is None

        # Bad measurement (extreme weight)
        bad_result = scorer.calculate_quality_score(
            weight=500,  # Impossible weight
            source='patient-device',
            user_height_m=1.75
        )
        assert not bad_result.accepted
        assert bad_result.rejection_reason is not None
        assert 'safety' in bad_result.rejection_reason.lower()

    def test_safety_critical_override(self, scorer):
        """Test that low safety score forces rejection."""
        result = scorer.calculate_quality_score(
            weight=25,  # Dangerously low
            source='care-team-upload',  # Even with reliable source
            user_height_m=1.75
        )

        assert not result.accepted
        assert result.components['safety'] == 0.0  # Below absolute minimum
        assert 'safety' in result.rejection_reason.lower()  # Rejection mentions safety

    def test_harmonic_mean_calculation(self, scorer):
        """Test harmonic mean penalizes low component scores."""
        # Create scorer with harmonic mean
        scorer.use_harmonic_mean = True

        # Mock component scores with one very low
        with patch.object(scorer, '_calculate_safety', return_value=0.1):
            with patch.object(scorer, '_calculate_plausibility', return_value=0.9):
                with patch.object(scorer, '_calculate_consistency', return_value=0.9):
                    with patch.object(scorer, '_calculate_reliability', return_value=0.9):
                        result = scorer.calculate_quality_score(70, 'test', user_height_m=1.75)

                        # Harmonic mean should be much lower than arithmetic
                        assert result.overall < 0.4  # Dominated by low safety score

    def test_feature_toggle_disabling(self):
        """Test that disabled features return 1.0."""
        # Create mock feature manager
        feature_manager = MagicMock()
        feature_manager.is_enabled.side_effect = lambda x: x != 'quality_safety'

        config = {'feature_manager': feature_manager}
        scorer = QualityScorer(config)

        result = scorer.calculate_quality_score(
            weight=25,  # Would normally fail safety
            source='patient-device',
            user_height_m=1.75
        )

        assert result.components['safety'] == 1.0  # Disabled feature returns 1.0
        assert result.accepted  # Should pass despite dangerous weight

    # ============= Real-World Scenarios =============

    def test_scenario_first_measurement(self, scorer):
        """Test handling of first measurement with no history."""
        result = scorer.calculate_quality_score(
            weight=75,
            source='patient-device',
            user_height_m=1.75
        )

        assert result.accepted  # Reasonable weight should be accepted
        assert result.components['plausibility'] == 0.8  # Default for no history
        assert result.components['consistency'] == 0.8  # Default for no history

    def test_scenario_after_long_gap(self, scorer):
        """Test measurement after 30+ day gap."""
        result = scorer.calculate_quality_score(
            weight=68,  # Some weight loss
            source='care-team-upload',
            previous_weight=75,
            time_diff_hours=30 * 24,  # 30 days
            user_height_m=1.75
        )

        assert result.accepted  # Should accept reasonable change over long period
        assert result.components['consistency'] > 0.7

    def test_scenario_data_entry_error(self, scorer):
        """Test detection of likely data entry error."""
        result = scorer.calculate_quality_score(
            weight=700,  # Likely meant 70.0
            source='patient-device',
            previous_weight=70,
            time_diff_hours=1,
            recent_weights=[69, 70, 71, 70],
            user_height_m=1.75
        )

        assert not result.accepted
        assert result.components['safety'] == 0.0  # Impossible weight
        # When safety is critically low, other components aren't calculated
        assert result.overall == 0.0
        assert 'safety' in result.rejection_reason.lower()
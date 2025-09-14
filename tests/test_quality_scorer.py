"""
Test suite for the Quality Scoring System.
Tests individual components and overall scoring logic.
"""

import unittest
from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.quality_scorer import QualityScorer, QualityScore, MeasurementHistory
    from src.constants import PHYSIOLOGICAL_LIMITS
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from quality_scorer import QualityScorer, QualityScore, MeasurementHistory
    from constants import PHYSIOLOGICAL_LIMITS


class TestQualityScorer(unittest.TestCase):
    """Test the QualityScorer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scorer = QualityScorer()
        self.default_height = 1.67
    
    def test_safety_score_normal_weight(self):
        """Test safety score for normal weight values."""
        score = self.scorer.calculate_safety_score(75.0, self.default_height)
        self.assertAlmostEqual(score, 1.0, places=2)
        
        score = self.scorer.calculate_safety_score(92.0, self.default_height)
        self.assertAlmostEqual(score, 1.0, places=2)
    
    def test_safety_score_extreme_weights(self):
        """Test safety score for extreme weight values."""
        score = self.scorer.calculate_safety_score(25.0, self.default_height)
        self.assertEqual(score, 0.0)
        
        score = self.scorer.calculate_safety_score(450.0, self.default_height)
        self.assertEqual(score, 0.0)
        
        score = self.scorer.calculate_safety_score(35.0, self.default_height)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)
    
    def test_plausibility_score_with_history(self):
        """Test plausibility score with recent weight history."""
        recent_weights = [92.0, 92.5, 91.8, 92.2, 92.1]
        
        score = self.scorer.calculate_plausibility_score(92.3, recent_weights)
        self.assertGreater(score, 0.9)
        
        score = self.scorer.calculate_plausibility_score(100.0, recent_weights)
        self.assertLess(score, 0.5)
        
        score = self.scorer.calculate_plausibility_score(110.0, recent_weights)
        self.assertLess(score, 0.3)
    
    def test_plausibility_score_no_history(self):
        """Test plausibility score without history."""
        score = self.scorer.calculate_plausibility_score(92.0, None)
        self.assertEqual(score, 0.8)
        
        score = self.scorer.calculate_plausibility_score(92.0, [])
        self.assertEqual(score, 0.8)
    
    def test_consistency_score_normal_change(self):
        """Test consistency score for normal weight changes."""
        score = self.scorer.calculate_consistency_score(
            weight=92.5,
            previous_weight=92.0,
            time_diff_hours=24
        )
        self.assertGreater(score, 0.9)
        
        score = self.scorer.calculate_consistency_score(
            weight=94.0,
            previous_weight=92.0,
            time_diff_hours=24
        )
        self.assertAlmostEqual(score, 1.0, places=1)
    
    def test_consistency_score_extreme_change(self):
        """Test consistency score for extreme weight changes."""
        score = self.scorer.calculate_consistency_score(
            weight=100.0,
            previous_weight=92.0,
            time_diff_hours=24
        )
        self.assertLess(score, 0.5)
        
        score = self.scorer.calculate_consistency_score(
            weight=110.0,
            previous_weight=92.0,
            time_diff_hours=24
        )
        self.assertLess(score, 0.2)
    
    def test_reliability_score_by_source(self):
        """Test reliability score for different sources."""
        score = self.scorer.calculate_reliability_score('care-team-upload')
        self.assertGreater(score, 0.9)
        
        score = self.scorer.calculate_reliability_score('patient-upload')
        self.assertGreater(score, 0.8)
        
        score = self.scorer.calculate_reliability_score('https://api.iglucose.com')
        self.assertLess(score, 0.6)
        
        score = self.scorer.calculate_reliability_score('unknown-source')
        self.assertAlmostEqual(score, 0.54, places=2)  # 0.6 * 0.9 multiplier
    
    def test_overall_quality_score_good_measurement(self):
        """Test overall quality score for a good measurement."""
        quality = self.scorer.calculate_quality_score(
            weight=92.5,
            source='patient-upload',
            previous_weight=92.0,
            time_diff_hours=24,
            recent_weights=[92.0, 91.8, 92.2, 92.1],
            user_height_m=self.default_height
        )
        
        self.assertGreater(quality.overall, 0.8)
        self.assertTrue(quality.accepted)
        self.assertIsNone(quality.rejection_reason)
    
    def test_overall_quality_score_outlier(self):
        """Test overall quality score for an outlier measurement."""
        quality = self.scorer.calculate_quality_score(
            weight=100.0,
            source='patient-upload',
            previous_weight=92.0,
            time_diff_hours=24,
            recent_weights=[92.0, 91.8, 92.2, 92.1],
            user_height_m=self.default_height
        )
        
        self.assertLess(quality.overall, 0.6)
        self.assertFalse(quality.accepted)
        self.assertIsNotNone(quality.rejection_reason)
    
    def test_overall_quality_score_extreme_outlier(self):
        """Test overall quality score for extreme outlier."""
        quality = self.scorer.calculate_quality_score(
            weight=110.0,
            source='patient-upload',
            previous_weight=92.0,
            time_diff_hours=24,
            recent_weights=[92.0, 91.8, 92.2, 92.1],
            user_height_m=self.default_height
        )
        
        self.assertLess(quality.overall, 0.4)
        self.assertFalse(quality.accepted)
        self.assertIn('below threshold', quality.rejection_reason)
    
    def test_safety_critical_rejection(self):
        """Test that safety critical values cause immediate rejection."""
        quality = self.scorer.calculate_quality_score(
            weight=500.0,
            source='care-team-upload',
            previous_weight=92.0,
            time_diff_hours=24,
            recent_weights=[92.0, 91.8, 92.2],
            user_height_m=self.default_height
        )
        
        self.assertEqual(quality.overall, 0.0)
        self.assertFalse(quality.accepted)
        self.assertIn('Safety score', quality.rejection_reason)
    
    def test_harmonic_vs_arithmetic_mean(self):
        """Test difference between harmonic and arithmetic mean."""
        scorer_harmonic = QualityScorer({'use_harmonic_mean': True})
        scorer_arithmetic = QualityScorer({'use_harmonic_mean': False})
        
        components = {
            'safety': 0.9,
            'plausibility': 0.3,
            'consistency': 0.8,
            'reliability': 0.85
        }
        
        harmonic = scorer_harmonic._weighted_harmonic_mean(
            components, scorer_harmonic.weights
        )
        arithmetic = scorer_arithmetic._weighted_arithmetic_mean(
            components, scorer_arithmetic.weights
        )
        
        self.assertLess(harmonic, arithmetic)
    
    def test_custom_threshold(self):
        """Test custom quality threshold."""
        scorer = QualityScorer({'threshold': 0.7})
        
        quality = scorer.calculate_quality_score(
            weight=94.0,
            source='patient-upload',
            previous_weight=92.0,
            time_diff_hours=24,
            recent_weights=[92.0, 91.8, 92.2],
            user_height_m=self.default_height
        )
        
        self.assertEqual(quality.threshold, 0.7)
        if quality.overall < 0.7:
            self.assertFalse(quality.accepted)


class TestMeasurementHistory(unittest.TestCase):
    """Test the MeasurementHistory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.history = MeasurementHistory(max_size=5)
    
    def test_add_measurements(self):
        """Test adding measurements to history."""
        now = datetime.now()
        
        self.history.add(92.0, now, 0.85)
        self.history.add(92.5, now + timedelta(days=1), 0.90)
        
        self.assertEqual(len(self.history.weights), 2)
        self.assertEqual(list(self.history.weights), [92.0, 92.5])
    
    def test_max_size_limit(self):
        """Test that history respects max size."""
        now = datetime.now()
        
        for i in range(10):
            self.history.add(90.0 + i, now + timedelta(days=i), 0.8)
        
        self.assertEqual(len(self.history.weights), 5)
        self.assertEqual(list(self.history.weights), [95.0, 96.0, 97.0, 98.0, 99.0])
    
    def test_get_recent_weights_with_quality_filter(self):
        """Test getting recent weights filtered by quality."""
        now = datetime.now()
        
        self.history.add(92.0, now, 0.85)
        self.history.add(100.0, now + timedelta(days=1), 0.4)
        self.history.add(92.5, now + timedelta(days=2), 0.9)
        
        good_weights = self.history.get_recent_weights(min_quality=0.6)
        self.assertEqual(good_weights, [92.0, 92.5])
    
    def test_get_statistics(self):
        """Test calculating statistics from history."""
        now = datetime.now()
        
        weights = [92.0, 92.5, 91.8, 92.2, 92.1]
        for i, w in enumerate(weights):
            self.history.add(w, now + timedelta(days=i), 0.8)
        
        stats = self.history.get_statistics()
        
        self.assertAlmostEqual(stats['mean'], 92.12, places=1)
        self.assertAlmostEqual(stats['median'], 92.1, places=1)
        self.assertEqual(stats['count'], 5)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)


class TestIntegrationScenarios(unittest.TestCase):
    """Test real-world scenarios from the investigation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scorer = QualityScorer()
        self.baseline_92kg_history = [92.0, 91.8, 92.2, 92.1, 91.9]
    
    def test_scenario_100kg_from_92kg_baseline(self):
        """Test the problematic case: 100kg from 92kg baseline."""
        quality = self.scorer.calculate_quality_score(
            weight=100.0,
            source='patient-upload',
            previous_weight=92.0,
            time_diff_hours=24,
            recent_weights=self.baseline_92kg_history,
            user_height_m=1.67
        )
        
        self.assertLess(quality.overall, 0.6, 
                       "100kg from 92kg baseline should be rejected")
        self.assertFalse(quality.accepted)
        
        self.assertLess(quality.components['plausibility'], 0.5,
                       "Plausibility should be low for 8kg jump")
        self.assertLess(quality.components['consistency'], 0.5,
                       "Consistency should be low for 8kg/day change")
    
    def test_scenario_110kg_from_92kg_baseline(self):
        """Test extreme case: 110kg from 92kg baseline."""
        quality = self.scorer.calculate_quality_score(
            weight=110.0,
            source='patient-upload',
            previous_weight=92.0,
            time_diff_hours=24,
            recent_weights=self.baseline_92kg_history,
            user_height_m=1.67
        )
        
        self.assertLess(quality.overall, 0.4,
                       "110kg from 92kg baseline should be strongly rejected")
        self.assertFalse(quality.accepted)
        
        self.assertLess(quality.components['consistency'], 0.2,
                       "Consistency should be very low for 18kg/day change")
    
    def test_scenario_gradual_increase(self):
        """Test gradual weight increase should be accepted."""
        quality = self.scorer.calculate_quality_score(
            weight=93.0,
            source='patient-upload',
            previous_weight=92.5,
            time_diff_hours=24,
            recent_weights=[91.5, 91.8, 92.0, 92.3, 92.5],
            user_height_m=1.67
        )
        
        self.assertGreater(quality.overall, 0.7,
                          "Gradual increase should be accepted")
        self.assertTrue(quality.accepted)
    
    def test_scenario_after_gap(self):
        """Test measurement after long gap."""
        quality = self.scorer.calculate_quality_score(
            weight=95.0,
            source='patient-upload',
            previous_weight=92.0,
            time_diff_hours=30 * 24,
            recent_weights=None,
            user_height_m=1.67
        )
        
        self.assertGreater(quality.components['consistency'], 0.9,
                          "3kg change over 30 days should be consistent")


if __name__ == '__main__':
    unittest.main()
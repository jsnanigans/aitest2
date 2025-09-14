#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from datetime import datetime, timedelta
from src.validation import PhysiologicalValidator, BMIValidator, ThresholdCalculator

class TestValidation(unittest.TestCase):
    
    def setUp(self):
        self.config = {
            'min_weight': 30,
            'max_weight': 400,
            'max_daily_change': 0.05,
            'physiological': {
                'enable_physiological_limits': True,
                'max_change_1h_percent': 0.02,
                'max_change_6h_percent': 0.025,
                'max_change_24h_percent': 0.035,
                'max_change_1h_absolute': 3.0,
                'max_change_6h_absolute': 4.0,
                'max_change_24h_absolute': 5.0,
                'max_sustained_daily': 1.5,
                'session_timeout_minutes': 5,
                'session_variance_threshold': 5.0,
                'limit_tolerance': 0.1,
                'sustained_tolerance': 0.25
            }
        }
    
    def test_weight_bounds_validation(self):
        is_valid, reason = PhysiologicalValidator.validate_weight(
            25.0, self.config, None, None
        )
        self.assertFalse(is_valid)
        self.assertIn('outside bounds', reason)
        
        is_valid, reason = PhysiologicalValidator.validate_weight(
            70.0, self.config, None, None
        )
        self.assertTrue(is_valid)
        self.assertIsNone(reason)
    
    def test_physiological_limits(self):
        limit_1h, reason = PhysiologicalValidator.get_physiological_limit(
            0.5, 70.0, self.config
        )
        self.assertLessEqual(limit_1h, 3.0)
        
        limit_24h, reason = PhysiologicalValidator.get_physiological_limit(
            24.0, 70.0, self.config
        )
        self.assertLessEqual(limit_24h, 5.0)
        
        limit_sustained, reason = PhysiologicalValidator.get_physiological_limit(
            48.0, 70.0, self.config
        )
        self.assertGreater(limit_sustained, limit_24h)
    
    def test_bmi_calculation(self):
        bmi = BMIValidator.calculate_bmi(70.0, 1.75)
        self.assertAlmostEqual(bmi, 22.86, places=1)
        
        bmi_none = BMIValidator.calculate_bmi(70.0, None)
        self.assertIsNone(bmi_none)
    
    def test_threshold_calculator(self):
        threshold = ThresholdCalculator.get_extreme_deviation_threshold(
            source="patient-upload",
            time_gap_days=1,
            current_weight=70.0,
            unit='percentage'
        )
        
        self.assertIsNotNone(threshold)
        self.assertGreater(threshold.value, 0)
        self.assertEqual(threshold.unit, 'percentage')

if __name__ == '__main__':
    unittest.main()
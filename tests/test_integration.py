#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import tempfile
import csv
from datetime import datetime, timedelta
from src.processor import process_weight_enhanced
from src.database import get_state_db

class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        self.db = get_state_db()
        self.user_id = "integration_test_user"
        self.processing_config = {
            'min_weight': 30,
            'max_weight': 400,
            'extreme_threshold': 0.15,
            'max_daily_change': 0.05,
            'min_valid_bmi': 10,
            'max_valid_bmi': 60,
            'config': {
                'adaptive_noise': {
                    'enabled': True,
                    'multipliers': {
                        'patient-upload': 1.0,
                        'patient-device': 2.5
                    },
                    'default_multiplier': 1.5
                }
            }
        }
        self.kalman_config = {
            'initial_variance': 1.0,
            'transition_covariance_weight': 0.1,
            'transition_covariance_trend': 0.001,
            'observation_covariance': 1.0,
            'reset_gap_days': 30
        }
    
    def tearDown(self):
        self.db.delete_state(self.user_id)
    
    def test_full_processing_pipeline(self):
        base_time = datetime.now()
        weights = [70.0, 70.2, 70.1, 70.3, 70.0, 69.8]
        
        results = []
        for i, weight in enumerate(weights):
            result = process_weight_enhanced(
                user_id=self.user_id,
                weight=weight,
                timestamp=base_time + timedelta(days=i),
                source="patient-upload",
                processing_config=self.processing_config,
                kalman_config=self.kalman_config,
                unit='kg'
            )
            results.append(result)
        
        self.assertEqual(len(results), 6)
        
        for result in results:
            self.assertIsNotNone(result)
            self.assertIn('accepted', result)
            self.assertIn('raw_weight', result)
        
        accepted_count = sum(1 for r in results if r['accepted'])
        self.assertGreater(accepted_count, 4)
    
    def test_unit_conversion(self):
        result = process_weight_enhanced(
            user_id=self.user_id,
            weight=154.0,
            timestamp=datetime.now(),
            source="test",
            processing_config=self.processing_config,
            kalman_config=self.kalman_config,
            unit='lb'
        )
        
        self.assertIsNotNone(result)
        self.assertIn('bmi_details', result)
        self.assertTrue(result['bmi_details']['unit_converted'])
    
    def test_adaptive_noise_by_source(self):
        base_time = datetime.now()
        
        result1 = process_weight_enhanced(
            user_id=self.user_id + "_1",
            weight=70.0,
            timestamp=base_time,
            source="patient-upload",
            processing_config=self.processing_config,
            kalman_config=self.kalman_config,
            unit='kg'
        )
        
        result2 = process_weight_enhanced(
            user_id=self.user_id + "_2",
            weight=70.0,
            timestamp=base_time,
            source="patient-device",
            processing_config=self.processing_config,
            kalman_config=self.kalman_config,
            unit='kg'
        )
        
        self.assertNotEqual(
            result1['measurement_noise_used'],
            result2['measurement_noise_used']
        )
        
        self.db.delete_state(self.user_id + "_1")
        self.db.delete_state(self.user_id + "_2")

if __name__ == '__main__':
    unittest.main()
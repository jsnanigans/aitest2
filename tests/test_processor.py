#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from datetime import datetime, timedelta
from src.processor import WeightProcessor, process_weight_enhanced
from src.database import get_state_db

class TestProcessor(unittest.TestCase):
    
    def setUp(self):
        self.db = get_state_db()
        self.user_id = "test_user_001"
        self.processing_config = {
            'min_weight': 30,
            'max_weight': 400,
            'extreme_threshold': 0.15,
            'max_daily_change': 0.05
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
    
    def test_first_measurement_accepted(self):
        result = WeightProcessor.process_weight(
            user_id=self.user_id,
            weight=70.0,
            timestamp=datetime.now(),
            source="test",
            processing_config=self.processing_config,
            kalman_config=self.kalman_config,
            db=self.db
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['raw_weight'], 70.0)
    
    def test_extreme_deviation_rejected(self):
        timestamp1 = datetime.now()
        WeightProcessor.process_weight(
            user_id=self.user_id,
            weight=70.0,
            timestamp=timestamp1,
            source="test",
            processing_config=self.processing_config,
            kalman_config=self.kalman_config,
            db=self.db
        )
        
        # Use a more reasonable deviation that passes physiological but fails Kalman
        result = WeightProcessor.process_weight(
            user_id=self.user_id,
            weight=85.0,  # 21% change - should fail extreme threshold of 15%
            timestamp=timestamp1 + timedelta(days=1),
            source="test",
            processing_config=self.processing_config,
            kalman_config=self.kalman_config,
            db=self.db
        )
        
        self.assertFalse(result['accepted'])
        # Accept either physiological or deviation rejection
        reason = result.get('reason', '').lower()
        self.assertTrue('deviation' in reason or 'limit' in reason or 'change' in reason)
    
    def test_gap_reset(self):
        timestamp1 = datetime.now()
        WeightProcessor.process_weight(
            user_id=self.user_id,
            weight=70.0,
            timestamp=timestamp1,
            source="test",
            processing_config=self.processing_config,
            kalman_config=self.kalman_config,
            db=self.db
        )
        
        result = WeightProcessor.process_weight(
            user_id=self.user_id,
            weight=75.0,
            timestamp=timestamp1 + timedelta(days=35),
            source="test",
            processing_config=self.processing_config,
            kalman_config=self.kalman_config,
            db=self.db
        )
        
        self.assertTrue(result['accepted'])
        self.assertTrue(result.get('was_reset', False))

if __name__ == '__main__':
    unittest.main()
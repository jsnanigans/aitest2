#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import numpy as np
from datetime import datetime, timedelta
from src.kalman import KalmanFilterManager

class TestKalman(unittest.TestCase):
    
    def setUp(self):
        self.kalman_config = {
            'initial_variance': 1.0,
            'transition_covariance_weight': 0.1,
            'transition_covariance_trend': 0.001,
            'observation_covariance': 1.0
        }
    
    def test_initialize_immediate(self):
        weight = 70.0
        timestamp = datetime.now()
        
        state = KalmanFilterManager.initialize_immediate(
            weight, timestamp, self.kalman_config
        )
        
        self.assertIsNotNone(state['kalman_params'])
        self.assertIsNotNone(state['last_state'])
        self.assertEqual(state['last_timestamp'], timestamp)
        
        last_state = state['last_state']
        self.assertAlmostEqual(last_state[0][0], weight, places=5)
        self.assertAlmostEqual(last_state[0][1], 0.0, places=5)
    
    def test_update_state(self):
        initial_state = KalmanFilterManager.initialize_immediate(
            70.0, datetime.now(), self.kalman_config
        )
        
        new_weight = 70.5
        new_timestamp = datetime.now() + timedelta(days=1)
        
        updated_state = KalmanFilterManager.update_state(
            initial_state, new_weight, new_timestamp, "test", {}, 1.0
        )
        
        self.assertIsNotNone(updated_state['last_state'])
        self.assertEqual(updated_state['last_timestamp'], new_timestamp)
    
    def test_confidence_calculation(self):
        confidence_low = KalmanFilterManager.calculate_confidence(3.0)
        confidence_high = KalmanFilterManager.calculate_confidence(0.5)
        
        self.assertLess(confidence_low, 0.5)
        self.assertGreater(confidence_high, 0.5)
        self.assertLessEqual(confidence_low, 1.0)
        self.assertGreaterEqual(confidence_low, 0.0)

if __name__ == '__main__':
    unittest.main()
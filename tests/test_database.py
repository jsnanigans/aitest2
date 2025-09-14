#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import numpy as np
from datetime import datetime
from src.database import ProcessorStateDB

class TestDatabase(unittest.TestCase):
    
    def setUp(self):
        self.db = ProcessorStateDB()
        self.user_id = "test_user_001"
    
    def tearDown(self):
        self.db.delete_state(self.user_id)
    
    def test_create_initial_state(self):
        state = self.db.create_initial_state()
        
        self.assertIsNone(state['last_state'])
        self.assertIsNone(state['last_covariance'])
        self.assertIsNone(state['last_timestamp'])
        self.assertIsNone(state['kalman_params'])
    
    def test_save_and_retrieve_state(self):
        test_state = {
            'last_state': np.array([[70.0, 0.0]]),
            'last_covariance': np.array([[[1.0, 0.0], [0.0, 0.001]]]),
            'last_timestamp': datetime.now(),
            'kalman_params': {'test': 'params'}
        }
        
        self.db.save_state(self.user_id, test_state)
        retrieved = self.db.get_state(self.user_id)
        
        self.assertIsNotNone(retrieved)
        self.assertIsNotNone(retrieved['last_state'])
        self.assertEqual(retrieved['kalman_params']['test'], 'params')
    
    def test_delete_state(self):
        test_state = self.db.create_initial_state()
        self.db.save_state(self.user_id, test_state)
        
        deleted = self.db.delete_state(self.user_id)
        self.assertTrue(deleted)
        
        retrieved = self.db.get_state(self.user_id)
        self.assertIsNone(retrieved)
    
    def test_get_all_users(self):
        self.db.save_state("user1", self.db.create_initial_state())
        self.db.save_state("user2", self.db.create_initial_state())
        
        users = self.db.get_all_users()
        self.assertIn("user1", users)
        self.assertIn("user2", users)
        
        self.db.delete_state("user1")
        self.db.delete_state("user2")

if __name__ == '__main__':
    unittest.main()
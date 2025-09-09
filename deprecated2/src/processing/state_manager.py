"""
State persistence manager for real-time production deployment.
Handles saving and restoring Kalman filter state to/from database or cache.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import pickle
import numpy as np

from src.core.types import KalmanState
from src.filters.layer3_kalman import PureKalmanFilter

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages Kalman filter state persistence for real-time processing.
    In production, this would interface with a database or Redis cache.
    """
    
    def __init__(self, storage_path: str = "output/states"):
        """
        Initialize state manager.
        
        Args:
            storage_path: Directory for state files (in production, this would be DB connection)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def save_state(self, user_id: str, state_data: Dict[str, Any]) -> bool:
        """
        Save Kalman filter state for a user.
        
        Args:
            user_id: Unique user identifier
            state_data: Dictionary containing:
                - kalman_state: Current Kalman filter state
                - baseline: Baseline parameters
                - last_processed: Timestamp of last processed measurement
                - measurement_count: Total measurements processed
                
        Returns:
            True if successful, False otherwise
        """
        try:
            state_file = self.storage_path / f"user_{user_id}_state.json"
            
            # Convert to serializable format
            serializable_state = {
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'kalman': {
                    'weight': state_data['kalman_state'].weight,
                    'trend': state_data['kalman_state'].trend,
                    'covariance': state_data['kalman_state'].covariance,
                    'measurement_count': state_data['kalman_state'].measurement_count,
                    'last_timestamp': state_data['kalman_state'].timestamp.isoformat()
                },
                'baseline': state_data.get('baseline', {}),
                'last_processed': state_data.get('last_processed', datetime.now()).isoformat(),
                'measurement_count': state_data.get('measurement_count', 0)
            }
            
            with open(state_file, 'w') as f:
                json.dump(serializable_state, f, indent=2)
                
            logger.debug(f"Saved state for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state for user {user_id}: {e}")
            return False
    
    def load_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Load Kalman filter state for a user.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Dictionary with state data or None if not found
        """
        try:
            state_file = self.storage_path / f"user_{user_id}_state.json"
            
            if not state_file.exists():
                logger.debug(f"No saved state found for user {user_id}")
                return None
                
            with open(state_file, 'r') as f:
                data = json.load(f)
                
            # Reconstruct KalmanState object
            kalman_data = data['kalman']
            kalman_state = KalmanState(
                weight=kalman_data['weight'],
                trend=kalman_data['trend'],
                covariance=kalman_data['covariance'],
                timestamp=datetime.fromisoformat(kalman_data['last_timestamp']),
                measurement_count=kalman_data['measurement_count']
            )
            
            return {
                'kalman_state': kalman_state,
                'baseline': data.get('baseline', {}),
                'last_processed': datetime.fromisoformat(data['last_processed']),
                'measurement_count': data.get('measurement_count', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to load state for user {user_id}: {e}")
            return None
    
    def delete_state(self, user_id: str) -> bool:
        """
        Delete saved state for a user (e.g., for re-initialization).
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            state_file = self.storage_path / f"user_{user_id}_state.json"
            if state_file.exists():
                state_file.unlink()
                logger.debug(f"Deleted state for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete state for user {user_id}: {e}")
            return False
    
    def check_gap(self, last_processed: datetime, current_time: datetime, 
                  gap_threshold_days: int = 30) -> bool:
        """
        Check if there's a significant gap requiring re-initialization.
        
        Args:
            last_processed: Timestamp of last processed measurement
            current_time: Current measurement timestamp
            gap_threshold_days: Days threshold for gap detection
            
        Returns:
            True if gap detected, False otherwise
        """
        gap_days = (current_time - last_processed).days
        return gap_days > gap_threshold_days


class RealTimeKalmanProcessor:
    """
    Real-time Kalman filter processor for production use.
    Processes single measurements using only saved state.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize real-time processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.state_manager = StateManager(
            self.config.get('state_storage_path', 'output/states')
        )
        
        # Kalman configuration
        self.kalman_config = {
            'process_noise_weight': config.get('kalman', {}).get('process_noise_weight', 0.5),
            'process_noise_trend': config.get('kalman', {}).get('process_noise_trend', 0.01),
            'validation_gamma': config.get('validation_gamma', 3.0)
        }
        
    def process_single_measurement(self, user_id: str, weight: float, 
                                  timestamp: datetime, source: str = 'unknown') -> Dict[str, Any]:
        """
        Process a single weight measurement in real-time.
        
        This is the main entry point for production use.
        It loads the saved state, processes the measurement, and saves the updated state.
        
        Args:
            user_id: Unique user identifier
            weight: Weight measurement in kg
            timestamp: Measurement timestamp
            source: Data source identifier
            
        Returns:
            Dictionary with processing results including:
                - accepted: Whether measurement was accepted
                - confidence: Confidence score (0-1)
                - filtered_weight: Kalman filtered weight estimate
                - trend_kg_per_week: Current weight trend
                - prediction_error: Innovation/prediction error
                - needs_initialization: True if user needs baseline establishment
        """
        # Load saved state
        saved_state = self.state_manager.load_state(user_id)
        
        if saved_state is None:
            # New user - needs initialization
            return {
                'accepted': False,
                'confidence': 0.0,
                'filtered_weight': None,
                'trend_kg_per_week': None,
                'prediction_error': None,
                'needs_initialization': True,
                'message': 'User needs baseline establishment (3+ measurements over 7 days)'
            }
        
        # Check for significant gap
        if self.state_manager.check_gap(saved_state['last_processed'], timestamp):
            # Gap detected - mark for re-initialization
            self.state_manager.delete_state(user_id)
            return {
                'accepted': False,
                'confidence': 0.0,
                'filtered_weight': None,
                'trend_kg_per_week': None,
                'prediction_error': None,
                'needs_initialization': True,
                'message': 'Significant gap detected - re-initialization required'
            }
        
        # Restore Kalman filter from saved state
        kalman_state = saved_state['kalman_state']
        baseline = saved_state['baseline']
        
        # Initialize Kalman filter with saved state
        kalman = PureKalmanFilter(
            initial_weight=kalman_state.weight,
            initial_variance=kalman_state.covariance[0][0],
            process_noise_weight=self.kalman_config['process_noise_weight'],
            process_noise_trend=self.kalman_config['process_noise_trend'],
            measurement_noise=baseline.get('measurement_noise', 1.0)
        )
        
        # Restore full state
        kalman.state = np.array([kalman_state.weight, kalman_state.trend])
        kalman.covariance = np.array(kalman_state.covariance)
        kalman.measurement_count = kalman_state.measurement_count
        kalman.last_timestamp = kalman_state.timestamp
        
        # Calculate time delta
        time_delta_days = (timestamp - kalman.last_timestamp).total_seconds() / 86400.0
        time_delta_days = max(0.1, time_delta_days)
        
        # PREDICT step
        predicted_weight, prediction_variance = kalman.predict(time_delta_days)
        
        # Calculate innovation (prediction error)
        innovation = weight - predicted_weight
        innovation_variance = prediction_variance + kalman.R
        
        # Normalized innovation for validation gate
        normalized_innovation = abs(innovation) / np.sqrt(innovation_variance)
        
        # Validation gate
        accepted = normalized_innovation <= self.kalman_config['validation_gamma']
        
        # Calculate confidence based on normalized innovation
        if normalized_innovation <= 1.0:
            confidence = 0.95
        elif normalized_innovation <= 2.0:
            confidence = 0.8
        elif normalized_innovation <= 3.0:
            confidence = 0.5
        else:
            confidence = 0.1
        
        # UPDATE step (only if accepted)
        if accepted:
            update_results = kalman.update(weight)
            filtered_weight = update_results['filtered_weight']
            trend = update_results['trend_kg_per_day']
        else:
            # Rejected - use prediction as best estimate
            filtered_weight = predicted_weight
            trend = kalman.state[1]
        
        # Save updated state
        new_kalman_state = kalman.get_state()
        new_kalman_state.timestamp = timestamp  # Update to current measurement time
        
        self.state_manager.save_state(user_id, {
            'kalman_state': new_kalman_state,
            'baseline': baseline,
            'last_processed': timestamp,
            'measurement_count': saved_state['measurement_count'] + 1
        })
        
        # Return results
        return {
            'accepted': accepted,
            'confidence': confidence,
            'filtered_weight': filtered_weight,
            'trend_kg_per_week': trend * 7.0,
            'prediction_error': innovation,
            'normalized_innovation': normalized_innovation,
            'needs_initialization': False,
            'measurement_count': saved_state['measurement_count'] + 1
        }
    
    def initialize_user(self, user_id: str, measurements: List[Dict[str, Any]]) -> bool:
        """
        Initialize a new user with baseline measurements.
        
        This should be called when a user has collected enough initial measurements
        (typically 3+ over 7 days) to establish a baseline.
        
        Args:
            user_id: Unique user identifier
            measurements: List of dictionaries with 'weight', 'timestamp', 'source'
            
        Returns:
            True if initialization successful, False otherwise
        """
        if len(measurements) < 3:
            logger.warning(f"Insufficient measurements for initialization: {len(measurements)}")
            return False
        
        # Calculate baseline statistics
        weights = [m['weight'] for m in measurements]
        baseline_weight = np.median(weights)
        
        # Calculate robust variance using MAD
        mad = np.median(np.abs(weights - baseline_weight))
        measurement_variance = (1.4826 * mad) ** 2
        measurement_noise = np.sqrt(measurement_variance)
        
        # Create initial Kalman state
        initial_state = KalmanState(
            weight=baseline_weight,
            trend=0.0,
            covariance=[[measurement_variance, 0.0], [0.0, 0.001]],
            timestamp=measurements[-1]['timestamp'],
            measurement_count=len(measurements)
        )
        
        # Save initial state
        self.state_manager.save_state(user_id, {
            'kalman_state': initial_state,
            'baseline': {
                'weight': baseline_weight,
                'variance': measurement_variance,
                'measurement_noise': measurement_noise,
                'confidence': 'high' if len(measurements) >= 5 else 'medium'
            },
            'last_processed': measurements[-1]['timestamp'],
            'measurement_count': len(measurements)
        })
        
        logger.info(f"Initialized user {user_id} with baseline {baseline_weight:.1f}kg")
        return True



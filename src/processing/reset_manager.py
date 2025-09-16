"""
Reset Manager for parameterized reset handling.
Supports hard, initial, and soft resets with different adaptation parameters.
"""

from enum import Enum
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
import numpy as np

class ResetType(Enum):
    """Types of resets with different adaptation strategies."""
    HARD = "hard"      # 30+ day gaps - aggressive adaptation
    INITIAL = "initial" # First measurement - most aggressive adaptation
    SOFT = "soft"      # Manual data entry - gentle adaptation

MANUAL_DATA_SOURCES = {
    'internal-questionnaire',
    'initial-questionnaire',
    'questionnaire',
    'patient-upload',
    'user-upload',
    'care-team-upload',
    'care-team-entry'
}

class ResetManager:
    """Manages different types of resets with configurable parameters."""
    
    @staticmethod
    def should_trigger_reset(
        state: Dict[str, Any],
        weight: float,
        timestamp: datetime,
        source: str,
        config: Dict[str, Any]
    ) -> Optional[ResetType]:
        """
        Determine if and what type of reset to trigger.
        
        Priority order:
        1. Initial (no Kalman params)
        2. Hard (30+ day gap)
        3. Soft (manual data with significant change)
        """
        
        # 1. Check for initial reset (no Kalman params yet)
        if not state or not state.get('kalman_params'):
            return ResetType.INITIAL
        
        # 2. Check for hard reset (30+ day gap)
        hard_config = config.get('kalman', {}).get('reset', {}).get('hard', {})
        if hard_config.get('enabled', True):
            last_timestamp = state.get('last_accepted_timestamp') or state.get('last_timestamp')
            if last_timestamp:
                if isinstance(last_timestamp, str):
                    last_timestamp = datetime.fromisoformat(last_timestamp)
                gap_days = (timestamp - last_timestamp).total_seconds() / 86400.0
                threshold = hard_config.get('gap_threshold_days', 30)
                if gap_days >= threshold:
                    return ResetType.HARD
        
        # 3. Check for soft reset (manual data with significant change)
        soft_config = config.get('kalman', {}).get('reset', {}).get('soft', {})
        if soft_config.get('enabled', True):
            if source in MANUAL_DATA_SOURCES or source in soft_config.get('trigger_sources', []):
                last_weight = state.get('last_raw_weight') or state.get('last_accepted_weight')
                if last_weight:
                    weight_change = abs(weight - last_weight)
                    min_change = soft_config.get('min_weight_change_kg', 5)
                    
                    if weight_change >= min_change:
                        cooldown_days = soft_config.get('cooldown_days', 3)
                        last_reset = ResetManager.get_last_reset_timestamp(state)
                        if not last_reset or (timestamp - last_reset).total_seconds() / 86400.0 > cooldown_days:
                            return ResetType.SOFT
        
        return None
    
    @staticmethod
    def get_last_reset_timestamp(state: Dict[str, Any]) -> Optional[datetime]:
        """Get timestamp of the most recent reset."""
        reset_events = state.get('reset_events', [])
        if reset_events:
            last_reset = reset_events[-1]
            timestamp = last_reset.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                return timestamp
        
        reset_timestamp = state.get('reset_timestamp')
        if reset_timestamp:
            if isinstance(reset_timestamp, str):
                reset_timestamp = datetime.fromisoformat(reset_timestamp)
            return reset_timestamp
        
        return None
    
    @staticmethod
    def get_reset_parameters(reset_type: ResetType, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get parameters for specific reset type from config.
        
        Returns dict with adaptation parameters using new naming convention.
        """
        
        # Default parameters for each type (using new names)
        defaults = {
            ResetType.INITIAL: {
                # Multipliers for Kalman parameters
                'initial_variance_multiplier': 10,
                'weight_noise_multiplier': 50,
                'trend_noise_multiplier': 500,
                'observation_noise_multiplier': 0.3,
                # Adaptation duration
                'adaptation_measurements': 20,
                'adaptation_days': 21,
                'adaptation_decay_rate': 1.5,
                # Quality scoring
                'quality_acceptance_threshold': 0.25,
                'quality_safety_weight': 0.50,
                'quality_plausibility_weight': 0.05,
                'quality_consistency_weight': 0.05,
                'quality_reliability_weight': 0.40
            },
            ResetType.HARD: {
                # Multipliers for Kalman parameters
                'initial_variance_multiplier': 5,
                'weight_noise_multiplier': 20,
                'trend_noise_multiplier': 200,
                'observation_noise_multiplier': 0.5,
                # Adaptation duration
                'adaptation_measurements': 10,
                'adaptation_days': 7,
                'adaptation_decay_rate': 2.5,
                # Quality scoring
                'quality_acceptance_threshold': 0.35,
                'quality_safety_weight': 0.45,
                'quality_plausibility_weight': 0.10,
                'quality_consistency_weight': 0.10,
                'quality_reliability_weight': 0.35
            },
            ResetType.SOFT: {
                # Multipliers for Kalman parameters
                'initial_variance_multiplier': 2,
                'weight_noise_multiplier': 5,
                'trend_noise_multiplier': 20,
                'observation_noise_multiplier': 0.7,
                # Adaptation duration
                'adaptation_measurements': 15,
                'adaptation_days': 10,
                'adaptation_decay_rate': 4,
                # Quality scoring
                'quality_acceptance_threshold': 0.45,
                'quality_safety_weight': 0.40,
                'quality_plausibility_weight': 0.15,
                'quality_consistency_weight': 0.15,
                'quality_reliability_weight': 0.30
            }
        }
        
        # Get from config or use defaults
        reset_config = config.get('kalman', {}).get('reset', {}).get(reset_type.value, {})
        default_params = defaults[reset_type]
        
        # Merge config with defaults
        params = {}
        for key, default_value in default_params.items():
            params[key] = reset_config.get(key, default_value)
        
        return params
    
    @staticmethod
    def perform_reset(
        state: Dict[str, Any],
        reset_type: ResetType,
        timestamp: datetime,
        weight: float,
        source: str,
        config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform reset with appropriate parameters.
        
        Returns:
            Tuple of (new_state, reset_event)
        """
        
        # Get parameters for this reset type
        reset_params = ResetManager.get_reset_parameters(reset_type, config)
        
        # Calculate gap if applicable
        gap_days = None
        last_timestamp = state.get('last_accepted_timestamp') or state.get('last_timestamp')
        if last_timestamp:
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp)
            gap_days = (timestamp - last_timestamp).total_seconds() / 86400.0
        
        # Create reset event
        reset_event = {
            'timestamp': timestamp,
            'type': reset_type.value,
            'source': source,
            'weight': weight,
            'last_weight': state.get('last_raw_weight'),
            'gap_days': gap_days,
            'reason': ResetManager.get_reset_reason(reset_type, gap_days, weight, state),
            'parameters': reset_params
        }
        
        # Create new state with reset
        new_state = {
            'kalman_params': None,
            'last_state': None,
            'last_covariance': None,
            'measurements_since_reset': 0,
            'reset_type': reset_type.value,
            'reset_parameters': reset_params,
            'reset_timestamp': timestamp,
            'reset_events': state.get('reset_events', []) + [reset_event],
            'last_timestamp': state.get('last_timestamp'),
            'last_source': state.get('last_source'),
            'last_raw_weight': state.get('last_raw_weight'),
            'last_accepted_timestamp': state.get('last_accepted_timestamp'),
            'measurement_history': []
        }
        
        return new_state, reset_event
    
    @staticmethod
    def get_reset_reason(
        reset_type: ResetType,
        gap_days: Optional[float],
        weight: float,
        state: Dict[str, Any]
    ) -> str:
        """Generate human-readable reset reason."""
        
        if reset_type == ResetType.INITIAL:
            return "initial_measurement"
        elif reset_type == ResetType.HARD:
            return f"gap_exceeded_{gap_days:.0f}_days" if gap_days else "gap_exceeded"
        elif reset_type == ResetType.SOFT:
            last_weight = state.get('last_raw_weight')
            if last_weight:
                change = abs(weight - last_weight)
                return f"manual_entry_change_{change:.1f}kg"
            return "manual_entry"
        else:
            return "unknown"
    
    @staticmethod
    def is_in_adaptive_period(
        state: Dict[str, Any],
        timestamp: datetime
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if currently in adaptive period and return parameters if so.
        
        Returns:
            Tuple of (is_adaptive, parameters_dict)
        """
        
        reset_timestamp = state.get('reset_timestamp')
        if not reset_timestamp:
            return False, None
        
        if isinstance(reset_timestamp, str):
            reset_timestamp = datetime.fromisoformat(reset_timestamp)
        
        reset_params = state.get('reset_parameters', {})
        if not reset_params:
            return False, None
        
        # Check measurements-based
        measurements_since = state.get('measurements_since_reset', 0)
        adaptation_measurements = reset_params.get('adaptation_measurements', 10)
        
        # Check time-based
        days_since = (timestamp - reset_timestamp).total_seconds() / 86400.0
        adaptation_days = reset_params.get('adaptation_days', 7)
        
        # In adaptive period if either condition is met
        if measurements_since < adaptation_measurements or days_since < adaptation_days:
            return True, reset_params
        
        return False, None
    
    @staticmethod
    def get_adaptive_factor(
        state: Dict[str, Any],
        timestamp: datetime
    ) -> float:
        """
        Calculate adaptive factor (0-1) based on time/measurements since reset.
        0 = fully adaptive, 1 = normal operation
        """
        
        reset_timestamp = state.get('reset_timestamp')
        if not reset_timestamp:
            return 1.0
        
        if isinstance(reset_timestamp, str):
            reset_timestamp = datetime.fromisoformat(reset_timestamp)
        
        reset_params = state.get('reset_parameters', {})
        decay_rate = reset_params.get('adaptation_decay_rate', 3)
        
        # Use measurements-based decay
        measurements_since = state.get('measurements_since_reset', 0)
        factor = 1.0 - np.exp(-measurements_since / decay_rate)
        
        return min(1.0, max(0.0, factor))

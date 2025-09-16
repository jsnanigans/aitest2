"""
Simplified weight processor with flattened pipeline.
Single processing function with clear flow.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple
import numpy as np

from .database import get_state_db
from .kalman import (
    KalmanFilterManager,
    ResetManager,
    ResetType,
    get_adaptive_kalman_params,
    get_reset_timestamp
)
from .validation import PhysiologicalValidator, BMIValidator, ThresholdCalculator, DataQualityPreprocessor
from .constants import QUESTIONNAIRE_SOURCES
try:
    from .quality_scorer import QualityScorer, MeasurementHistory
except ImportError:
    from quality_scorer import QualityScorer, MeasurementHistory

# Hard-coded constants (to be moved to constants.py)
MIN_WEIGHT = 30.0
MAX_WEIGHT = 400.0
MIN_VALID_BMI = 10.0
MAX_VALID_BMI = 90.0

SOURCE_NOISE_MULTIPLIERS = {
    "patient-upload": 1.0,
    "care-team-upload": 1.2,
    "internal-questionnaire": 1.6,
    "initial-questionnaire": 1.6,
    "questionnaire": 1.6,
    "https://connectivehealth.io": 2.2,
    "patient-device": 2.5,
    "https://api.iglucose.com": 2.6,
}
from .constants import KALMAN_DEFAULTS, categorize_rejection_enhanced, get_rejection_severity


def check_and_reset_for_gap(state: Dict[str, Any], current_timestamp: datetime, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Check for 30+ day gap and reset if needed."""
    reset_config = config.get('kalman', {}).get('reset', {})
    if not reset_config.get('enabled', True):
        return state, None
    
    gap_threshold_days = reset_config.get('gap_threshold_days', 30)
    
    # Check both last_accepted_timestamp and last_timestamp for backward compatibility
    last_timestamp = state.get('last_accepted_timestamp') or state.get('last_timestamp')
    if last_timestamp:
        if isinstance(last_timestamp, str):
            last_timestamp = datetime.fromisoformat(last_timestamp)
        
        gap_days = (current_timestamp - last_timestamp).total_seconds() / 86400.0
        
        if gap_days >= gap_threshold_days:
            last_weight = state.get('last_raw_weight')
            
            reset_event = {
                'timestamp': current_timestamp,
                'gap_days': gap_days,
                'last_timestamp': last_timestamp,
                'last_weight': last_weight,
                'reason': 'gap_exceeded'
            }
            
            new_state = {
                'kalman_params': None,
                'last_state': None,
                'last_covariance': None,
                'last_timestamp': None,
                'last_accepted_timestamp': None,
                'last_source': None,
                'last_raw_weight': None,
                'measurement_history': [],
                'reset_events': state.get('reset_events', []) + [reset_event],
                'measurements_since_reset': 0,
                'reset_timestamp': current_timestamp
            }
            
            return new_state, reset_event
    
    return state, None


def process_measurement(
    user_id: str,
    weight: float,
    timestamp: datetime,
    source: str,
    config: Dict[str, Any],
    unit: str = 'kg',
    db=None
) -> Dict[str, Any]:
    """
    Process a single weight measurement through the complete pipeline.
    
    Single function that:
    1. Cleans and validates data
    2. Manages Kalman state
    3. Applies filtering
    4. Returns comprehensive result
    
    Args:
        user_id: User identifier
        weight: Weight measurement value
        timestamp: Measurement timestamp
        source: Data source identifier
        config: Combined configuration dictionary
        unit: Unit of measurement
        db: Optional database instance
        
    Returns:
        Complete processing result with all metadata
    """
    if db is None:
        db = get_state_db()
    
    # Step 1: Data cleaning and preprocessing
    cleaned_weight, preprocess_metadata = DataQualityPreprocessor.preprocess(
        weight, source, timestamp, user_id, unit
    )
    
    # If preprocessing rejected the measurement
    if cleaned_weight is None:
        return {
            'accepted': False,
            'rejected': True,
            'timestamp': timestamp,
            'source': source,
            'raw_weight': weight,
            'reason': preprocess_metadata.get('rejected', 'Preprocessing failed'),
            'stage': 'preprocessing',
            'metadata': preprocess_metadata
        }
    
    # Step 2: Load or create user state
    state = db.get_state(user_id)
    if state is None:
        state = db.create_initial_state()
    
    # Add user height to config for validation
    user_height = DataQualityPreprocessor.get_user_height(user_id)
    
    # Step 3: Check for any type of reset using ResetManager
    kalman_config = config.get('kalman', {})
    
    # Check if reset is needed
    reset_type = ResetManager.should_trigger_reset(
        state, cleaned_weight, timestamp, source, config
    )
    
    reset_event = None
    reset_occurred = False
    
    if reset_type:
        # Perform the reset
        state, reset_event = ResetManager.perform_reset(
            state, reset_type, timestamp, cleaned_weight, source, config
        )
        reset_occurred = True
    
    # Step 4: Initialize Kalman if needed
    if not state.get('kalman_params'):
        # Check if this is a post-reset initialization
        # For initial measurements, treat current timestamp as "reset" to get adaptive params
        reset_timestamp = get_reset_timestamp(state) if reset_occurred else timestamp
        
        # Get adaptive Kalman config if within post-reset period
        adaptive_kalman_config = get_adaptive_kalman_params(
            reset_timestamp, timestamp, kalman_config, adaptive_days=7, state=state
        )
        
        # Get adaptive noise for this source
        adaptive_config = config.get('adaptive_noise', {})
        if adaptive_config.get('enabled', True):
            default_multiplier = adaptive_config.get('default_multiplier', 1.5)
            noise_multiplier = SOURCE_NOISE_MULTIPLIERS.get(source, default_multiplier)
        else:
            noise_multiplier = 1.0
        observation_covariance = adaptive_kalman_config.get('observation_covariance', 3.49) * noise_multiplier
        
        kalman_state = KalmanFilterManager.initialize_immediate(
            cleaned_weight, timestamp, adaptive_kalman_config, observation_covariance
        )
        # Merge Kalman state with existing state to preserve reset parameters
        state.update(kalman_state)
    
        state = KalmanFilterManager.update_state(
            state, cleaned_weight, timestamp, source, {}, observation_covariance
        )
        
        result = KalmanFilterManager.create_result(
            state, cleaned_weight, timestamp, source, True, observation_covariance
        )
        
        # Add metadata
        result['stage'] = 'initialization'
        result['preprocessing'] = preprocess_metadata
        result['noise_multiplier'] = noise_multiplier
        
        # Add reset event info if it occurred
        if reset_occurred:
            result['reset_event'] = {
                'type': reset_event.get('type', 'unknown'),
                'gap_days': reset_event.get('gap_days'),
                'reason': reset_event.get('reason', 'unknown')
            }
        
        # Save state
        state['last_source'] = source
        state['last_timestamp'] = timestamp  # Keep for backward compatibility
        state['last_accepted_timestamp'] = timestamp
        state['last_raw_weight'] = cleaned_weight  # Track for soft reset detection
        state["measurements_since_reset"] = state.get("measurements_since_reset", 0) + 1
        db.save_state(user_id, state)
        
        return result
    
    # Step 5: Quality scoring (replaces physiological validation)
    processing_config = config.get('processing', {})
    quality_config = config.get('quality_scoring', {})
    use_quality_scoring = quality_config.get('enabled', False)
    
    # Get previous weight and time diff
    previous_weight = None
    time_diff_hours = None
    
    # Try to get previous weight from Kalman state
    if state:
        current_weight, _ = KalmanFilterManager.get_current_state_values(state)
        if current_weight is not None:
            previous_weight = current_weight
        elif 'last_raw_weight' in state:
            previous_weight = state['last_raw_weight']
        
        # Get time diff
        if 'last_timestamp' in state:
            prev_time = state['last_timestamp']
            if isinstance(prev_time, str):
                prev_time = datetime.fromisoformat(prev_time)
            time_diff_hours = (timestamp - prev_time).total_seconds() / 3600
    
    # Get recent weights for statistical analysis
    recent_weights = []
    if state and 'measurement_history' in state:
        history = state['measurement_history']
        if isinstance(history, list):
            recent_weights = [h['weight'] for h in history[-20:] if 'weight' in h]
    
    if use_quality_scoring:
        # Use new quality scoring system
        # Check if we're in adaptive period (initial or post-reset)
        in_adaptive_period = False
        if state:
            measurements_since_reset = state.get("measurements_since_reset", 100)
            # Get adaptive parameters from reset state
            reset_params = state.get('reset_parameters', {})
            adaptation_measurements = reset_params.get('adaptation_measurements', 10)
            if measurements_since_reset < adaptation_measurements:
                in_adaptive_period = True
            else:
                # Also check time-based (7 days)
                reset_timestamp = get_reset_timestamp(state)
                if not reset_timestamp and not state.get("kalman_params"):
                    # Initial measurement
                    reset_timestamp = timestamp
                if reset_timestamp:
                    days_since = (timestamp - reset_timestamp).total_seconds() / 86400.0
                    adaptation_days = reset_params.get('adaptation_days', 7)
                    if days_since < adaptation_days:
                        in_adaptive_period = True
        
        # Adjust quality config if in adaptive period
        adaptive_quality_config = quality_config.copy()
        if in_adaptive_period:
            # Lower threshold during adaptation
            # Use threshold from reset parameters if available
            reset_params = state.get('reset_parameters', {})
            adaptive_threshold = reset_params.get('quality_acceptance_threshold', 0.4)
            adaptive_quality_config['threshold'] = adaptive_threshold
            # Adjust component weights to be more forgiving
            if "component_weights" in adaptive_quality_config:
                weights = adaptive_quality_config["component_weights"].copy()
                # Reduce plausibility weight (often fails during adaptation)
                weights["plausibility"] = 0.1  # vs normal 0.25
                weights["consistency"] = 0.15  # vs normal 0.25
                weights["safety"] = 0.45  # vs normal 0.35 (keep safety high)
                weights["reliability"] = 0.30  # vs normal 0.15
                adaptive_quality_config["component_weights"] = weights
        
        quality_score = PhysiologicalValidator.calculate_quality_score(
            weight=cleaned_weight,
            source=source,
            previous_weight=previous_weight,
            time_diff_hours=time_diff_hours,
            recent_weights=recent_weights,
            user_height_m=user_height,
            config=adaptive_quality_config
        )
        
        if not quality_score.accepted:
            return {
                'accepted': False,
                'timestamp': timestamp,
                'raw_weight': weight,
                'cleaned_weight': cleaned_weight,
                'source': source,
                'reason': quality_score.rejection_reason,
                'stage': 'quality_scoring',
                'quality_score': quality_score.overall,
                'quality_components': quality_score.components,
                'quality_details': quality_score.to_dict()
            }
        
        # Store quality score for later use
        quality_score_value = quality_score.overall
        quality_components = quality_score.components
    else:
        # Use legacy validation
        validation_result = PhysiologicalValidator.validate_comprehensive(
            cleaned_weight,
            previous_weight=previous_weight,
            time_diff_hours=time_diff_hours,
            source=source
        )
        
        is_valid = validation_result['valid']
        rejection_reason = validation_result.get('rejection_reason')
        
        if not is_valid:
            return {
                'accepted': False,
                'timestamp': timestamp,
                'raw_weight': weight,
                'cleaned_weight': cleaned_weight,
                'source': source,
                'reason': rejection_reason,
                'stage': 'physiological_validation'
            }
        
        quality_score_value = None
        quality_components = None
    
    # Step 6: Check deviation from Kalman prediction
    current_weight, current_trend = KalmanFilterManager.get_current_state_values(state)
    
    if current_weight is not None:
        time_delta_days = KalmanFilterManager.calculate_time_delta_days(
            timestamp, state.get('last_timestamp')
        )
        predicted_weight = current_weight + current_trend * time_delta_days
        deviation = abs(cleaned_weight - predicted_weight) / predicted_weight
        
        # Check if we're in adaptation phase for more lenient threshold
        extreme_threshold = processing_config.get('extreme_threshold', 0.20)
        if state:
            measurements_since_reset = state.get("measurements_since_reset", 100)
            reset_params = state.get('reset_parameters', {})
            adaptation_measurements = reset_params.get('adaptation_measurements', 10)
            if measurements_since_reset < adaptation_measurements:
                # During adaptation, use a much more lenient threshold
                # or skip the check entirely if quality_acceptance_threshold is 0
                quality_threshold = reset_params.get('quality_acceptance_threshold', 0.4)
                if quality_threshold == 0:
                    # Skip extreme deviation check during initial adaptation
                    extreme_threshold = float('inf')  # Effectively disable the check
                else:
                    # Use a more lenient threshold
                    extreme_threshold = 0.5  # 50% deviation allowed during adaptation
        
        if deviation > extreme_threshold:
            pseudo_normalized_innovation = (deviation / extreme_threshold) * 3.0
            confidence = KalmanFilterManager.calculate_confidence(pseudo_normalized_innovation)
            
            return {
                'accepted': False,
                'timestamp': timestamp,
                'raw_weight': weight,
                'cleaned_weight': cleaned_weight,
                'filtered_weight': float(predicted_weight),
                'trend': float(current_trend),
                'reason': f"Extreme deviation: {deviation:.1%}",
                'confidence': confidence,
                'source': source,
                'stage': 'kalman_deviation'
            }
    
    # Step 7: Update Kalman filter
    # Check if we should use adaptive parameters (within 7 days of reset)
    reset_timestamp = get_reset_timestamp(state)
    adaptive_kalman_config = get_adaptive_kalman_params(
        reset_timestamp, timestamp, kalman_config, adaptive_days=7, state=state
    )
    
    # Update state's kalman_params with adaptive values
    if reset_timestamp and (timestamp - reset_timestamp).total_seconds() / 86400.0 < 7:
        state['kalman_params']['transition_covariance'] = [
            [adaptive_kalman_config['transition_covariance_weight'], 0],
            [0, adaptive_kalman_config['transition_covariance_trend']]
        ]
    
    adaptive_config = config.get('adaptive_noise', {})
    if adaptive_config.get('enabled', True):
        default_multiplier = adaptive_config.get('default_multiplier', 1.5)
        noise_multiplier = SOURCE_NOISE_MULTIPLIERS.get(source, default_multiplier)
    else:
        noise_multiplier = 1.0
    observation_covariance = adaptive_kalman_config.get('observation_covariance', 3.49) * noise_multiplier
    
    state = KalmanFilterManager.update_state(
        state, cleaned_weight, timestamp, source, {}, observation_covariance
    )
    
    result = KalmanFilterManager.create_result(
        state, cleaned_weight, timestamp, source, True, observation_covariance
    )
    
    # Step 8: Add comprehensive metadata
    result['preprocessing'] = preprocess_metadata
    result['noise_multiplier'] = noise_multiplier
    result['stage'] = 'accepted'
    
    # Add quality score if available
    if quality_score_value is not None:
        result['quality_score'] = quality_score_value
        result['quality_components'] = quality_components
    

    
    # Add reset event info if it occurred
    if reset_occurred:
        result['reset_event'] = {
            'type': reset_event.get('type', 'unknown'),
            'gap_days': reset_event.get('gap_days'),
            'reason': reset_event.get('reason', 'unknown')
        }
    
    # Calculate BMI details
    implied_bmi = cleaned_weight / (user_height ** 2)
    result['bmi_details'] = {
        'user_height_m': user_height,
        'implied_bmi': round(implied_bmi, 1),
        'original_weight': weight,
        'original_unit': unit,
        'cleaned_weight': cleaned_weight
    }
    
    # Update measurement history for quality scoring
    if use_quality_scoring:
        if 'measurement_history' not in state:
            state['measurement_history'] = []
        
        state['measurement_history'].append({
            'weight': cleaned_weight,
            'timestamp': timestamp.isoformat(),
            'quality_score': quality_score_value,
            'source': source
        })
        
        # Keep only last 30 measurements
        state['measurement_history'] = state['measurement_history'][-30:]
    
    # Save updated state
    # Increment measurements counter for adaptation tracking
    state['measurements_since_reset'] = state.get('measurements_since_reset', 0) + 1
    state['last_source'] = source
    state['last_timestamp'] = timestamp  # Keep for backward compatibility
    state['last_accepted_timestamp'] = timestamp
    state['last_raw_weight'] = cleaned_weight  # Track for soft reset detection
    db.save_state(user_id, state)
    
    return result


# Backward compatibility wrapper
def process_weight_enhanced(
    user_id: str,
    weight: float,
    timestamp: datetime,
    source: str,
    processing_config: Dict,
    kalman_config: Dict,
    unit: str = 'kg'
) -> Optional[Dict]:
    """Backward compatibility wrapper for process_measurement."""
    config = {
        'processing': processing_config,
        'kalman': kalman_config
    }
    
    # Handle nested config for adaptive noise
    if 'config' in processing_config:
        config.update(processing_config['config'])
    
    return process_measurement(user_id, weight, timestamp, source, config, unit)


class WeightProcessor:
    """Stateless weight processor for backward compatibility."""
    
    @staticmethod
    def process_weight(
        user_id: str,
        weight: float,
        timestamp: datetime,
        source: str,
        processing_config: Dict,
        kalman_config: Dict,
        db=None,
        observation_covariance: Optional[float] = None
    ) -> Optional[Dict]:
        """Backward compatibility wrapper."""
        config = {
            'processing': processing_config,
            'kalman': kalman_config
        }
        
        if observation_covariance is not None:
            config['kalman']['observation_covariance'] = observation_covariance
        
        return process_measurement(user_id, weight, timestamp, source, config, 'kg', db)
    
    @staticmethod
    def get_user_state(user_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state for a user."""
        db = get_state_db()
        return db.get_state(user_id)
    
    @staticmethod
    def reset_user(user_id: str) -> bool:
        """Reset a user's state."""
        db = get_state_db()
        return db.delete_state(user_id)
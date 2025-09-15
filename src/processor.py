"""
Simplified weight processor with flattened pipeline.
Single processing function with clear flow.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple
import numpy as np

from .database import get_state_db
from .kalman import KalmanFilterManager
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


def handle_gap_detection(state: Dict[str, Any], current_timestamp: datetime, config: Dict[str, Any]) -> Dict[str, Any]:
    """Detect gap and initialize buffer if needed."""
    gap_config = config.get('kalman', {}).get('gap_handling', {})
    if not gap_config.get('enabled', True):
        return state
    
    gap_threshold = gap_config.get('gap_threshold_days', 10)
    
    if state.get('last_timestamp'):
        last_timestamp = state['last_timestamp']
        if isinstance(last_timestamp, str):
            last_timestamp = datetime.fromisoformat(last_timestamp)
        
        gap_days = (current_timestamp - last_timestamp).total_seconds() / 86400.0
        
        if gap_days > gap_threshold:
            state['gap_buffer'] = {
                'active': True,
                'gap_days': gap_days,
                'gap_detected_at': current_timestamp.isoformat(),
                'measurements': [],
                'target_size': gap_config.get('warmup_size', 3),
                'timeout_days': gap_config.get('max_warmup_days', 7)
            }
            
            if state.get('last_state') is not None:
                _, trend = KalmanFilterManager.get_current_state_values(state)
                if trend is not None:
                    state['gap_buffer']['pre_gap_trend'] = float(trend)
    
    return state


def update_gap_buffer(state: Dict[str, Any], weight: float, timestamp: datetime, source: str) -> Tuple[Dict[str, Any], bool]:
    """Add measurement to buffer and check completion."""
    buffer = state.get('gap_buffer', {})
    
    if not buffer.get('active'):
        return state, False
    
    buffer['measurements'].append({
        'weight': weight,
        'timestamp': timestamp.isoformat(),
        'source': source
    })
    
    is_complete = False
    if len(buffer['measurements']) >= buffer['target_size']:
        is_complete = True
    elif len(buffer['measurements']) >= 2:
        first_timestamp = datetime.fromisoformat(buffer['measurements'][0]['timestamp'])
        time_span_days = (timestamp - first_timestamp).total_seconds() / 86400
        if time_span_days >= buffer['timeout_days']:
            is_complete = True
    
    state['gap_buffer'] = buffer
    return state, is_complete


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
    
    # Step 3: Check for gaps and handle buffer
    kalman_config = config.get('kalman', {})
    gap_handling_config = kalman_config.get('gap_handling', {})
    gap_handling_enabled = gap_handling_config.get('enabled', True)
    
    if state.get('last_timestamp'):
        last_timestamp = state['last_timestamp']
        if isinstance(last_timestamp, str):
            last_timestamp = datetime.fromisoformat(last_timestamp)
        
        gap_days = (timestamp - last_timestamp).total_seconds() / 86400.0
        
        # Check if gap handling is enabled and gap exceeds threshold
        gap_threshold = gap_handling_config.get('gap_threshold_days', 10)
        
        if gap_handling_enabled and gap_days > gap_threshold:
            # Initialize gap buffer for adaptive handling
            state = handle_gap_detection(state, timestamp, config)
    
    # Step 3.5: Handle gap buffer if active
    gap_buffer = state.get('gap_buffer')
    if gap_buffer and gap_buffer.get('active'):
        state, buffer_complete = update_gap_buffer(state, cleaned_weight, timestamp, source)
        
        if buffer_complete:
            # Initialize Kalman from buffer
            buffer_state = KalmanFilterManager.initialize_from_buffer(state['gap_buffer'], config)
            state.update(buffer_state)
            state['gap_buffer']['active'] = False
            state['last_timestamp'] = timestamp
            state['last_source'] = source
            state['last_raw_weight'] = cleaned_weight
            
            # Process the last measurement through Kalman
            adaptive_config = config.get('adaptive_noise', {})
            if adaptive_config.get('enabled', True):
                default_multiplier = adaptive_config.get('default_multiplier', 1.5)
                noise_multiplier = SOURCE_NOISE_MULTIPLIERS.get(source, default_multiplier)
            else:
                noise_multiplier = 1.0
            observation_covariance = kalman_config.get('observation_covariance', 3.49) * noise_multiplier
            
            state = KalmanFilterManager.update_state(
                state, cleaned_weight, timestamp, source, {}, observation_covariance
            )
            
            result = KalmanFilterManager.create_result(
                state, cleaned_weight, timestamp, source, True, observation_covariance
            )
            
            result['stage'] = 'gap_buffer_complete'
            result['gap_buffer_size'] = len(state['gap_buffer']['measurements'])
            result['gap_days'] = state['gap_buffer']['gap_days']
            result['preprocessing'] = preprocess_metadata
            result['noise_multiplier'] = noise_multiplier
            
            db.save_state(user_id, state)
            return result
        else:
            # Buffer not complete, return preliminary result
            result = {
                'accepted': True,
                'timestamp': timestamp,
                'raw_weight': weight,
                'cleaned_weight': cleaned_weight,
                'source': source,
                'stage': 'gap_buffer_collecting',
                'gap_buffer_size': len(state['gap_buffer']['measurements']),
                'gap_buffer_target': state['gap_buffer']['target_size'],
                'preprocessing': preprocess_metadata,
                'filtered_weight': cleaned_weight,
                'trend': 0.0,
                'confidence': 0.5
            }
            
            state['last_timestamp'] = timestamp
            state['last_source'] = source
            state['last_raw_weight'] = cleaned_weight
            db.save_state(user_id, state)
            return result
    
    # Step 4: Initialize Kalman if needed
    if not state.get('kalman_params'):
        # Get adaptive noise for this source
        adaptive_config = config.get('adaptive_noise', {})
        if adaptive_config.get('enabled', True):
            default_multiplier = adaptive_config.get('default_multiplier', 1.5)
            noise_multiplier = SOURCE_NOISE_MULTIPLIERS.get(source, default_multiplier)
        else:
            noise_multiplier = 1.0
        observation_covariance = kalman_config.get('observation_covariance', 3.49) * noise_multiplier
        
        state = KalmanFilterManager.initialize_immediate(
            cleaned_weight, timestamp, kalman_config, observation_covariance
        )
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
        
        # Save state
        state['last_source'] = source
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
        quality_score = PhysiologicalValidator.calculate_quality_score(
            weight=cleaned_weight,
            source=source,
            previous_weight=previous_weight,
            time_diff_hours=time_diff_hours,
            recent_weights=recent_weights,
            user_height_m=user_height,
            config=quality_config
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
        
        extreme_threshold = processing_config.get('extreme_threshold', 0.20)
        
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
    
    # Step 7: Update Kalman filter with adaptation decay if active
    if state.get('gap_adaptation', {}).get('active'):
        state = KalmanFilterManager.apply_adaptation_decay(state, config)
    
    adaptive_config = config.get('adaptive_noise', {})
    if adaptive_config.get('enabled', True):
        default_multiplier = adaptive_config.get('default_multiplier', 1.5)
        noise_multiplier = SOURCE_NOISE_MULTIPLIERS.get(source, default_multiplier)
    else:
        noise_multiplier = 1.0
    observation_covariance = kalman_config.get('observation_covariance', 3.49) * noise_multiplier
    
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
    

    
    # Add gap adaptation info if active
    if state.get('gap_adaptation', {}).get('active'):
        result['gap_adaptation'] = {
            'active': True,
            'measurements_since_gap': state['gap_adaptation']['measurements_since_gap'],
            'gap_factor': state['gap_adaptation']['gap_factor']
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
    state['last_source'] = source
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
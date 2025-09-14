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
    
    # Step 3: Check for gaps and potential reset
    kalman_config = config.get('kalman', {})
    reset_gap_days = kalman_config.get('reset_gap_days', 30)
    
    if state.get('last_timestamp'):
        last_timestamp = state['last_timestamp']
        if isinstance(last_timestamp, str):
            last_timestamp = datetime.fromisoformat(last_timestamp)
        
        gap_days = (timestamp - last_timestamp).total_seconds() / 86400.0
        
        # Adjust reset gap for questionnaire sources
        if state.get('last_source') in QUESTIONNAIRE_SOURCES:
            reset_gap_days = kalman_config.get('questionnaire_reset_days', 10)
        
        # Reset if gap is too large
        if gap_days > reset_gap_days:
            # Validate with BMI before reset
            # Use BMI consistency validation
            bmi_result = BMIValidator.validate_weight_bmi_consistency(
                cleaned_weight, user_height, source
            )
            is_valid = bmi_result['valid']
            rejection_reason = bmi_result.get('rejection_reason')
            
            if not is_valid:
                return {
                    'accepted': False,
                    'timestamp': timestamp,
                    'raw_weight': weight,
                    'cleaned_weight': cleaned_weight,
                    'source': source,
                    'reason': f"Post-gap BMI validation failed: {rejection_reason}",
                    'gap_days': gap_days,
                    'stage': 'gap_validation'
                }
            
            # Reset state for fresh start
            state = db.create_initial_state()
            was_reset = True
            reset_reason = f"Gap of {gap_days:.1f} days"
    else:
        was_reset = False
        reset_reason = None
    
    # Step 4: Initialize Kalman if needed
    if not state.get('kalman_params'):
        # Get adaptive noise for this source
        noise_multiplier = SOURCE_NOISE_MULTIPLIERS.get(source, 1.5)
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
        if 'was_reset' in locals() and was_reset:
            result['was_reset'] = True
            result['reset_reason'] = reset_reason
            result['gap_days'] = gap_days
        
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
    if state and 'weight' in state:
        previous_weight = state['weight']
        if 'timestamp' in state:
            prev_time = datetime.fromisoformat(state['timestamp'])
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
    
    # Step 7: Update Kalman filter
    noise_multiplier = SOURCE_NOISE_MULTIPLIERS.get(source, 1.5)
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
    
    # Add reset info if applicable
    if 'was_reset' in locals() and was_reset:
        result['was_reset'] = True
        result['reset_reason'] = reset_reason
        result['gap_days'] = gap_days
    
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
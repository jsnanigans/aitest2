"""
Simplified weight processor with flattened pipeline.
Single processing function with clear flow.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple
import numpy as np
import copy
import logging

from ..database.database import get_state_db
from ..feature_manager import FeatureManager
from .kalman import (
    KalmanFilterManager,
    ResetManager,
    ResetType,
    get_adaptive_kalman_params,
    get_reset_timestamp
)
from .validation import PhysiologicalValidator, BMIValidator, ThresholdCalculator, DataQualityPreprocessor
from .persistence_validator import PersistenceValidator
from .reset_transaction import ResetTransaction, ResetOperation
from .state_validator import StateValidator
from .circuit_breaker import CircuitBreaker, CircuitOpenError

logger = logging.getLogger(__name__)
from ..constants import (
    QUESTIONNAIRE_SOURCES,
    KALMAN_DEFAULTS,
    PHYSIOLOGICAL_LIMITS,
    BMI_LIMITS,
    get_noise_multiplier,
    categorize_rejection_enhanced,
    get_rejection_severity
)
try:
    from .quality_scorer import QualityScorer, MeasurementHistory
except ImportError:
    from quality_scorer import QualityScorer, MeasurementHistory

# Use constants from constants.py
MIN_WEIGHT = PHYSIOLOGICAL_LIMITS['ABSOLUTE_MIN_WEIGHT']
MAX_WEIGHT = PHYSIOLOGICAL_LIMITS['ABSOLUTE_MAX_WEIGHT']
MIN_VALID_BMI = 10.0  # Below critical low BMI
MAX_VALID_BMI = 90.0  # Well above morbidly obese


def check_and_reset_for_gap(state: Dict[str, Any], current_timestamp: datetime, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Check for 30+ day gap and reset if needed."""
    # This function is now deprecated - reset logic is handled by ResetManager
    # Kept for backward compatibility but should not be called
    reset_config = config.get('kalman', {}).get('reset', {})
    feature_manager = config.get('feature_manager')
    if feature_manager and not feature_manager.is_enabled('reset_hard'):
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
    # Get feature manager
    feature_manager = config.get('feature_manager')
    if not feature_manager:
        feature_manager = FeatureManager(config)

    # Only load state if persistence is enabled
    if feature_manager.is_enabled('state_persistence'):
        state = db.get_state(user_id)
        if state is None:
            state = db.create_initial_state()
    else:
        # Use minimal state without persistence
        state = db.create_initial_state()

    # Add user height to config for validation
    user_height = DataQualityPreprocessor.get_user_height(user_id)
    
    # Step 3: Check for any type of reset using ResetManager
    kalman_config = config.get('kalman', {})
    
    # Check if reset is needed (only if reset features are enabled)
    reset_type = None
    if (feature_manager.is_enabled('reset_initial') or
        feature_manager.is_enabled('reset_hard') or
        feature_manager.is_enabled('reset_soft')):
        reset_type = ResetManager.should_trigger_reset(
            state, cleaned_weight, timestamp, source, config
        )
    
    reset_event = None
    reset_occurred = False

    if reset_type:
        # Perform the reset with transaction safety
        state, reset_event, reset_occurred = _handle_reset_with_transaction(
            user_id, state, reset_type, timestamp, cleaned_weight, source, config
        )
    
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
        if feature_manager.is_enabled('adaptive_noise'):
            default_multiplier = adaptive_config.get('default_multiplier', 1.5)
            noise_multiplier = get_noise_multiplier(source)
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
        
        # Save state - This is after outlier rejection (early rejection path)
        state['last_source'] = source
        state['last_timestamp'] = timestamp  # Keep for backward compatibility
        state['last_accepted_timestamp'] = timestamp
        state['last_raw_weight'] = cleaned_weight  # Track for soft reset detection
        state["measurements_since_reset"] = state.get("measurements_since_reset", 0) + 1

        # Validate state before persistence
        if feature_manager.is_enabled('state_persistence'):
            is_valid, error_msg = PersistenceValidator.validate_state(
                state, user_id, reason="outlier_rejection_accept"
            )
            if is_valid:
                # Get previous state for change detection
                previous_state = db.get_state(user_id)
                should_persist, audit_msg = PersistenceValidator.should_persist(
                    state, previous_state, user_id, reason="outlier_rejection_accept"
                )

                if should_persist:
                    db.save_state(user_id, state)
                    PersistenceValidator.create_audit_log(
                        user_id, "persist", state, True,
                        reason="outlier_rejection_accept", error=None
                    )
                else:
                    # Log why we're not persisting
                    PersistenceValidator.create_audit_log(
                        user_id, "skip", state, True,
                        reason=audit_msg, error=None
                    )
            else:
                # Log validation failure
                PersistenceValidator.create_audit_log(
                    user_id, "validate_failed", state, False,
                    reason="outlier_rejection_accept", error=error_msg
                )
        
        return result
    
    # Step 5: Quality scoring (replaces physiological validation)
    processing_config = config.get('processing', {})
    quality_config = config.get('quality_scoring', {})
    use_quality_scoring = feature_manager.is_enabled('quality_scoring')
    
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
            source=source,
            feature_manager=feature_manager
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

    # Only check Kalman deviation if feature is enabled
    kalman_deviation_enabled = feature_manager.is_enabled('kalman_deviation_check') and feature_manager.is_enabled('kalman_filtering')

    if current_weight is not None and kalman_deviation_enabled:
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
                    # Use a moderately lenient threshold during adaptation
                    extreme_threshold = 0.25  # 25% deviation allowed during adaptation (reduced from 50%)
        
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
    # Only update Kalman if feature is enabled
    if not feature_manager.is_enabled('kalman_filtering'):
        # Skip Kalman filtering - just pass through the weight
        result = {
            'accepted': True,
            'timestamp': timestamp,
            'raw_weight': weight,
            'cleaned_weight': cleaned_weight,
            'filtered_weight': cleaned_weight,  # No filtering
            'trend': 0.0,
            'confidence': 1.0,
            'source': source,
            'stage': 'no_filtering'
        }

        # Save minimal state if persistence enabled (early return path - no Kalman filtering)
        if feature_manager.is_enabled('state_persistence'):
            state['last_source'] = source
            state['last_timestamp'] = timestamp
            state['last_accepted_timestamp'] = timestamp
            state['last_raw_weight'] = cleaned_weight

            # Validate state before persistence
            is_valid, error_msg = PersistenceValidator.validate_state(
                state, user_id, reason="no_kalman_filtering"
            )
            if is_valid:
                # Get previous state for change detection
                previous_state = db.get_state(user_id)
                should_persist, audit_msg = PersistenceValidator.should_persist(
                    state, previous_state, user_id, reason="no_kalman_filtering"
                )

                if should_persist:
                    db.save_state(user_id, state)
                    PersistenceValidator.create_audit_log(
                        user_id, "persist", state, True,
                        reason="no_kalman_filtering", error=None
                    )
                else:
                    # Log why we're not persisting
                    PersistenceValidator.create_audit_log(
                        user_id, "skip", state, True,
                        reason=audit_msg, error=None
                    )
            else:
                # Log validation failure
                PersistenceValidator.create_audit_log(
                    user_id, "validate_failed", state, False,
                    reason="no_kalman_filtering", error=error_msg
                )

        return result

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
    if feature_manager.is_enabled('adaptive_noise'):
        default_multiplier = adaptive_config.get('default_multiplier', 1.5)
        noise_multiplier = get_noise_multiplier(source)
    else:
        noise_multiplier = 1.0
    observation_covariance = adaptive_kalman_config.get('observation_covariance', 3.49) * noise_multiplier
    
    # Apply trend limiting before update
    current_weight, current_trend = KalmanFilterManager.get_current_state_values(state)
    if current_trend is not None:
        # Limit trend to ±5kg/week (±0.714kg/day)
        max_daily_trend = 0.714  # 5kg/week
        if abs(current_trend) > max_daily_trend:
            # Clamp the trend in the state before update
            limited_trend = max_daily_trend if current_trend > 0 else -max_daily_trend
            if state.get('last_state') is not None:
                last_state = state['last_state']
                if len(last_state.shape) > 1:
                    last_state[-1][1] = limited_trend
                else:
                    last_state[1] = limited_trend

    state = KalmanFilterManager.update_state(
        state, cleaned_weight, timestamp, source, {}, observation_covariance
    )

    # Apply trend limiting after update
    current_weight, current_trend = KalmanFilterManager.get_current_state_values(state)
    if current_trend is not None and abs(current_trend) > 0.714:
        # Clamp the trend after update
        limited_trend = 0.714 if current_trend > 0 else -0.714
        if state.get('last_state') is not None:
            last_state = state['last_state']
            if len(last_state.shape) > 1:
                last_state[-1][1] = limited_trend
            else:
                last_state[1] = limited_trend
    
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
    
    # Save updated state - Main successful processing path
    # Increment measurements counter for adaptation tracking
    state['measurements_since_reset'] = state.get('measurements_since_reset', 0) + 1
    state['last_source'] = source
    state['last_timestamp'] = timestamp  # Keep for backward compatibility
    state['last_accepted_timestamp'] = timestamp
    state['last_raw_weight'] = cleaned_weight  # Track for soft reset detection

    # Validate state before persistence
    if feature_manager.is_enabled('state_persistence'):
        is_valid, error_msg = PersistenceValidator.validate_state(
            state, user_id, reason="successful_processing"
        )
        if is_valid:
            # Get previous state for change detection
            previous_state = db.get_state(user_id)
            should_persist, audit_msg = PersistenceValidator.should_persist(
                state, previous_state, user_id, reason="successful_processing"
            )

            if should_persist:
                db.save_state(user_id, state)
                PersistenceValidator.create_audit_log(
                    user_id, "persist", state, True,
                    reason="successful_processing", error=None
                )
            else:
                # Log why we're not persisting
                PersistenceValidator.create_audit_log(
                    user_id, "skip", state, True,
                    reason=audit_msg, error=None
                )
        else:
            # Log validation failure
            PersistenceValidator.create_audit_log(
                user_id, "validate_failed", state, False,
                reason="successful_processing", error=error_msg
            )
    
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


# Circuit breaker for reset operations (module level)
_reset_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    timeout=60,
    success_threshold=2,
    name="reset_operations"
)


def _handle_reset_with_transaction(
    user_id: str,
    state: Dict[str, Any],
    reset_type: ResetType,
    timestamp: datetime,
    weight: float,
    source: str,
    config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[Dict], bool]:
    """
    Handle reset operations with transaction safety and circuit breaker.

    Args:
        user_id: User identifier
        state: Current state
        reset_type: Type of reset to perform
        timestamp: Measurement timestamp
        weight: Weight value
        source: Data source
        config: Configuration

    Returns:
        Tuple of (new_state, reset_event, reset_occurred)
    """
    try:
        # Try through circuit breaker first
        return _reset_circuit_breaker.call(
            _perform_transactional_reset,
            user_id, state, reset_type, timestamp, weight, source, config
        )
    except CircuitOpenError as e:
        logger.error(f"Reset circuit open for user {user_id}: {e}")
        # Return original state without reset
        return state, None, False
    except Exception as e:
        logger.error(f"Reset failed for user {user_id}: {e}")
        # Return original state without reset
        return state, None, False


def _perform_transactional_reset(
    user_id: str,
    state: Dict[str, Any],
    reset_type: ResetType,
    timestamp: datetime,
    weight: float,
    source: str,
    config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[Dict], bool]:
    """
    Perform reset with transaction management.

    Returns:
        Tuple of (new_state, reset_event, reset_occurred)
    """
    with ResetTransaction(user_id) as txn:
        # Save original state for potential rollback
        txn.save_original_state(ResetOperation.STATE_UPDATE, state)

        try:
            # Step 1: Perform the actual reset
            # Handle both ResetType enum and string
            reset_type_value = reset_type.value if hasattr(reset_type, 'value') else reset_type
            logger.info(f"Applying {reset_type_value} reset for user {user_id}")
            new_state, reset_event = ResetManager.perform_reset(
                state, reset_type, timestamp, weight, source, config
            )

            # Save checkpoint and validate
            txn.save_checkpoint(ResetOperation.STATE_UPDATE, new_state)
            if not txn.validate_checkpoint(ResetOperation.STATE_UPDATE):
                raise ValueError(f"State validation failed after {reset_type_value} reset")

            txn.mark_completed(ResetOperation.STATE_UPDATE)

            # Step 2: Validate Kalman reset (kalman_params should be None)
            kalman_state = {
                'kalman_params': new_state.get('kalman_params'),
                'reset_parameters': new_state.get('reset_parameters'),
                'measurements_since_reset': new_state.get('measurements_since_reset', 0),
                'reset_type': new_state.get('reset_type'),
                'reset_timestamp': new_state.get('reset_timestamp')
            }

            txn.save_checkpoint(ResetOperation.KALMAN_RESET, kalman_state)
            if not txn.validate_checkpoint(ResetOperation.KALMAN_RESET):
                raise ValueError("Kalman state validation failed after reset")

            txn.mark_completed(ResetOperation.KALMAN_RESET)

            # All operations succeeded
            logger.info(f"Reset transaction completed successfully for user {user_id}")
            return new_state, reset_event, True

        except Exception as e:
            import traceback
            logger.error(f"Reset transaction failed for user {user_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Transaction will automatically rollback
            # Return original state
            original_state = txn.get_original_state(ResetOperation.STATE_UPDATE)
            if original_state:
                return original_state, None, False
            return state, None, False
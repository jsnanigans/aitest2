"""
Enhanced weight processor with data quality improvements.
Based on analysis of 709,246 measurements revealing source-specific patterns.
"""

from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List, Any
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from pykalman import KalmanFilter

try:
    from .processor_database import ProcessorStateDB, get_state_db
    from .threshold_calculator import ThresholdCalculator
except ImportError:
    from processor_database import ProcessorStateDB, get_state_db
    from threshold_calculator import ThresholdCalculator


class WeightProcessor:
    """
    Stateless weight processor - ALL values processed immediately.
    No buffering, no waiting for initialization - processes everything.
    """

    @staticmethod
    def process_weight(
        user_id: str,
        weight: float,
        timestamp: datetime,
        source: str,
        processing_config: Dict,
        kalman_config: Dict,
        db=None,
    ) -> Optional[Dict]:
        """
        Process a single weight measurement for a user.
        
        ALWAYS returns a result - no buffering, no waiting.
        After long gaps (>30 days), resets state for fresh start.
        
        Args:
            user_id: User identifier
            weight: Weight measurement in kg
            timestamp: Measurement timestamp
            source: Data source identifier
            processing_config: Processing configuration dict
            kalman_config: Kalman filter configuration dict
            db: Optional database instance (creates new if None)
            
        Returns:
            Result dictionary - NEVER None, all measurements processed
        """
        if db is None:
            db = get_state_db()
        
        state = db.get_state(user_id)
        if state is None:
            state = db.create_initial_state()
        
        result, updated_state = WeightProcessor._process_weight_internal(
            weight, timestamp, source, state, processing_config, kalman_config
        )
        
        if updated_state:
            updated_state['last_source'] = source
            db.save_state(user_id, updated_state)
        
        return result

    @staticmethod
    def _process_weight_internal(
        weight: float,
        timestamp: datetime,
        source: str,
        state: Dict[str, Any],
        processing_config: dict,
        kalman_config: dict
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Internal processing logic - pure functional.
        
        Returns:
            Tuple of (result, updated_state)
            Result is never None - all measurements are processed
        """
        new_state = state.copy()
        
        if not state.get('kalman_params'):
            new_state = WeightProcessor._initialize_kalman_immediate(
                weight, timestamp, kalman_config
            )
            
            new_state = WeightProcessor._update_kalman_state(
                new_state, weight, timestamp, source, processing_config
            )
            
            result = WeightProcessor._create_result(
                new_state, weight, timestamp, source, True
            )
            return result, new_state
        
        time_delta_days = 1.0
        should_reset = False
        if new_state.get('last_timestamp'):
            last_timestamp = new_state['last_timestamp']
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp)
            delta = (timestamp - last_timestamp).total_seconds() / 86400.0
            time_delta_days = max(0.1, min(30.0, delta))
            
            reset_gap_days = kalman_config.get("reset_gap_days", 30)
            
            questionnaire_sources = {
                'internal-questionnaire',
                'initial-questionnaire',
                'care-team-upload',
                'questionnaire'
            }
            if state.get('last_source') in questionnaire_sources:
                reset_gap_days = kalman_config.get("questionnaire_reset_days", 10)
            
            if delta > reset_gap_days:
                should_reset = True
                new_state = WeightProcessor._initialize_kalman_immediate(
                    weight, timestamp, kalman_config
                )
                new_state = WeightProcessor._update_kalman_state(
                    new_state, weight, timestamp, source, processing_config
                )
                result = WeightProcessor._create_result(
                    new_state, weight, timestamp, source, True
                )
                result['was_reset'] = True
                result['gap_days'] = delta
                return result, new_state
        
        if not should_reset:
            is_valid, rejection_reason = WeightProcessor._validate_weight(
                weight,
                processing_config,
                state,
                timestamp
            )
            if not is_valid:
                return {
                    "timestamp": timestamp,
                    "raw_weight": weight,
                    "accepted": False,
                    "reason": rejection_reason or "Basic validation failed",
                    "source": source,
                }, None
        
        if new_state.get('last_state') is not None:
            last_state = new_state['last_state']
            if len(last_state.shape) > 1:
                current_state = last_state[-1]
            else:
                current_state = last_state
            
            predicted_weight = current_state[0] + current_state[1] * time_delta_days
            deviation = abs(weight - predicted_weight) / predicted_weight
            
            extreme_threshold = processing_config["extreme_threshold"]
            
            if deviation > extreme_threshold:
                pseudo_normalized_innovation = (deviation / extreme_threshold) * 3.0
                confidence = WeightProcessor._calculate_confidence(
                    pseudo_normalized_innovation
                )
                
                return {
                    "timestamp": timestamp,
                    "raw_weight": weight,
                    "filtered_weight": float(predicted_weight),
                    "trend": float(current_state[1]),
                    "accepted": False,
                    "reason": f"Extreme deviation: {deviation:.1%}",
                    "confidence": confidence,
                    "source": source,
                }, None
        
        new_state = WeightProcessor._update_kalman_state(
            new_state, weight, timestamp, source, processing_config
        )
        
        result = WeightProcessor._create_result(
            new_state, weight, timestamp, source, True
        )
        
        return result, new_state

    @staticmethod
    def _calculate_time_delta_hours(
        current_timestamp: datetime,
        last_timestamp: Optional[datetime]
    ) -> float:
        """Calculate hours between measurements."""
        if last_timestamp is None:
            return float('inf')
        
        if isinstance(last_timestamp, str):
            last_timestamp = datetime.fromisoformat(last_timestamp)
        
        delta = (current_timestamp - last_timestamp).total_seconds() / 3600
        return max(0.0, delta)
    
    @staticmethod
    def _get_physiological_limit(
        time_delta_hours: float,
        last_weight: float,
        config: dict
    ) -> tuple[float, str]:
        """
        Get maximum allowed weight change based on time elapsed.
        Based on framework document recommendations (Section 3.1).
        """
        phys_config = config.get('physiological', {})
        
        if not phys_config.get('enable_physiological_limits', True):
            legacy_limit = config.get('max_daily_change', 0.05) * last_weight
            return legacy_limit, "legacy percentage limit"
        
        if time_delta_hours < 1:
            percent_limit = phys_config.get('max_change_1h_percent', 0.02)
            absolute_limit = phys_config.get('max_change_1h_absolute', 3.0)
            reason = "hydration/bathroom"
        elif time_delta_hours < 6:
            percent_limit = phys_config.get('max_change_6h_percent', 0.025)
            absolute_limit = phys_config.get('max_change_6h_absolute', 4.0)
            reason = "meals+hydration"
        elif time_delta_hours <= 24:
            percent_limit = phys_config.get('max_change_24h_percent', 0.035)
            absolute_limit = phys_config.get('max_change_24h_absolute', 5.0)
            reason = "daily fluctuation"
        else:
            daily_rate = phys_config.get('max_sustained_daily', 1.5)
            days = time_delta_hours / 24
            absolute_limit = days * daily_rate
            percent_limit = absolute_limit / last_weight
            reason = f"sustained ({daily_rate}kg/day)"
        
        percentage_based = last_weight * percent_limit
        limit = min(percentage_based, absolute_limit)
        
        if limit == absolute_limit and time_delta_hours <= 24:
            reason += f" (capped at {absolute_limit}kg)"
        
        return limit, reason
    
    @staticmethod
    def _validate_weight(
        weight: float,
        config: dict,
        state: Optional[Dict[str, Any]],
        timestamp: Optional[datetime] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate weight with physiological limits.
        Returns (is_valid, rejection_reason).
        """
        min_weight = config.get('min_weight', 30)
        max_weight = config.get('max_weight', 400)
        
        if weight < min_weight or weight > max_weight:
            return False, f"Weight {weight}kg outside bounds [{min_weight}, {max_weight}]"
        
        if state is None:
            return True, None
        
        if 'last_raw_weight' in state:
            last_weight = state['last_raw_weight']
        elif state.get('last_state') is not None:
            last_state_val = state.get('last_state')
            if isinstance(last_state_val, np.ndarray):
                last_weight = last_state_val[-1][0] if len(last_state_val.shape) > 1 else last_state_val[0]
            else:
                last_weight = last_state_val[0] if isinstance(last_state_val, (list, tuple)) else last_state_val
        else:
            return True, None
        
        if timestamp and state.get('last_timestamp'):
            time_delta_hours = WeightProcessor._calculate_time_delta_hours(
                timestamp, state.get('last_timestamp')
            )
        else:
            time_delta_hours = 24
        
        change = abs(weight - last_weight)
        max_change, reason = WeightProcessor._get_physiological_limit(
            time_delta_hours, last_weight, config
        )
        
        phys_config = config.get('physiological', {})
        if time_delta_hours > 24:
            tolerance = phys_config.get('sustained_tolerance', 0.25)
        else:
            tolerance = phys_config.get('limit_tolerance', 0.10)
        effective_limit = max_change * (1 + tolerance)
        
        if change > effective_limit:
            return False, (f"Change of {change:.1f}kg in {time_delta_hours:.1f}h "
                          f"exceeds {reason} limit of {max_change:.1f}kg")
        
        phys_config = config.get('physiological', {})
        session_timeout = phys_config.get('session_timeout_minutes', 5) / 60
        
        if time_delta_hours < session_timeout:
            session_variance_threshold = phys_config.get('session_variance_threshold', 5.0)
            if change > session_variance_threshold:
                return False, (f"Session variance {change:.1f}kg exceeds threshold "
                              f"{session_variance_threshold}kg (likely different user)")
        
        return True, None

    @staticmethod
    def _initialize_kalman_immediate(
        weight: float,
        timestamp: datetime,
        kalman_config: dict
    ) -> Dict[str, Any]:
        """Initialize Kalman filter immediately with first measurement."""
        initial_variance = kalman_config.get("initial_variance", 1.0)
        
        kalman_params = {
            'initial_state_mean': [weight, 0],
            'initial_state_covariance': [[initial_variance, 0], [0, 0.001]],
            'transition_covariance': [
                [kalman_config["transition_covariance_weight"], 0],
                [0, kalman_config["transition_covariance_trend"]]
            ],
            'observation_covariance': [[kalman_config["observation_covariance"]]],
        }
        
        return {
            'kalman_params': kalman_params,
            'last_state': np.array([[weight, 0]]),
            'last_covariance': np.array([[[initial_variance, 0], [0, 0.001]]]),
            'last_timestamp': timestamp,
            'last_raw_weight': weight,
        }
    
    @staticmethod
    def _update_kalman_state(
        state: Dict[str, Any],
        weight: float,
        timestamp: datetime,
        source: str,
        processing_config: dict
    ) -> Dict[str, Any]:
        """Update Kalman filter state with new measurement."""
        time_delta_days = 1.0
        if state.get('last_timestamp'):
            last_timestamp = state['last_timestamp']
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp)
            delta = (timestamp - last_timestamp).total_seconds() / 86400.0
            time_delta_days = max(0.1, min(30.0, delta))
        
        kalman_params = state['kalman_params']
        kalman = KalmanFilter(
            transition_matrices=np.array([[1, time_delta_days], [0, 1]]),
            observation_matrices=np.array([[1, 0]]),
            initial_state_mean=np.array(kalman_params['initial_state_mean']),
            initial_state_covariance=np.array(kalman_params['initial_state_covariance']),
            transition_covariance=np.array(kalman_params['transition_covariance']),
            observation_covariance=np.array(kalman_params['observation_covariance']),
        )
        
        observation = np.array([[weight]])
        
        if state.get('last_state') is None:
            filtered_state_means, filtered_state_covariances = kalman.filter(observation)
            new_last_state = filtered_state_means
            new_last_covariance = filtered_state_covariances
        else:
            last_state = state['last_state']
            last_covariance = state['last_covariance']
            
            if len(last_state.shape) > 1:
                current_state = last_state[-1]
                current_covariance = last_covariance[-1]
            else:
                current_state = last_state
                current_covariance = last_covariance[0] if len(last_covariance.shape) > 2 else last_covariance
            
            filtered_state_mean, filtered_state_covariance = kalman.filter_update(
                current_state,
                current_covariance,
                observation=observation[0],
            )
            
            if len(last_state.shape) == 1:
                new_last_state = np.array([last_state, filtered_state_mean])
                new_last_covariance = np.array([last_covariance, filtered_state_covariance])
            else:
                new_last_state = np.array([last_state[-1], filtered_state_mean])
                new_last_covariance = np.array([last_covariance[-1], filtered_state_covariance])
        
        state['last_state'] = new_last_state
        state['last_covariance'] = new_last_covariance
        state['last_timestamp'] = timestamp
        state['last_raw_weight'] = weight
        
        return state

    @staticmethod
    def _reset_kalman_state(state: Dict[str, Any], new_weight: float) -> Dict[str, Any]:
        """Reset Kalman state after long gap."""
        state['last_state'] = np.array([[new_weight, 0]])
        state['last_covariance'] = np.array([[[1.0, 0], [0, 0.001]]])
        state['last_raw_weight'] = new_weight
        
        if state.get('kalman_params'):
            state['kalman_params']['initial_state_mean'] = [new_weight, 0]
        
        return state

    @staticmethod
    def _calculate_confidence(normalized_innovation: float) -> float:
        """Calculate confidence using smooth exponential decay."""
        alpha = 0.5
        return np.exp(-alpha * normalized_innovation ** 2)

    @staticmethod
    def _create_result(
        state: Dict[str, Any],
        weight: float,
        timestamp: datetime,
        source: str,
        accepted: bool
    ) -> Dict[str, Any]:
        """Create result dictionary from current state."""
        if state.get('last_state') is None:
            return None
        
        last_state = state['last_state']
        if len(last_state.shape) > 1:
            current_state = last_state[-1]
        else:
            current_state = last_state
        
        filtered_weight = current_state[0]
        trend = current_state[1]
        
        innovation = weight - filtered_weight
        
        last_covariance = state.get('last_covariance')
        if last_covariance is not None:
            if len(last_covariance.shape) > 2:
                current_covariance = last_covariance[-1]
            else:
                current_covariance = last_covariance
            
            obs_covariance = state['kalman_params']['observation_covariance'][0][0]
            innovation_variance = current_covariance[0, 0] + obs_covariance
            normalized_innovation = (
                abs(innovation) / np.sqrt(innovation_variance)
                if innovation_variance > 0
                else 0
            )
        else:
            normalized_innovation = 0
        
        confidence = WeightProcessor._calculate_confidence(normalized_innovation)
        
        return {
            "timestamp": timestamp,
            "raw_weight": weight,
            "filtered_weight": float(filtered_weight),
            "trend": float(trend),
            "trend_weekly": float(trend * 7),
            "accepted": accepted,
            "confidence": float(confidence),
            "innovation": float(innovation),
            "normalized_innovation": float(normalized_innovation),
            "source": source,
        }

    @staticmethod
    def get_user_state(user_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state for a user (for debugging/inspection)."""
        db = get_state_db()
        return db.get_state(user_id)

    @staticmethod
    def reset_user(user_id: str) -> bool:
        """Reset a user's state (delete from database)."""
        db = get_state_db()
        return db.delete_state(user_id)


def categorize_rejection_enhanced(reason: str) -> str:
    """Enhanced categorization including BMI and unit issues."""
    reason_lower = reason.lower()
    
    if "bmi" in reason_lower:
        return "BMI_Detection"
    elif "unit" in reason_lower or "pound" in reason_lower or "conversion" in reason_lower:
        return "Unit_Conversion"
    elif "physiological" in reason_lower:
        return "Physiological_Limit"
    elif "outside bounds" in reason_lower:
        return "Bounds"
    elif "extreme deviation" in reason_lower:
        return "Extreme"
    elif "session variance" in reason_lower or "different user" in reason_lower:
        return "Variance"
    elif "sustained" in reason_lower:
        return "Sustained"
    elif "daily fluctuation" in reason_lower:
        return "Daily"
    else:
        return "Other"


def get_rejection_severity(reason: str, weight_change: float = 0) -> str:
    """Determine severity of rejection."""
    reason_lower = reason.lower()
    
    if "impossible" in reason_lower or "physiologically impossible" in reason_lower:
        return "Critical"
    elif "extreme" in reason_lower or weight_change > 20:
        return "High"
    elif "suspicious" in reason_lower or weight_change > 10:
        return "Medium"
    else:
        return "Low"


class DataQualityPreprocessor:
    """Pre-process and clean data before Kalman filtering."""
    
    # BMI detection parameters
    DEFAULT_HEIGHT_M = 1.67
    BMI_IMPOSSIBLE_LOW = 10
    BMI_IMPOSSIBLE_HIGH = 100
    BMI_SUSPICIOUS_LOW = 13
    BMI_SUSPICIOUS_HIGH = 60
    
    # Height data cache
    _height_data = None
    _height_data_loaded = False
    
    @classmethod
    def load_height_data(cls):
        """Load height data from CSV file once."""
        if not cls._height_data_loaded:
            try:
                df = pd.read_csv('data/2025-09-11_height_values_latest.csv')
                df['value_m'] = df.apply(lambda row: cls._convert_height_to_meters(
                    row['value_quantity'], row['unit']
                ), axis=1)
                cls._height_data = df.set_index('user_id')['value_m'].to_dict()
                cls._height_data_loaded = True
                print(f"Loaded height data for {len(cls._height_data)} users")
            except Exception as e:
                print(f"Could not load height data: {e}")
                cls._height_data = {}
                cls._height_data_loaded = True
    
    @staticmethod
    def _convert_height_to_meters(value: float, unit: str) -> float:
        """Convert height to meters."""
        unit_lower = unit.lower() if unit else 'm'
        
        if 'cm' in unit_lower or 'centimeter' in unit_lower:
            return value / 100.0
        elif 'in' in unit_lower or 'inch' in unit_lower:
            return value * 0.0254
        elif 'ft' in unit_lower or 'feet' in unit_lower:
            return value * 0.3048
        elif 'm' in unit_lower or 'meter' in unit_lower:
            return value
        else:
            return value / 100.0
    
    @classmethod
    def get_user_height(cls, user_id: str) -> float:
        """Get user's height in meters, using default if not found."""
        if not cls._height_data_loaded:
            cls.load_height_data()
        
        return cls._height_data.get(user_id, cls.DEFAULT_HEIGHT_M)
    
    @staticmethod
    def preprocess(weight: float, source: str, timestamp: datetime, user_id: Optional[str] = None, unit: str = 'kg') -> Tuple[Optional[float], Dict]:
        """
        Clean and standardize weight data with BMI detection.
        
        Args:
            weight: The weight value
            source: Data source identifier
            timestamp: Measurement timestamp
            user_id: User identifier for height lookup
            unit: Unit of the weight measurement ('kg', 'lb', 'lbs', 'pound', 'pounds', etc.)
        
        Returns:
            (cleaned_weight, metadata) or (None, metadata) if rejected
        """
        metadata = {
            'original_weight': weight,
            'original_unit': unit,
            'source': source,
            'timestamp': timestamp.isoformat(),
            'corrections': [],
            'warnings': [],
            'checks_passed': []
        }
        
        # Get user's actual height if available
        user_height = DataQualityPreprocessor.get_user_height(user_id) if user_id else DataQualityPreprocessor.DEFAULT_HEIGHT_M
        
        # 1. Convert units based on explicit unit parameter
        unit_lower = unit.lower() if unit else 'kg'
        if unit_lower in ['lb', 'lbs', 'pound', 'pounds']:
            # Explicitly marked as pounds - convert
            weight_kg = weight * 0.453592
            metadata['corrections'].append(f'Converted {weight:.1f} {unit} to {weight_kg:.1f} kg')
            weight = weight_kg
        elif unit_lower in ['st', 'stone', 'stones']:
            # Explicitly marked as stones - convert
            weight_kg = weight * 6.35029
            metadata['corrections'].append(f'Converted {weight:.1f} {unit} to {weight_kg:.1f} kg')
            weight = weight_kg
        elif unit_lower not in ['kg', 'kilogram', 'kilograms']:
            # Unknown unit - add warning but continue
            metadata['warnings'].append(f'Unknown unit: {unit}')
        
        # 2. BMI detection - only for kg values that look suspiciously like BMI
        if unit_lower in ['kg', 'kilogram', 'kilograms'] and 15 <= weight <= 50:
            # Calculate what weight would be for this BMI
            implied_weight = weight * (user_height ** 2)
            
            # Check if implied weight is reasonable
            if 40 <= implied_weight <= 200:
                # High confidence this is BMI
                metadata['warnings'].append(
                    f'Value {weight:.1f} likely BMI (implies {implied_weight:.1f}kg weight for height {user_height:.2f}m)'
                )
                metadata['corrections'].append(f'Converted BMI {weight:.1f} to weight {implied_weight:.1f}kg')
                weight = implied_weight
            elif 30 <= implied_weight <= 250:
                # Moderate confidence - check source
                if 'connectivehealth' in source.lower():
                    metadata['warnings'].append(f'Value {weight:.1f} appears to be BMI from {source}')
                    metadata['corrections'].append(f'Converted BMI {weight:.1f} to weight {implied_weight:.1f}kg')
                    weight = implied_weight
                else:
                    metadata['warnings'].append(f'Value {weight:.1f} might be BMI')
        
        # 3. Validate against physiological limits using BMI
        implied_bmi = weight / (user_height ** 2)
        
        if implied_bmi < DataQualityPreprocessor.BMI_IMPOSSIBLE_LOW:
            metadata['warnings'].append(
                f'Implied BMI {implied_bmi:.1f} physiologically impossible (height: {user_height:.2f}m)'
            )
            metadata['rejected'] = f'BMI {implied_bmi:.1f} outside physiological limits'
            return None, metadata
        
        if implied_bmi > DataQualityPreprocessor.BMI_IMPOSSIBLE_HIGH:
            metadata['rejected'] = f'Implied BMI {implied_bmi:.1f} physiologically impossible'
            return None, metadata
        
        # Warnings for suspicious but possible values
        if implied_bmi < DataQualityPreprocessor.BMI_SUSPICIOUS_LOW:
            metadata['warnings'].append(f'Implied BMI {implied_bmi:.1f} suspiciously low')
        
        if implied_bmi > DataQualityPreprocessor.BMI_SUSPICIOUS_HIGH:
            metadata['warnings'].append(f'Implied BMI {implied_bmi:.1f} suspiciously high')
        
        metadata['checks_passed'].append('physiological_limits')
        metadata['implied_bmi'] = round(implied_bmi, 1)
        metadata['user_height_m'] = round(user_height, 2)
        
        # Add BMI classification
        if implied_bmi < 18.5:
            metadata['bmi_category'] = 'underweight'
        elif implied_bmi < 25:
            metadata['bmi_category'] = 'normal'
        elif implied_bmi < 30:
            metadata['bmi_category'] = 'overweight'
        else:
            metadata['bmi_category'] = 'obese'
        
        # 4. Flag high-risk sources
        if 'iglucose' in source.lower():
            metadata['warnings'].append('High-outlier source - increased scrutiny')
            metadata['high_risk'] = True
        
        return weight, metadata


class AdaptiveOutlierDetector:
    """Adaptive outlier detection based on source reliability."""
    
    # Outlier rates per 1000 from analysis
    SOURCE_PROFILES = {
        'care-team-upload': {
            'outlier_rate': 3.6,
            'reliability': 'excellent'
        },
        'patient-upload': {
            'outlier_rate': 13.0,
            'reliability': 'excellent'
        },
        'internal-questionnaire': {
            'outlier_rate': 14.0,
            'reliability': 'good'
        },
        'patient-device': {
            'outlier_rate': 20.7,
            'reliability': 'good'
        },
        'https://connectivehealth.io': {
            'outlier_rate': 35.8,
            'reliability': 'moderate'
        },
        'https://api.iglucose.com': {
            'outlier_rate': 151.4,
            'reliability': 'poor'
        }
    }
    
    @staticmethod
    def get_adaptive_threshold(source: str, time_gap_days: int) -> float:
        """
        Get outlier threshold based on source reliability and time gap.
        
        Args:
            source: Data source identifier
            time_gap_days: Days since last measurement
            
        Returns:
            Maximum acceptable weight change in kg
        """
        profile = AdaptiveOutlierDetector.SOURCE_PROFILES.get(source, {})
        outlier_rate = profile.get('outlier_rate', 50.0)
        
        # Base physiological limit: 2kg/week is extreme but possible
        physiological_rate = 2.0 / 7.0  # kg per day
        
        if outlier_rate > 100:  # Very unreliable (iGlucose)
            # Much stricter - only allow 1kg/week
            max_rate = 1.0 / 7.0
            threshold = max(3.0, max_rate * time_gap_days)
        elif outlier_rate > 30:  # Moderate reliability
            # Standard physiological limit
            max_rate = physiological_rate
            threshold = max(5.0, max_rate * time_gap_days)
        else:  # Excellent reliability
            # More lenient - trust the source more
            max_rate = physiological_rate * 1.5
            threshold = max(10.0, max_rate * time_gap_days)
        
        # Cap at reasonable maximum
        return min(threshold, 20.0)
    
    @staticmethod
    def check_outlier(
        weight_change: float,
        source: str,
        time_gap_days: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if weight change is an outlier.
        
        Returns:
            (is_outlier, reason)
        """
        threshold = AdaptiveOutlierDetector.get_adaptive_threshold(source, time_gap_days)
        
        if abs(weight_change) > threshold:
            profile = AdaptiveOutlierDetector.SOURCE_PROFILES.get(source, {})
            reliability = profile.get('reliability', 'unknown')
            
            reason = (f"Weight change {weight_change:.1f}kg exceeds threshold "
                     f"{threshold:.1f}kg for {reliability} source over {time_gap_days} days")
            return True, reason
        
        return False, None


class AdaptiveKalmanConfig:
    """Adapt Kalman filter parameters based on source quality."""
    
    # Noise multipliers based on source reliability
    NOISE_MULTIPLIERS = {
        'care-team-upload': 0.5,           # Most reliable - trust more
        'patient-upload': 0.7,              # Very reliable
        'internal-questionnaire': 0.8,      # Reliable but sparse
        'patient-device': 1.0,              # Baseline
        'https://connectivehealth.io': 1.5, # Less reliable
        'https://api.iglucose.com': 3.0     # Least reliable - trust less
    }
    
    @staticmethod
    def get_adapted_config(source: str, base_config: Dict) -> Dict:
        """
        Adapt Kalman configuration based on source reliability.
        
        Args:
            source: Data source identifier
            base_config: Base Kalman configuration
            
        Returns:
            Adapted configuration
        """
        config = base_config.copy()
        
        # Get noise multiplier for this source
        multiplier = AdaptiveKalmanConfig.NOISE_MULTIPLIERS.get(source, 1.0)
        
        # Adjust measurement noise
        # Higher noise = Kalman trusts measurement less
        base_noise = config.get('measurement_noise', 1.0)
        config['measurement_noise'] = base_noise * multiplier
        
        # For very unreliable sources, also increase initial uncertainty
        if multiplier > 2.0:
            config['initial_uncertainty'] = config.get('initial_uncertainty', 10.0) * 1.5
        
        return config


class SourceQualityMonitor:
    """Monitor source quality in real-time."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize monitor with rolling window."""
        self.window_size = window_size
        self.stats = defaultdict(lambda: {
            'measurements': deque(maxlen=window_size),
            'outliers': deque(maxlen=window_size),
            'rejections': deque(maxlen=window_size),
            'alerts': []
        })
    
    def record_measurement(
        self,
        source: str,
        is_outlier: bool,
        is_rejected: bool
    ) -> Optional[Dict]:
        """
        Record measurement and check for quality issues.
        
        Returns:
            Alert dictionary if quality issue detected, None otherwise
        """
        stats = self.stats[source]
        stats['measurements'].append(1)
        stats['outliers'].append(1 if is_outlier else 0)
        stats['rejections'].append(1 if is_rejected else 0)
        
        # Need minimum measurements for statistics
        if len(stats['measurements']) < 100:
            return None
        
        # Calculate current rates
        outlier_rate = sum(stats['outliers']) / len(stats['measurements']) * 1000
        rejection_rate = sum(stats['rejections']) / len(stats['measurements']) * 1000
        
        # Get expected rates
        profile = AdaptiveOutlierDetector.SOURCE_PROFILES.get(source, {})
        expected_outlier_rate = profile.get('outlier_rate', 30.0)
        
        alert = None
        
        # Check for degradation
        if outlier_rate > expected_outlier_rate * 1.5:
            alert = {
                'type': 'quality_degradation',
                'source': source,
                'metric': 'outlier_rate',
                'current': outlier_rate,
                'expected': expected_outlier_rate,
                'severity': 'high' if outlier_rate > expected_outlier_rate * 2 else 'medium',
                'message': f"{source} outlier rate {outlier_rate:.1f}/1000 "
                          f"exceeds expected {expected_outlier_rate:.1f}/1000"
            }
            stats['alerts'].append(alert)
        
        # Check for improvement in bad sources
        elif source == 'https://api.iglucose.com' and outlier_rate < 100:
            alert = {
                'type': 'quality_improvement',
                'source': source,
                'metric': 'outlier_rate',
                'current': outlier_rate,
                'expected': expected_outlier_rate,
                'severity': 'info',
                'message': f"{source} showing improvement: {outlier_rate:.1f}/1000"
            }
        
        return alert
    
    def get_source_summary(self, source: str) -> Dict:
        """Get quality summary for a source."""
        stats = self.stats[source]
        
        if not stats['measurements']:
            return {'status': 'no_data'}
        
        total = len(stats['measurements'])
        outlier_rate = sum(stats['outliers']) / total * 1000 if total > 0 else 0
        rejection_rate = sum(stats['rejections']) / total * 1000 if total > 0 else 0
        
        return {
            'measurements': total,
            'outlier_rate': outlier_rate,
            'rejection_rate': rejection_rate,
            'recent_alerts': stats['alerts'][-5:] if stats['alerts'] else []
        }


# Global monitor instance
quality_monitor = SourceQualityMonitor()


def process_weight_enhanced(
    user_id: str,
    weight: float,
    timestamp: datetime,
    source: str,
    processing_config: Dict,
    kalman_config: Dict,
    unit: str = 'kg'
) -> Optional[Dict]:
    """
    Enhanced weight processing with data quality improvements.
    
    This wraps the original WeightProcessor with additional defensive layers:
    1. Pre-processing for unit conversion and data cleaning
    2. Adaptive outlier thresholds based on source reliability
    3. Kalman noise adaptation based on source quality
    4. Real-time quality monitoring
    
    Args:
        user_id: User identifier
        weight: Weight measurement value
        timestamp: Measurement timestamp
        source: Data source identifier
        processing_config: Processing configuration
        kalman_config: Kalman filter configuration
        unit: Unit of measurement ('kg', 'lb', 'lbs', 'pound', 'pounds', etc.)
        
    Returns:
        Processing result with additional metadata, or None if rejected
    """
    
    # Step 1: Pre-process data with unit conversion and BMI detection
    cleaned_weight, preprocess_metadata = DataQualityPreprocessor.preprocess(
        weight, source, timestamp, user_id, unit
    )
    
    if cleaned_weight is None:
        return {
            'rejected': True,
            'accepted': False,
            'stage': 'preprocessing',
            'reason': preprocess_metadata.get('rejected'),
            'rejection_reason': preprocess_metadata.get('rejected'),
            'metadata': preprocess_metadata,
            'timestamp': timestamp,
            'source': source,
            'raw_weight': weight,
            'original_weight': weight,
            'original_unit': unit,
            'bmi_details': {
                'detected_as_bmi': 'BMI' in preprocess_metadata.get('rejected', ''),
                'user_height_m': preprocess_metadata.get('user_height_m'),
                'implied_bmi': preprocess_metadata.get('implied_bmi'),
                'bmi_category': preprocess_metadata.get('bmi_category')
            }
        }
    
    # Step 2: Adapt configuration based on source
    # Get time gap for adaptive thresholds
    db = get_state_db()
    state = db.get_state(user_id)
    
    time_gap_days = 0
    if state and state.get('last_timestamp'):
        time_gap_days = (timestamp - state['last_timestamp']).days
    
    # FIXED: Use unified threshold calculator with explicit percentage unit
    adapted_config = processing_config.copy()
    
    # Get threshold as percentage (what the base processor expects)
    threshold_result = ThresholdCalculator.get_extreme_deviation_threshold(
        source=source,
        time_gap_days=time_gap_days,
        current_weight=cleaned_weight,  # Use cleaned weight as reference
        unit='percentage'  # Explicitly request percentage
    )
    
    adapted_config['extreme_threshold'] = threshold_result.value
    
    # Store both units for debugging/transparency
    adapted_config['extreme_threshold_pct'] = threshold_result.value
    adapted_config['extreme_threshold_kg'] = threshold_result.metadata.get('absolute_threshold_kg')
    
    # Step 3: Adapt Kalman configuration using unified calculator
    adapted_kalman = kalman_config.copy()
    
    # Get noise multiplier from threshold calculator
    noise_multiplier = ThresholdCalculator.get_measurement_noise_multiplier(source)
    
    # Adjust measurement noise based on source reliability
    base_noise = kalman_config.get('observation_covariance', 1.0)
    adapted_kalman['observation_covariance'] = base_noise * noise_multiplier
    
    # For very unreliable sources, also increase initial uncertainty
    if noise_multiplier > 2.0:
        adapted_kalman['initial_variance'] = kalman_config.get('initial_variance', 1.0) * 1.5
    
    # Step 4: Process with original Kalman filter
    result = WeightProcessor.process_weight(
        user_id=user_id,
        weight=cleaned_weight,
        timestamp=timestamp,
        source=source,
        processing_config=adapted_config,
        kalman_config=adapted_kalman
    )
    
    # Step 5: Monitor and enhance result
    if result:
        # Check if it was an outlier
        is_outlier = False
        outlier_reason = None
        if state and state.get('last_state') is not None:
            last_state = state['last_state']
            # Extract weight value from state
            try:
                if isinstance(last_state, (list, tuple, np.ndarray)):
                    last_weight = float(last_state[0])
                else:
                    last_weight = float(last_state)
                
                weight_change = abs(cleaned_weight - last_weight)
                is_outlier, outlier_reason = AdaptiveOutlierDetector.check_outlier(
                    weight_change, source, time_gap_days
                )
            except (IndexError, TypeError, ValueError):
                # If we can't extract last weight, skip outlier check
                pass
        
        # Record for monitoring
        is_rejected = result.get('rejected', False)
        alert = quality_monitor.record_measurement(source, is_outlier, is_rejected)
        
        # Enhance result with metadata
        result['preprocessing_metadata'] = preprocess_metadata
        
        # Include threshold information with units
        result['threshold_info'] = {
            'extreme_threshold_pct': adapted_config.get('extreme_threshold_pct'),
            'extreme_threshold_kg': adapted_config.get('extreme_threshold_kg'),
            'source_reliability': ThresholdCalculator.get_source_reliability(source),
            'measurement_noise_multiplier': noise_multiplier if 'noise_multiplier' in locals() else 1.0
        }
        
        # Legacy fields for backward compatibility
        result['adaptive_threshold'] = adapted_config['extreme_threshold']
        result['measurement_noise_used'] = adapted_kalman.get('observation_covariance', 1.0)
        
        # Add BMI details
        user_height = DataQualityPreprocessor.get_user_height(user_id)
        implied_bmi = round(cleaned_weight / (user_height ** 2), 1)
        
        # Determine BMI category
        if implied_bmi < 18.5:
            bmi_category = 'underweight'
        elif implied_bmi < 25:
            bmi_category = 'normal'
        elif implied_bmi < 30:
            bmi_category = 'overweight'
        else:
            bmi_category = 'obese'
        
        result['bmi_details'] = {
            'user_height_m': user_height,
            'original_weight': weight,
            'original_unit': unit,
            'cleaned_weight': cleaned_weight,
            'implied_bmi': implied_bmi,
            'bmi_category': bmi_category,
            'bmi_converted': weight != cleaned_weight and 15 <= weight <= 50,
            'unit_converted': any('Converted' in c for c in preprocess_metadata.get('corrections', [])),
            'corrections': preprocess_metadata.get('corrections', []),
            'warnings': preprocess_metadata.get('warnings', [])
        }
        
        # Add rejection insights
        if result.get('accepted') == False:
            rejection_reason = result.get('reason', '')
            result['rejection_insights'] = {
                'category': categorize_rejection_enhanced(rejection_reason),
                'severity': get_rejection_severity(rejection_reason, weight_change if 'weight_change' in locals() else 0),
                'source_reliability': AdaptiveOutlierDetector.SOURCE_PROFILES.get(source, {}).get('reliability', 'unknown'),
                'adaptive_threshold_used': adapted_config['extreme_threshold'],
                'outlier_detected': is_outlier,
                'outlier_reason': outlier_reason
            }
        
        if alert:
            result['quality_alert'] = alert
        
        # Add source quality summary
        result['source_quality'] = quality_monitor.get_source_summary(source)
    
    return result

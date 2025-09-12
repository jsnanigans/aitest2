"""
Validation logic for weight measurements.
"""

from datetime import datetime
from typing import Dict, Tuple, Optional, Any, Literal
import numpy as np
try:
    from .models import (
        ThresholdResult, 
        SOURCE_PROFILES, 
        DEFAULT_PROFILE,
        BMI_LIMITS,
        PHYSIOLOGICAL_LIMITS
    )
except ImportError:
    from models import (
        ThresholdResult, 
        SOURCE_PROFILES, 
        DEFAULT_PROFILE,
        BMI_LIMITS,
        PHYSIOLOGICAL_LIMITS
    )


class PhysiologicalValidator:
    """Validates weight measurements against physiological limits."""
    
    @staticmethod
    def calculate_time_delta_hours(
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
    def get_physiological_limit(
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
    def validate_weight(
        weight: float,
        config: dict,
        state: Optional[Dict[str, Any]],
        timestamp: Optional[datetime] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate weight with physiological limits.
        Returns (is_valid, rejection_reason).
        """
        min_weight = config.get('min_weight', PHYSIOLOGICAL_LIMITS['MIN_WEIGHT'])
        max_weight = config.get('max_weight', PHYSIOLOGICAL_LIMITS['MAX_WEIGHT'])
        
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
            time_delta_hours = PhysiologicalValidator.calculate_time_delta_hours(
                timestamp, state.get('last_timestamp')
            )
        else:
            time_delta_hours = 24
        
        change = abs(weight - last_weight)
        max_change, reason = PhysiologicalValidator.get_physiological_limit(
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
        
        session_timeout = phys_config.get('session_timeout_minutes', 5) / 60
        
        if time_delta_hours < session_timeout:
            session_variance_threshold = phys_config.get('session_variance_threshold', 5.0)
            if change > session_variance_threshold:
                return False, (f"Session variance {change:.1f}kg exceeds threshold "
                              f"{session_variance_threshold}kg (likely different user)")
        
        return True, None


class BMIValidator:
    """
    Validates weight measurements using BMI thresholds and percentage changes.
    Detects when Kalman filter should be reset due to unrealistic changes.
    """
    
    @staticmethod
    def calculate_bmi(weight_kg: float, height_m: Optional[float]) -> Optional[float]:
        """Calculate BMI if height is available."""
        if height_m is None or height_m <= 0:
            return None
        return weight_kg / (height_m ** 2)
    
    @staticmethod
    def should_reset_kalman(
        current_weight: float,
        last_weight: float,
        time_delta_hours: float,
        height_m: Optional[float] = None,
        source: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if Kalman filter should be reset based on physiological impossibility.
        
        Returns:
            (should_reset, reason)
        """
        if last_weight <= 0:
            return False, None
        
        weight_change = current_weight - last_weight
        pct_change = (weight_change / last_weight) * 100
        daily_rate = (weight_change / (time_delta_hours / 24)) if time_delta_hours > 0 else weight_change
        
        if current_weight < PHYSIOLOGICAL_LIMITS['MIN_WEIGHT']:
            return True, f"Weight below minimum threshold: {current_weight:.1f}kg"
        
        if current_weight > 300:
            return True, f"Weight above maximum threshold: {current_weight:.1f}kg"
        
        if abs(pct_change) > 50:
            return True, f"Extreme change > 50%: {pct_change:+.1f}%"
        
        if time_delta_hours < 1 and abs(pct_change) > 30:
            return True, f"Instant change > 30%: {pct_change:+.1f}% in {time_delta_hours:.1f}h"
        
        if time_delta_hours <= 24:
            if abs(pct_change) > 20:
                return True, f"Daily change > 20%: {pct_change:+.1f}%"
            if abs(weight_change) > 10:
                return True, f"Daily change > 10kg: {weight_change:+.1f}kg"
        
        if time_delta_hours <= 168:
            if abs(pct_change) > 30:
                return True, f"Weekly change > 30%: {pct_change:+.1f}%"
        
        if abs(daily_rate) > 3.0:
            return True, f"Daily rate > 3kg/day: {daily_rate:+.1f}kg/day"
        
        if height_m and height_m > 0:
            current_bmi = BMIValidator.calculate_bmi(current_weight, height_m)
            last_bmi = BMIValidator.calculate_bmi(last_weight, height_m)
            
            if current_bmi:
                if current_bmi < BMI_LIMITS['CRITICAL_LOW']:
                    return True, f"Critical BMI < {BMI_LIMITS['CRITICAL_LOW']}: {current_bmi:.1f}"
                
                if current_bmi > BMI_LIMITS['CRITICAL_HIGH']:
                    return True, f"Critical BMI > {BMI_LIMITS['CRITICAL_HIGH']}: {current_bmi:.1f}"
                
                if current_bmi < BMI_LIMITS['SEVERE_LOW'] and abs(pct_change) > 20:
                    return True, f"Severe underweight (BMI {current_bmi:.1f}) with {pct_change:+.1f}% change"
                
                if current_bmi > BMI_LIMITS['MORBID_OBESE'] and abs(pct_change) > 20:
                    return True, f"Morbid obesity (BMI {current_bmi:.1f}) with {pct_change:+.1f}% change"
                
                if last_bmi:
                    bmi_change = abs(current_bmi - last_bmi)
                    if bmi_change > 10:
                        return True, f"BMI change > 10: {last_bmi:.1f} → {current_bmi:.1f}"
        
        suspicious_sources = {'iglucose', 'api.iglucose.com', 'https://api.iglucose.com'}
        if source and any(s in source.lower() for s in suspicious_sources):
            if abs(pct_change) > 25:
                return True, f"Suspicious source with > 25% change: {pct_change:+.1f}%"
        
        return False, None
    
    @staticmethod
    def get_confidence_multiplier(
        current_weight: float,
        last_weight: float,
        time_delta_hours: float,
        height_m: Optional[float] = None
    ) -> float:
        """
        Get a confidence multiplier for Kalman filter based on measurement plausibility.
        Lower values = less trust in measurement.
        
        Returns:
            Multiplier between 0.1 and 1.0
        """
        if last_weight <= 0:
            return 1.0
        
        pct_change = abs((current_weight - last_weight) / last_weight) * 100
        
        if pct_change < 5:
            base_confidence = 1.0
        elif pct_change < 10:
            base_confidence = 0.9
        elif pct_change < 15:
            base_confidence = 0.7
        elif pct_change < 20:
            base_confidence = 0.5
        elif pct_change < 30:
            base_confidence = 0.3
        else:
            base_confidence = 0.1
        
        if time_delta_hours < 1:
            time_factor = 0.5
        elif time_delta_hours < 24:
            time_factor = 0.8
        elif time_delta_hours < 168:
            time_factor = 0.9
        else:
            time_factor = 1.0
        
        bmi_factor = 1.0
        if height_m and height_m > 0:
            bmi = BMIValidator.calculate_bmi(current_weight, height_m)
            if bmi:
                if bmi < BMI_LIMITS['SEVERE_LOW'] or bmi > BMI_LIMITS['MORBID_OBESE']:
                    bmi_factor = 0.5
                elif bmi < BMI_LIMITS['UNDERWEIGHT'] or bmi > BMI_LIMITS['SEVERE_OBESE']:
                    bmi_factor = 0.7
                else:
                    bmi_factor = 1.0
        
        return max(0.1, base_confidence * time_factor * bmi_factor)
    
    @staticmethod
    def get_rejection_reason(
        current_weight: float,
        last_weight: float,
        time_delta_hours: float,
        height_m: Optional[float] = None
    ) -> Optional[str]:
        """
        Get a detailed rejection reason for impossible measurements.
        """
        if last_weight <= 0:
            return None
        
        weight_change = current_weight - last_weight
        pct_change = (weight_change / last_weight) * 100
        
        if current_weight < PHYSIOLOGICAL_LIMITS['MIN_WEIGHT']:
            return f"Weight {current_weight:.1f}kg below human minimum"
        
        if current_weight > 300:
            return f"Weight {current_weight:.1f}kg above human maximum"
        
        if height_m and height_m > 0:
            bmi = BMIValidator.calculate_bmi(current_weight, height_m)
            if bmi:
                if bmi < 13:
                    return f"BMI {bmi:.1f} incompatible with life"
                if bmi > 60:
                    return f"BMI {bmi:.1f} physiologically impossible"
        
        if time_delta_hours <= 1:
            max_change = min(3.0, last_weight * 0.03)
            if abs(weight_change) > max_change:
                return f"Change {weight_change:+.1f}kg exceeds 1-hour limit of {max_change:.1f}kg"
        
        elif time_delta_hours <= 24:
            max_change = min(5.0, last_weight * 0.05)
            if abs(weight_change) > max_change:
                return f"Change {weight_change:+.1f}kg exceeds daily limit of {max_change:.1f}kg"
        
        elif time_delta_hours <= 168:
            max_change = min(7.0, last_weight * 0.07)
            if abs(weight_change) > max_change:
                return f"Change {weight_change:+.1f}kg exceeds weekly limit of {max_change:.1f}kg"
        
        else:
            days = time_delta_hours / 24
            max_sustained_rate = PHYSIOLOGICAL_LIMITS['MAX_SUSTAINED_DAILY_KG']
            max_change = days * max_sustained_rate
            if abs(weight_change) > max_change:
                return f"Change {weight_change:+.1f}kg exceeds sustained rate of {max_sustained_rate}kg/day over {days:.0f} days"
        
        return None


class ThresholdCalculator:
    """
    Unified threshold calculator with explicit unit handling.
    All methods are static to maintain stateless architecture.
    """
    
    @staticmethod
    def get_extreme_deviation_threshold(
        source: str,
        time_gap_days: float,
        current_weight: float,
        unit: Literal['percentage', 'kg'] = 'percentage'
    ) -> ThresholdResult:
        """
        Calculate adaptive threshold for extreme deviation detection.
        
        Args:
            source: Data source identifier
            time_gap_days: Days since last measurement
            current_weight: Current weight in kg (for percentage calculation)
            unit: 'percentage' (0.0-1.0) or 'kg' for absolute weight
            
        Returns:
            ThresholdResult with value in requested units and metadata
        """
        profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
        
        absolute_kg = ThresholdCalculator._calculate_adaptive_threshold_kg(
            profile['outlier_rate'],
            time_gap_days
        )
        
        metadata = {
            'source': source,
            'source_reliability': profile['reliability'],
            'outlier_rate_per_1000': profile['outlier_rate'],
            'time_gap_days': time_gap_days,
            'weight_reference_kg': current_weight,
            'absolute_threshold_kg': absolute_kg
        }
        
        if unit == 'percentage':
            percentage = absolute_kg / current_weight
            
            MIN_PERCENTAGE = 0.03
            MAX_PERCENTAGE = 0.30
            
            bounded_percentage = max(MIN_PERCENTAGE, min(percentage, MAX_PERCENTAGE))
            
            metadata['unbounded_percentage'] = percentage
            metadata['bounds_applied'] = percentage != bounded_percentage
            metadata['percentage_threshold'] = bounded_percentage
            
            return ThresholdResult(bounded_percentage, 'percentage', metadata)
            
        elif unit == 'kg':
            return ThresholdResult(absolute_kg, 'kg', metadata)
            
        else:
            raise ValueError(f"Unknown unit: {unit}. Use 'percentage' or 'kg'")
    
    @staticmethod
    def _calculate_adaptive_threshold_kg(
        outlier_rate: float,
        time_gap_days: float
    ) -> float:
        """
        Calculate absolute threshold in kg based on source reliability.
        
        Args:
            outlier_rate: Outliers per 1000 measurements
            time_gap_days: Days since last measurement
            
        Returns:
            Threshold in kg
        """
        physiological_rate_kg_per_day = 2.0 / 7.0
        
        if outlier_rate > 100:
            max_rate = 1.0 / 7.0
            threshold = max(3.0, max_rate * time_gap_days)
        elif outlier_rate > 30:
            max_rate = physiological_rate_kg_per_day
            threshold = max(5.0, max_rate * time_gap_days)
        else:
            max_rate = physiological_rate_kg_per_day * 1.5
            threshold = max(10.0, max_rate * time_gap_days)
        
        return min(threshold, 20.0)
    
    @staticmethod
    def get_physiological_limit(
        time_delta_hours: float,
        last_weight: float,
        config: Dict,
        unit: Literal['kg', 'percentage'] = 'kg'
    ) -> ThresholdResult:
        """
        Get physiological limits with explicit units.
        Based on framework document recommendations.
        
        Args:
            time_delta_hours: Hours since last measurement
            last_weight: Previous weight in kg
            config: Configuration dictionary
            unit: 'kg' or 'percentage' for return value
            
        Returns:
            ThresholdResult with limit in requested units
        """
        phys_config = config.get('physiological', {})
        
        if not phys_config.get('enable_physiological_limits', True):
            legacy_pct = config.get('max_daily_change_pct', 
                                   config.get('max_daily_change', 0.05))
            limit_kg = legacy_pct * last_weight
            reason = "legacy percentage limit"
            
        elif time_delta_hours < 1:
            percent_limit = phys_config.get('max_change_1h_pct',
                                           phys_config.get('max_change_1h_percent', 0.02))
            absolute_limit = phys_config.get('max_change_1h_kg',
                                            phys_config.get('max_change_1h_absolute', 3.0))
            reason = "hydration/bathroom"
            
        elif time_delta_hours < 6:
            percent_limit = phys_config.get('max_change_6h_pct',
                                           phys_config.get('max_change_6h_percent', 0.025))
            absolute_limit = phys_config.get('max_change_6h_kg',
                                            phys_config.get('max_change_6h_absolute', 4.0))
            reason = "meals+hydration"
            
        elif time_delta_hours <= 24:
            percent_limit = phys_config.get('max_change_24h_pct',
                                           phys_config.get('max_change_24h_percent', 0.035))
            absolute_limit = phys_config.get('max_change_24h_kg',
                                            phys_config.get('max_change_24h_absolute', 5.0))
            reason = "daily fluctuation"
            
        else:
            daily_rate = phys_config.get('max_sustained_kg_per_day',
                                        phys_config.get('max_sustained_daily', 1.5))
            days = time_delta_hours / 24
            absolute_limit = days * daily_rate
            percent_limit = absolute_limit / last_weight
            reason = f"sustained ({daily_rate}kg/day)"
        
        if 'limit_kg' not in locals():
            percentage_based = last_weight * percent_limit
            limit_kg = min(percentage_based, absolute_limit)
            
            if limit_kg == absolute_limit and time_delta_hours <= 24:
                reason += f" (capped at {absolute_limit}kg)"
        
        if time_delta_hours > 24:
            tolerance = phys_config.get('sustained_tolerance', 0.25)
        else:
            tolerance = phys_config.get('limit_tolerance', 0.10)
        
        effective_limit_kg = limit_kg * (1 + tolerance)
        
        metadata = {
            'time_delta_hours': time_delta_hours,
            'last_weight_kg': last_weight,
            'base_limit_kg': limit_kg,
            'tolerance_applied': tolerance,
            'effective_limit_kg': effective_limit_kg,
            'reason': reason
        }
        
        if unit == 'kg':
            return ThresholdResult(effective_limit_kg, 'kg', metadata)
        elif unit == 'percentage':
            percentage = effective_limit_kg / last_weight
            metadata['percentage_limit'] = percentage
            return ThresholdResult(percentage, 'percentage', metadata)
        else:
            raise ValueError(f"Unknown unit: {unit}. Use 'kg' or 'percentage'")
    
    @staticmethod
    def convert_threshold(
        value: float,
        from_unit: str,
        to_unit: str,
        reference_weight: float
    ) -> float:
        """
        Convert threshold between units.
        
        Args:
            value: Threshold value in from_unit
            from_unit: Current unit ('kg' or 'percentage')
            to_unit: Target unit ('kg' or 'percentage')
            reference_weight: Weight in kg for conversion
            
        Returns:
            Converted value in to_unit
        """
        if from_unit == to_unit:
            return value
        
        if from_unit == 'kg' and to_unit == 'percentage':
            return value / reference_weight
        elif from_unit == 'percentage' and to_unit == 'kg':
            return value * reference_weight
        else:
            raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
    
    @staticmethod
    def get_measurement_noise_multiplier(source: str) -> float:
        """
        Get Kalman filter measurement noise multiplier for source.
        
        Args:
            source: Data source identifier
            
        Returns:
            Noise multiplier (1.0 = baseline)
        """
        profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
        return profile['noise_multiplier']
    
    @staticmethod
    def get_source_reliability(source: str) -> str:
        """
        Get reliability classification for source.
        
        Args:
            source: Data source identifier
            
        Returns:
            Reliability level: 'excellent', 'good', 'moderate', 'poor', or 'unknown'
        """
        profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
        return profile['reliability']
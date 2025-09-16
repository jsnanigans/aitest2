"""
Unified validation and data quality module for weight processing.
Combines physiological validation, BMI detection, and data preprocessing.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import numpy as np
import pandas as pd

try:
    from ..constants import (
        SOURCE_PROFILES,
        DEFAULT_PROFILE,
        BMI_LIMITS,
        PHYSIOLOGICAL_LIMITS,
        get_source_reliability,
        get_noise_multiplier,
        categorize_rejection_enhanced,
        get_rejection_severity
    )
except ImportError:
    from src.constants import (
        SOURCE_PROFILES,
        DEFAULT_PROFILE,
        BMI_LIMITS,
        PHYSIOLOGICAL_LIMITS,
        get_source_reliability,
        get_noise_multiplier,
        categorize_rejection_enhanced,
        get_rejection_severity
    )


try:
    from .quality_scorer import QualityScorer, QualityScore, MeasurementHistory
except ImportError:
    from src.processing.quality_scorer import QualityScorer, QualityScore, MeasurementHistory


class PhysiologicalValidator:
    """Validates weight measurements against physiological constraints."""
    
    ABSOLUTE_MIN_WEIGHT = PHYSIOLOGICAL_LIMITS['ABSOLUTE_MIN_WEIGHT']
    ABSOLUTE_MAX_WEIGHT = PHYSIOLOGICAL_LIMITS['ABSOLUTE_MAX_WEIGHT']
    SUSPICIOUS_MIN_WEIGHT = PHYSIOLOGICAL_LIMITS['SUSPICIOUS_MIN_WEIGHT']
    SUSPICIOUS_MAX_WEIGHT = PHYSIOLOGICAL_LIMITS['SUSPICIOUS_MAX_WEIGHT']
    
    MAX_DAILY_CHANGE_KG = PHYSIOLOGICAL_LIMITS['MAX_DAILY_CHANGE_KG']
    MAX_WEEKLY_CHANGE_KG = PHYSIOLOGICAL_LIMITS['MAX_WEEKLY_CHANGE_KG']
    TYPICAL_DAILY_VARIATION_KG = PHYSIOLOGICAL_LIMITS['TYPICAL_DAILY_VARIATION_KG']
    
    @staticmethod
    def validate_absolute_limits(weight: float) -> Tuple[bool, Optional[str]]:
        """Check if weight is within absolute physiological limits."""
        if weight < PhysiologicalValidator.ABSOLUTE_MIN_WEIGHT:
            return False, f"Weight {weight:.1f}kg below absolute minimum {PhysiologicalValidator.ABSOLUTE_MIN_WEIGHT}kg"
        if weight > PhysiologicalValidator.ABSOLUTE_MAX_WEIGHT:
            return False, f"Weight {weight:.1f}kg above absolute maximum {PhysiologicalValidator.ABSOLUTE_MAX_WEIGHT}kg"
        return True, None
    
    @staticmethod
    def check_suspicious_range(weight: float) -> Optional[str]:
        """Check if weight is in suspicious range."""
        if weight < PhysiologicalValidator.SUSPICIOUS_MIN_WEIGHT:
            return f"Weight {weight:.1f}kg suspiciously low"
        if weight > PhysiologicalValidator.SUSPICIOUS_MAX_WEIGHT:
            return f"Weight {weight:.1f}kg suspiciously high"
        return None
    
    @staticmethod
    def validate_rate_of_change(
        current_weight: float,
        previous_weight: float,
        time_diff_hours: float,
        source: str = None
    ) -> Tuple[bool, Optional[str], float]:
        """
        Validate rate of weight change.
        
        Returns:
            (is_valid, rejection_reason, daily_change_rate)
        """
        if time_diff_hours <= 0:
            return True, None, 0.0
        
        weight_diff = abs(current_weight - previous_weight)
        daily_rate = (weight_diff / time_diff_hours) * 24
        
        source_profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE) if source else DEFAULT_PROFILE
        max_daily_change = source_profile.get('max_daily_change_kg', PhysiologicalValidator.MAX_DAILY_CHANGE_KG)
        
        if daily_rate > max_daily_change:
            hours_str = f"{time_diff_hours:.1f}h" if time_diff_hours < 24 else f"{time_diff_hours/24:.1f}d"
            return False, f"Change of {weight_diff:.1f}kg in {hours_str} exceeds max rate", daily_rate
        
        return True, None, daily_rate
    
    @staticmethod
    def check_measurement_pattern(
        measurements: List[Tuple[datetime, float]],
        window_hours: float = 24
    ) -> Dict[str, Any]:
        """
        Analyze measurement patterns for anomalies.
        
        Args:
            measurements: List of (timestamp, weight) tuples
            window_hours: Time window to analyze
            
        Returns:
            Dictionary with pattern analysis results
        """
        if len(measurements) < 2:
            return {'sufficient_data': False}
        
        measurements = sorted(measurements, key=lambda x: x[0])
        
        now = measurements[-1][0]
        window_start = now - timedelta(hours=window_hours)
        recent = [(t, w) for t, w in measurements if t >= window_start]
        
        if len(recent) < 2:
            return {'sufficient_data': False}
        
        weights = [w for _, w in recent]
        mean_weight = np.mean(weights)
        std_weight = np.std(weights)
        
        oscillation_count = 0
        for i in range(1, len(weights) - 1):
            if (weights[i] > weights[i-1] and weights[i] > weights[i+1]) or \
               (weights[i] < weights[i-1] and weights[i] < weights[i+1]):
                oscillation_count += 1
        
        return {
            'sufficient_data': True,
            'mean': mean_weight,
            'std': std_weight,
            'cv': std_weight / mean_weight if mean_weight > 0 else 0,
            'measurement_count': len(recent),
            'oscillation_ratio': oscillation_count / (len(weights) - 2) if len(weights) > 2 else 0,
            'range': max(weights) - min(weights),
            'suspicious_pattern': std_weight > PhysiologicalValidator.TYPICAL_DAILY_VARIATION_KG * 2
        }
    
    @staticmethod
    def calculate_quality_score(
        weight: float,
        source: str,
        previous_weight: Optional[float] = None,
        time_diff_hours: Optional[float] = None,
        recent_weights: Optional[List[float]] = None,
        user_height_m: float = 1.67,
        config: Optional[Dict] = None
    ) -> QualityScore:
        """
        Calculate quality score for a weight measurement.
        
        Args:
            weight: Weight measurement in kg
            source: Data source identifier
            previous_weight: Previous weight measurement
            time_diff_hours: Hours since previous measurement
            recent_weights: List of recent accepted weights
            user_height_m: User's height in meters
            config: Optional configuration overrides
            
        Returns:
            QualityScore object with overall and component scores
        """
        scorer = QualityScorer(config)
        return scorer.calculate_quality_score(
            weight=weight,
            source=source,
            previous_weight=previous_weight,
            time_diff_hours=time_diff_hours,
            recent_weights=recent_weights,
            user_height_m=user_height_m
        )
    
    @staticmethod
    def validate_comprehensive(
        weight: float,
        previous_weight: Optional[float] = None,
        time_diff_hours: Optional[float] = None,
        source: Optional[str] = None,
        recent_measurements: Optional[List[Tuple[datetime, float]]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive validation combining all checks.
        
        Returns:
            Dictionary with validation results and metadata
        """
        result = {
            'valid': True,
            'weight': weight,
            'checks': [],
            'warnings': [],
            'rejection_reason': None
        }
        
        is_valid, reason = PhysiologicalValidator.validate_absolute_limits(weight)
        if not is_valid:
            result['valid'] = False
            result['rejection_reason'] = reason
            return result
        result['checks'].append('absolute_limits')
        
        warning = PhysiologicalValidator.check_suspicious_range(weight)
        if warning:
            result['warnings'].append(warning)
        
        if previous_weight is not None and time_diff_hours is not None:
            is_valid, reason, rate = PhysiologicalValidator.validate_rate_of_change(
                weight, previous_weight, time_diff_hours, source
            )
            if not is_valid:
                result['valid'] = False
                result['rejection_reason'] = reason
                result['daily_change_rate'] = rate
                return result
            result['checks'].append('rate_of_change')
            result['daily_change_rate'] = rate
        
        if recent_measurements:
            pattern_analysis = PhysiologicalValidator.check_measurement_pattern(recent_measurements)
            if pattern_analysis.get('sufficient_data'):
                result['pattern_analysis'] = pattern_analysis
                if pattern_analysis.get('suspicious_pattern'):
                    result['warnings'].append('Suspicious measurement pattern detected')
        
        return result


class BMIValidator:
    """Validates BMI-related measurements and detects BMI vs weight confusion."""
    
    BMI_RANGE = (BMI_LIMITS['IMPOSSIBLE_LOW'], BMI_LIMITS['IMPOSSIBLE_HIGH'])
    WEIGHT_RANGE = (PHYSIOLOGICAL_LIMITS['ABSOLUTE_MIN_WEIGHT'], PHYSIOLOGICAL_LIMITS['ABSOLUTE_MAX_WEIGHT'])
    
    @staticmethod
    def calculate_bmi(weight_kg: float, height_m: float) -> float:
        """Calculate BMI from weight and height."""
        if height_m <= 0:
            return 0
        return weight_kg / (height_m ** 2)
    
    @staticmethod
    def is_likely_bmi(value: float, unit: str = 'kg') -> bool:
        """
        Check if a value is likely BMI rather than weight.
        
        Args:
            value: The numeric value to check
            unit: The stated unit ('kg', 'lb', etc.)
            
        Returns:
            True if value is likely BMI
        """
        if unit.lower() in ['bmi', 'kg/m2', 'kg/m^2']:
            return True
        
        if unit.lower() in ['kg', 'kilogram', 'kilograms']:
            if 15 <= value <= 50:
                return True
        
        return False
    
    @staticmethod
    def convert_bmi_to_weight(bmi: float, height_m: float) -> float:
        """Convert BMI to weight given height."""
        return bmi * (height_m ** 2)
    
    @staticmethod
    def validate_bmi(bmi: float) -> Tuple[bool, Optional[str]]:
        """Validate if BMI is within physiological limits."""
        if bmi < BMI_LIMITS['IMPOSSIBLE_LOW']:
            return False, f"BMI {bmi:.1f} below physiological minimum"
        if bmi > BMI_LIMITS['IMPOSSIBLE_HIGH']:
            return False, f"BMI {bmi:.1f} above physiological maximum"
        return True, None
    
    @staticmethod
    def categorize_bmi(bmi: float) -> str:
        """Categorize BMI into standard categories."""
        if bmi < BMI_LIMITS['UNDERWEIGHT']:
            return 'underweight'
        elif bmi < BMI_LIMITS['OVERWEIGHT']:
            return 'normal'
        elif bmi < BMI_LIMITS['OBESE']:
            return 'overweight'
        else:
            return 'obese'
    
    @staticmethod
    def detect_and_convert(
        value: float,
        unit: str,
        height_m: float,
        source: Optional[str] = None
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Detect if value is BMI and convert to weight if needed.
        
        Args:
            value: The input value
            unit: The stated unit
            height_m: User's height in meters
            source: Data source (some sources are more likely to send BMI)
            
        Returns:
            (weight_kg, was_converted, metadata)
        """
        metadata = {
            'original_value': value,
            'original_unit': unit,
            'height_m': height_m
        }
        
        unit_lower = unit.lower() if unit else 'kg'
        
        if unit_lower in ['lb', 'lbs', 'pound', 'pounds']:
            weight_kg = value * 0.453592
            metadata['conversion'] = f'{value:.1f} lb to {weight_kg:.1f} kg'
            return weight_kg, False, metadata
        
        if unit_lower in ['st', 'stone', 'stones']:
            weight_kg = value * 6.35029
            metadata['conversion'] = f'{value:.1f} st to {weight_kg:.1f} kg'
            return weight_kg, False, metadata
        
        if BMIValidator.is_likely_bmi(value, unit):
            weight_kg = BMIValidator.convert_bmi_to_weight(value, height_m)
            
            if PHYSIOLOGICAL_LIMITS['ABSOLUTE_MIN_WEIGHT'] <= weight_kg <= PHYSIOLOGICAL_LIMITS['ABSOLUTE_MAX_WEIGHT']:
                metadata['detected_as_bmi'] = True
                metadata['conversion'] = f'BMI {value:.1f} to weight {weight_kg:.1f} kg'
                metadata['confidence'] = 'high' if 'connectivehealth' in (source or '').lower() else 'medium'
                return weight_kg, True, metadata
        
        return value, False, metadata
    
    @staticmethod
    def validate_weight_bmi_consistency(
        weight_kg: float,
        height_m: float,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate consistency between weight and implied BMI.
        
        Returns:
            Dictionary with validation results
        """
        bmi = BMIValidator.calculate_bmi(weight_kg, height_m)
        
        result = {
            'weight_kg': weight_kg,
            'height_m': height_m,
            'bmi': bmi,
            'bmi_category': BMIValidator.categorize_bmi(bmi),
            'valid': True,
            'warnings': []
        }
        
        is_valid, reason = BMIValidator.validate_bmi(bmi)
        if not is_valid:
            result['valid'] = False
            result['rejection_reason'] = reason
            return result
        
        if bmi < BMI_LIMITS['SUSPICIOUS_LOW']:
            result['warnings'].append(f'BMI {bmi:.1f} suspiciously low')
        elif bmi > BMI_LIMITS['SUSPICIOUS_HIGH']:
            result['warnings'].append(f'BMI {bmi:.1f} suspiciously high')
        
        if source and 'iglucose' in source.lower():
            result['warnings'].append('High-outlier source detected')
            result['high_risk'] = True
        
        return result
    
    @staticmethod
    def estimate_height_from_weights_and_bmis(
        weight_bmi_pairs: List[Tuple[float, float]]
    ) -> Optional[float]:
        """
        Estimate user height from pairs of weights and BMIs.
        
        Args:
            weight_bmi_pairs: List of (weight_kg, bmi) tuples
            
        Returns:
            Estimated height in meters or None if insufficient data
        """
        if len(weight_bmi_pairs) < 2:
            return None
        
        heights = []
        for weight, bmi in weight_bmi_pairs:
            if bmi > 0:
                height = np.sqrt(weight / bmi)
                if 1.0 <= height <= 2.5:  
                    heights.append(height)
        
        if not heights:
            return None
        
        return np.median(heights)
    
    @staticmethod
    def detect_unit_confusion(
        measurements: List[Tuple[datetime, float, str]],
        height_m: float
    ) -> Dict[str, Any]:
        """
        Detect patterns of unit confusion in measurements.
        
        Args:
            measurements: List of (timestamp, value, unit) tuples
            height_m: User's height in meters
            
        Returns:
            Analysis of potential unit confusion patterns
        """
        if len(measurements) < 3:
            return {'sufficient_data': False}
        
        potential_bmis = []
        potential_weights = []
        
        for _, value, unit in measurements:
            if 15 <= value <= 50:
                potential_bmis.append(value)
            if 40 <= value <= 300:
                potential_weights.append(value)
        
        bmi_ratio = len(potential_bmis) / len(measurements)
        weight_ratio = len(potential_weights) / len(measurements)
        
        result = {
            'sufficient_data': True,
            'total_measurements': len(measurements),
            'potential_bmi_count': len(potential_bmis),
            'potential_weight_count': len(potential_weights),
            'bmi_ratio': bmi_ratio,
            'weight_ratio': weight_ratio
        }
        
        if bmi_ratio > 0.3:
            result['likely_confusion'] = 'frequent_bmi_values'
            result['recommendation'] = 'Check source configuration for unit settings'
        
        if len(potential_bmis) > 0 and len(potential_weights) > 0:
            bmi_mean = np.mean(potential_bmis)
            implied_weight = bmi_mean * (height_m ** 2)
            weight_mean = np.mean(potential_weights)
            
            if abs(implied_weight - weight_mean) < 10:
                result['pattern_detected'] = 'consistent_bmi_weight_relationship'
                result['confidence'] = 'high'
        
        return result


class ThresholdCalculator:
    """Calculate adaptive thresholds for weight validation."""
    
    @staticmethod
    def calculate_adaptive_threshold(
        source: str,
        time_gap_hours: float,
        base_weight: float,
        measurement_noise: float = 0.5
    ) -> float:
        """
        Calculate adaptive threshold based on source and time gap.
        
        Args:
            source: Data source identifier
            time_gap_hours: Hours since last measurement
            base_weight: Reference weight for percentage calculations
            measurement_noise: Base measurement noise in kg
            
        Returns:
            Threshold in kg
        """
        source_profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
        
        base_threshold = source_profile.get('base_threshold_kg', 2.0)
        
        time_factor = 1.0 + (time_gap_hours / 24.0) * 0.5
        time_factor = min(time_factor, 3.0)
        
        weight_factor = 1.0 + (base_weight / 100.0) * 0.1
        
        noise_factor = 1.0 + measurement_noise
        
        threshold = base_threshold * time_factor * weight_factor * noise_factor
        
        max_threshold = source_profile.get('max_threshold_kg', 10.0)
        threshold = min(threshold, max_threshold)
        
        return threshold
    
    @staticmethod
    def calculate_rate_based_threshold(
        recent_changes: List[float],
        time_gap_hours: float,
        source: str
    ) -> float:
        """
        Calculate threshold based on recent rate of change.
        
        Args:
            recent_changes: List of recent weight changes
            time_gap_hours: Hours since last measurement
            source: Data source identifier
            
        Returns:
            Rate-based threshold in kg
        """
        if not recent_changes:
            return ThresholdCalculator.calculate_adaptive_threshold(source, time_gap_hours, 70.0)
        
        mean_change = np.mean(np.abs(recent_changes))
        std_change = np.std(recent_changes) if len(recent_changes) > 1 else mean_change
        
        expected_change = mean_change * (time_gap_hours / 24.0)
        
        threshold = expected_change + 2 * std_change
        
        source_profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
        max_threshold = source_profile.get('max_threshold_kg', 10.0)
        
        return min(threshold, max_threshold)
    
    @staticmethod
    def calculate_confidence_based_threshold(
        confidence: float,
        base_threshold: float,
        source: str
    ) -> float:
        """
        Adjust threshold based on confidence level.
        
        Args:
            confidence: Confidence level (0-1)
            base_threshold: Base threshold in kg
            source: Data source identifier
            
        Returns:
            Adjusted threshold in kg
        """
        source_profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
        
        if confidence > 0.8:
            multiplier = source_profile.get('high_confidence_multiplier', 0.8)
        elif confidence > 0.5:
            multiplier = 1.0
        else:
            multiplier = source_profile.get('low_confidence_multiplier', 1.5)
        
        return base_threshold * multiplier
    
    @staticmethod
    def get_rejection_threshold(
        source: str,
        category: str = 'default'
    ) -> float:
        """
        Get rejection threshold for specific source and category.
        
        Args:
            source: Data source identifier
            category: Rejection category
            
        Returns:
            Rejection threshold in kg
        """
        source_profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
        
        category_thresholds = {
            'spike': source_profile.get('spike_threshold_kg', 5.0),
            'drift': source_profile.get('drift_threshold_kg', 3.0),
            'noise': source_profile.get('noise_threshold_kg', 2.0),
            'default': source_profile.get('base_threshold_kg', 2.0)
        }
        
        return category_thresholds.get(category, category_thresholds['default'])


class DataQualityPreprocessor:
    """Pre-process and clean data before Kalman filtering."""
    
    DEFAULT_HEIGHT_M = PHYSIOLOGICAL_LIMITS['DEFAULT_HEIGHT_M']
    
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
            unit: Unit of the weight measurement
        
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
        
        user_height = DataQualityPreprocessor.get_user_height(user_id) if user_id else DataQualityPreprocessor.DEFAULT_HEIGHT_M
        
        unit_lower = unit.lower() if unit else 'kg'
        if unit_lower in ['lb', 'lbs', 'pound', 'pounds']:
            weight_kg = weight * 0.453592
            metadata['corrections'].append(f'Converted {weight:.1f} {unit} to {weight_kg:.1f} kg')
            weight = weight_kg
        elif unit_lower in ['st', 'stone', 'stones']:
            weight_kg = weight * 6.35029
            metadata['corrections'].append(f'Converted {weight:.1f} {unit} to {weight_kg:.1f} kg')
            weight = weight_kg
        elif unit_lower not in ['kg', 'kilogram', 'kilograms']:
            metadata['warnings'].append(f'Unknown unit: {unit}')
        
        if unit_lower in ['kg', 'kilogram', 'kilograms'] and 15 <= weight <= 50:
            implied_weight = weight * (user_height ** 2)
            
            if 40 <= implied_weight <= 200:
                metadata['warnings'].append(
                    f'Value {weight:.1f} likely BMI (implies {implied_weight:.1f}kg weight for height {user_height:.2f}m)'
                )
                metadata['corrections'].append(f'Converted BMI {weight:.1f} to weight {implied_weight:.1f}kg')
                weight = implied_weight
            elif 30 <= implied_weight <= 250:
                if 'connectivehealth' in source.lower():
                    metadata['warnings'].append(f'Value {weight:.1f} appears to be BMI from {source}')
                    metadata['corrections'].append(f'Converted BMI {weight:.1f} to weight {implied_weight:.1f}kg')
                    weight = implied_weight
                else:
                    metadata['warnings'].append(f'Value {weight:.1f} might be BMI')
        
        implied_bmi = weight / (user_height ** 2)
        
        if implied_bmi < BMI_LIMITS['IMPOSSIBLE_LOW']:
            metadata['warnings'].append(
                f'Implied BMI {implied_bmi:.1f} physiologically impossible (height: {user_height:.2f}m)'
            )
            metadata['rejected'] = f'BMI {implied_bmi:.1f} outside physiological limits'
            return None, metadata
        
        if implied_bmi > BMI_LIMITS['IMPOSSIBLE_HIGH']:
            metadata['rejected'] = f'Implied BMI {implied_bmi:.1f} physiologically impossible'
            return None, metadata
        
        if implied_bmi < BMI_LIMITS['SUSPICIOUS_LOW']:
            metadata['warnings'].append(f'Implied BMI {implied_bmi:.1f} suspiciously low')
        
        if implied_bmi > BMI_LIMITS['SUSPICIOUS_HIGH']:
            metadata['warnings'].append(f'Implied BMI {implied_bmi:.1f} suspiciously high')
        
        metadata['checks_passed'].append('physiological_limits')
        metadata['implied_bmi'] = round(implied_bmi, 1)
        metadata['user_height_m'] = round(user_height, 2)
        
        if implied_bmi < BMI_LIMITS['UNDERWEIGHT']:
            metadata['bmi_category'] = 'underweight'
        elif implied_bmi < BMI_LIMITS['OVERWEIGHT']:
            metadata['bmi_category'] = 'normal'
        elif implied_bmi < BMI_LIMITS['OBESE']:
            metadata['bmi_category'] = 'overweight'
        else:
            metadata['bmi_category'] = 'obese'
        
        if 'iglucose' in source.lower():
            metadata['warnings'].append('High-outlier source - increased scrutiny')
            metadata['high_risk'] = True
        
        return weight, metadata
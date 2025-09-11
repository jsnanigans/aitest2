"""
Unified Threshold Calculator for Weight Processing System.
Handles all threshold calculations with explicit unit specifications.
"""

from typing import Dict, Tuple, Optional, Literal
import numpy as np


class ThresholdResult:
    """Result from threshold calculation with explicit units."""
    
    def __init__(self, value: float, unit: str, metadata: Optional[Dict] = None):
        self.value = value
        self.unit = unit
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'value': self.value,
            'unit': self.unit,
            'metadata': self.metadata
        }


class ThresholdCalculator:
    """
    Unified threshold calculator with explicit unit handling.
    All methods are static to maintain stateless architecture.
    """
    
    # Source reliability profiles based on empirical data (outliers per 1000)
    SOURCE_PROFILES = {
        'care-team-upload': {
            'outlier_rate': 3.6,
            'reliability': 'excellent',
            'noise_multiplier': 0.5
        },
        'patient-upload': {
            'outlier_rate': 13.0,
            'reliability': 'excellent',
            'noise_multiplier': 0.7
        },
        'internal-questionnaire': {
            'outlier_rate': 14.0,
            'reliability': 'good',
            'noise_multiplier': 0.8
        },
        'patient-device': {
            'outlier_rate': 20.7,
            'reliability': 'good',
            'noise_multiplier': 1.0
        },
        'https://connectivehealth.io': {
            'outlier_rate': 35.8,
            'reliability': 'moderate',
            'noise_multiplier': 1.5
        },
        'https://api.iglucose.com': {
            'outlier_rate': 151.4,
            'reliability': 'poor',
            'noise_multiplier': 3.0
        }
    }
    
    # Default profile for unknown sources (median behavior)
    DEFAULT_PROFILE = {
        'outlier_rate': 20.0,
        'reliability': 'unknown',
        'noise_multiplier': 1.0
    }
    
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
        # Get source profile
        profile = ThresholdCalculator.SOURCE_PROFILES.get(
            source, 
            ThresholdCalculator.DEFAULT_PROFILE
        )
        
        # Calculate absolute threshold in kg
        absolute_kg = ThresholdCalculator._calculate_adaptive_threshold_kg(
            profile['outlier_rate'],
            time_gap_days
        )
        
        # Prepare metadata
        metadata = {
            'source': source,
            'source_reliability': profile['reliability'],
            'outlier_rate_per_1000': profile['outlier_rate'],
            'time_gap_days': time_gap_days,
            'weight_reference_kg': current_weight,
            'absolute_threshold_kg': absolute_kg
        }
        
        if unit == 'percentage':
            # Convert to percentage with bounds
            percentage = absolute_kg / current_weight
            
            # Apply reasonable bounds for percentage mode
            MIN_PERCENTAGE = 0.03  # 3% minimum even for good sources
            MAX_PERCENTAGE = 0.30  # 30% maximum even for bad sources
            
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
        # Base physiological limit: 2kg/week is extreme but possible
        physiological_rate_kg_per_day = 2.0 / 7.0
        
        if outlier_rate > 100:  # Very unreliable (e.g., iGlucose)
            # Much stricter - only allow 1kg/week
            max_rate = 1.0 / 7.0
            threshold = max(3.0, max_rate * time_gap_days)
        elif outlier_rate > 30:  # Moderate reliability
            # Standard physiological limit
            max_rate = physiological_rate_kg_per_day
            threshold = max(5.0, max_rate * time_gap_days)
        else:  # Excellent reliability
            # More lenient - trust the source more
            max_rate = physiological_rate_kg_per_day * 1.5
            threshold = max(10.0, max_rate * time_gap_days)
        
        # Cap at reasonable maximum
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
        
        # Determine base limits
        if not phys_config.get('enable_physiological_limits', True):
            # Legacy mode - use simple percentage
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
            # Sustained changes
            daily_rate = phys_config.get('max_sustained_kg_per_day',
                                        phys_config.get('max_sustained_daily', 1.5))
            days = time_delta_hours / 24
            absolute_limit = days * daily_rate
            percent_limit = absolute_limit / last_weight
            reason = f"sustained ({daily_rate}kg/day)"
        
        # Calculate final limit in kg
        if 'limit_kg' not in locals():
            percentage_based = last_weight * percent_limit
            limit_kg = min(percentage_based, absolute_limit)
            
            if limit_kg == absolute_limit and time_delta_hours <= 24:
                reason += f" (capped at {absolute_limit}kg)"
        
        # Apply tolerance
        if time_delta_hours > 24:
            tolerance = phys_config.get('sustained_tolerance', 0.25)  # 25% for sustained
        else:
            tolerance = phys_config.get('limit_tolerance', 0.10)  # 10% for short-term
        
        effective_limit_kg = limit_kg * (1 + tolerance)
        
        # Prepare metadata
        metadata = {
            'time_delta_hours': time_delta_hours,
            'last_weight_kg': last_weight,
            'base_limit_kg': limit_kg,
            'tolerance_applied': tolerance,
            'effective_limit_kg': effective_limit_kg,
            'reason': reason
        }
        
        # Return in requested units
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
        profile = ThresholdCalculator.SOURCE_PROFILES.get(
            source,
            ThresholdCalculator.DEFAULT_PROFILE
        )
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
        profile = ThresholdCalculator.SOURCE_PROFILES.get(
            source,
            ThresholdCalculator.DEFAULT_PROFILE
        )
        return profile['reliability']
#!/usr/bin/env python3
"""
BMI-based validation and reset detection for physiologically impossible weight changes.
"""

from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np


class BMIValidator:
    """
    Validates weight measurements using BMI thresholds and percentage changes.
    Detects when Kalman filter should be reset due to unrealistic changes.
    """
    
    BMI_CRITICAL_LOW = 15.0
    BMI_SEVERE_LOW = 16.0
    BMI_UNDERWEIGHT = 18.5
    BMI_OVERWEIGHT = 25.0
    BMI_OBESE = 30.0
    BMI_SEVERE_OBESE = 35.0
    BMI_MORBID_OBESE = 40.0
    BMI_CRITICAL_HIGH = 50.0
    
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
        
        if current_weight < 30:
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
                if current_bmi < BMIValidator.BMI_CRITICAL_LOW:
                    return True, f"Critical BMI < {BMIValidator.BMI_CRITICAL_LOW}: {current_bmi:.1f}"
                
                if current_bmi > BMIValidator.BMI_CRITICAL_HIGH:
                    return True, f"Critical BMI > {BMIValidator.BMI_CRITICAL_HIGH}: {current_bmi:.1f}"
                
                if current_bmi < BMIValidator.BMI_SEVERE_LOW and abs(pct_change) > 20:
                    return True, f"Severe underweight (BMI {current_bmi:.1f}) with {pct_change:+.1f}% change"
                
                if current_bmi > BMIValidator.BMI_MORBID_OBESE and abs(pct_change) > 20:
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
                if bmi < BMIValidator.BMI_SEVERE_LOW or bmi > BMIValidator.BMI_MORBID_OBESE:
                    bmi_factor = 0.5
                elif bmi < BMIValidator.BMI_UNDERWEIGHT or bmi > BMIValidator.BMI_SEVERE_OBESE:
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
        
        if current_weight < 30:
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
            max_sustained_rate = 1.5
            max_change = days * max_sustained_rate
            if abs(weight_change) > max_change:
                return f"Change {weight_change:+.1f}kg exceeds sustained rate of {max_sustained_rate}kg/day over {days:.0f} days"
        
        return None
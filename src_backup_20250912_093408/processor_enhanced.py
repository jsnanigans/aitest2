"""
Enhanced weight processor with BMI validation and deferred reset logic.
Prevents acceptance of physiologically impossible values during reset.
"""

from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List, Any
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from pykalman import KalmanFilter

try:
    from src.processor_database import ProcessorStateDB, get_state_db
    from src.threshold_calculator import ThresholdCalculator
    from src.bmi_validator import BMIValidator
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.processor_database import ProcessorStateDB, get_state_db
    from src.threshold_calculator import ThresholdCalculator
    from src.bmi_validator import BMIValidator


class EnhancedWeightProcessor:
    """
    Enhanced processor with BMI validation and intelligent reset handling.
    Key improvements:
    1. BMI validation on reset path
    2. Deferred reset with end-of-day reprocessing
    3. Selection of closest value to pre-gap baseline
    """
    
    QUESTIONNAIRE_SOURCES = {
        'internal-questionnaire',
        'initial-questionnaire', 
        'care-team-upload',
        'questionnaire'
    }
    
    @staticmethod
    def process_weight(
        user_id: str,
        weight: float,
        timestamp: datetime,
        source: str,
        processing_config: Dict,
        kalman_config: Dict,
        db=None,
        height_m: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Process weight with enhanced validation.
        
        Args:
            height_m: User's height in meters for BMI calculation
        """
        if db is None:
            db = get_state_db()
        
        state = db.get_state(user_id)
        if state is None:
            state = db.create_initial_state()
        
        result, updated_state = EnhancedWeightProcessor._process_weight_internal(
            weight, timestamp, source, state, processing_config, kalman_config, height_m
        )
        
        if updated_state:
            updated_state['last_source'] = source
            if height_m:
                updated_state['height_m'] = height_m
            db.save_state(user_id, updated_state)
        
        return result
    
    @staticmethod
    def _process_weight_internal(
        weight: float,
        timestamp: datetime,
        source: str,
        state: Dict[str, Any],
        processing_config: dict,
        kalman_config: dict,
        height_m: Optional[float] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Internal processing with BMI validation and deferred reset.
        """
        new_state = state.copy()
        
        if not height_m and state.get('height_m'):
            height_m = state['height_m']
        
        if not state.get('kalman_params'):
            if not EnhancedWeightProcessor._is_weight_valid(
                weight, None, height_m, processing_config
            ):
                return {
                    "timestamp": timestamp,
                    "raw_weight": weight,
                    "accepted": False,
                    "reason": "Initial weight outside valid range",
                    "source": source,
                }, None
            
            new_state = EnhancedWeightProcessor._initialize_kalman_immediate(
                weight, timestamp, kalman_config
            )
            new_state = EnhancedWeightProcessor._update_kalman_state(
                new_state, weight, timestamp, source, processing_config
            )
            result = EnhancedWeightProcessor._create_result(
                new_state, weight, timestamp, source, True
            )
            return result, new_state
        
        time_delta_days = 1.0
        should_defer_reset = False
        
        if new_state.get('last_timestamp'):
            last_timestamp = new_state['last_timestamp']
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp)
            delta = (timestamp - last_timestamp).total_seconds() / 86400.0
            time_delta_days = max(0.1, min(30.0, delta))
            
            reset_gap_days = kalman_config.get("reset_gap_days", 30)
            
            if state.get('last_source') in EnhancedWeightProcessor.QUESTIONNAIRE_SOURCES:
                reset_gap_days = kalman_config.get("questionnaire_reset_days", 10)
            
            if delta > reset_gap_days:
                last_weight = EnhancedWeightProcessor._get_last_weight(new_state)
                
                is_valid, validation_reason = EnhancedWeightProcessor._validate_reset_weight(
                    weight, last_weight, delta, height_m, source, processing_config
                )
                
                if not is_valid:
                    return {
                        "timestamp": timestamp,
                        "raw_weight": weight,
                        "accepted": False,
                        "reason": f"Reset validation failed: {validation_reason}",
                        "source": source,
                        "gap_days": delta,
                        "requires_reset": True,
                        "deferred_for_reprocessing": True
                    }, None
                
                if not new_state.get('pending_reset'):
                    new_state['pending_reset'] = {
                        'triggered_at': timestamp.isoformat(),
                        'gap_days': delta,
                        'pre_gap_weight': last_weight,
                        'measurements': []
                    }
                
                new_state['pending_reset']['measurements'].append({
                    'timestamp': timestamp.isoformat(),
                    'weight': weight,
                    'source': source,
                    'distance_from_baseline': abs(weight - last_weight)
                })
                
                should_defer_reset = True
        
        if should_defer_reset:
            return {
                "timestamp": timestamp,
                "raw_weight": weight,
                "accepted": False,
                "reason": "Deferred for end-of-day reset processing",
                "source": source,
                "pending_reset": True,
                "gap_days": delta
            }, new_state
        
        is_valid, rejection_reason = EnhancedWeightProcessor._validate_weight(
            weight, processing_config, state, timestamp, height_m
        )
        
        if not is_valid:
            return {
                "timestamp": timestamp,
                "raw_weight": weight,
                "accepted": False,
                "reason": rejection_reason or "Validation failed",
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
                if height_m:
                    should_reset, reset_reason = BMIValidator.should_reset_kalman(
                        weight, predicted_weight, time_delta_days * 24, height_m, source
                    )
                    if should_reset:
                        return {
                            "timestamp": timestamp,
                            "raw_weight": weight,
                            "accepted": False,
                            "reason": f"BMI validation triggered reset: {reset_reason}",
                            "source": source,
                            "requires_reset": True
                        }, None
                
                pseudo_normalized_innovation = (deviation / extreme_threshold) * 3.0
                confidence = EnhancedWeightProcessor._calculate_confidence(
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
        
        new_state = EnhancedWeightProcessor._update_kalman_state(
            new_state, weight, timestamp, source, processing_config
        )
        
        result = EnhancedWeightProcessor._create_result(
            new_state, weight, timestamp, source, True
        )
        
        return result, new_state
    
    @staticmethod
    def _validate_reset_weight(
        weight: float,
        last_weight: Optional[float],
        gap_days: float,
        height_m: Optional[float],
        source: str,
        processing_config: Dict
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate weight during reset to prevent impossible values.
        """
        min_weight = processing_config.get("min_weight", 30.0)
        max_weight = processing_config.get("max_weight", 300.0)
        
        if weight < min_weight or weight > max_weight:
            return False, f"Weight {weight:.1f}kg outside bounds [{min_weight}, {max_weight}]"
        
        if height_m and height_m > 0:
            bmi = BMIValidator.calculate_bmi(weight, height_m)
            if bmi:
                if bmi < BMIValidator.BMI_CRITICAL_LOW:
                    return False, f"BMI {bmi:.1f} below critical threshold {BMIValidator.BMI_CRITICAL_LOW}"
                if bmi > BMIValidator.BMI_CRITICAL_HIGH:
                    return False, f"BMI {bmi:.1f} above critical threshold {BMIValidator.BMI_CRITICAL_HIGH}"
        
        if last_weight and last_weight > 0:
            pct_change = abs(weight - last_weight) / last_weight
            
            if pct_change > 0.5:
                if height_m:
                    should_reset, reason = BMIValidator.should_reset_kalman(
                        weight, last_weight, gap_days * 24, height_m, source
                    )
                    if should_reset:
                        return False, reason
                else:
                    return False, f"Change of {pct_change:.1%} exceeds 50% threshold"
        
        return True, None
    
    @staticmethod
    def _is_weight_valid(
        weight: float,
        last_weight: Optional[float],
        height_m: Optional[float],
        processing_config: Dict
    ) -> bool:
        """
        Basic weight validation.
        """
        min_weight = processing_config.get("min_weight", 30.0)
        max_weight = processing_config.get("max_weight", 300.0)
        
        if weight < min_weight or weight > max_weight:
            return False
        
        if height_m and height_m > 0:
            bmi = BMIValidator.calculate_bmi(weight, height_m)
            if bmi and (bmi < 13 or bmi > 55):
                return False
        
        return True
    
    @staticmethod
    def _get_last_weight(state: Dict) -> Optional[float]:
        """
        Extract last weight from state.
        """
        if state.get('last_state') is not None:
            last_state = state['last_state']
            if len(last_state.shape) > 1:
                return float(last_state[-1][0])
            else:
                return float(last_state[0])
        return None
    
    @staticmethod
    def process_pending_resets(
        user_id: str,
        db=None
    ) -> Dict[str, Any]:
        """
        Process pending resets at end of day.
        Selects the measurement closest to pre-gap baseline.
        """
        if db is None:
            db = get_state_db()
        
        state = db.get_state(user_id)
        if not state or not state.get('pending_reset'):
            return {"status": "no_pending_resets"}
        
        pending = state['pending_reset']
        measurements = pending['measurements']
        pre_gap_weight = pending['pre_gap_weight']
        
        if not measurements:
            return {"status": "no_measurements_to_process"}
        
        best_measurement = min(
            measurements,
            key=lambda m: m['distance_from_baseline']
        )
        
        selected_weight = best_measurement['weight']
        selected_timestamp = datetime.fromisoformat(best_measurement['timestamp'])
        selected_source = best_measurement['source']
        
        from src.processor import WeightProcessor
        kalman_config = state.get('kalman_params', {})
        
        new_state = WeightProcessor._initialize_kalman_immediate(
            selected_weight, selected_timestamp, kalman_config
        )
        new_state['last_source'] = selected_source
        new_state['reset_reason'] = f"Selected closest to pre-gap baseline ({pre_gap_weight:.1f}kg)"
        new_state['height_m'] = state.get('height_m')
        
        del state['pending_reset']
        
        for key, value in new_state.items():
            state[key] = value
        
        db.save_state(user_id, state)
        
        return {
            "status": "reset_completed",
            "selected_weight": selected_weight,
            "pre_gap_weight": pre_gap_weight,
            "measurements_considered": len(measurements),
            "timestamp": selected_timestamp.isoformat()
        }


from src.processor import WeightProcessor

for attr_name in dir(WeightProcessor):
    if attr_name.startswith('_') and attr_name not in [
        '_process_weight_internal', 
        '_validate_reset_weight',
        '_is_weight_valid',
        '_get_last_weight',
        'process_pending_resets'
    ]:
        attr = getattr(WeightProcessor, attr_name)
        if callable(attr) and not hasattr(EnhancedWeightProcessor, attr_name):
            setattr(EnhancedWeightProcessor, attr_name, attr)
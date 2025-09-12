"""
Enhanced weight processor with dynamic reset capability.
Implements shorter reset gaps after questionnaire data.
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
    Stateless weight processor with dynamic reset capability.
    """
    
    # Questionnaire sources that trigger shorter reset gaps
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
    ) -> Optional[Dict]:
        """
        Process a single weight measurement with dynamic reset support.
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
            # Track last source for dynamic reset
            updated_state['last_source'] = source
            db.save_state(user_id, updated_state)
        
        return result

    @staticmethod
    def _get_dynamic_reset_gap(
        state: Dict[str, Any],
        kalman_config: dict
    ) -> float:
        """
        Determine reset gap based on last measurement source.
        
        Returns:
            Reset gap in days (10 for questionnaire, 30 for others)
        """
        # Check if dynamic reset is enabled
        dynamic_config = kalman_config.get('dynamic_reset', {})
        if not dynamic_config.get('enabled', False):
            return kalman_config.get("reset_gap_days", 30)
        
        # Check last source
        last_source = state.get('last_source', '')
        
        # Use shorter gap if last measurement was from questionnaire
        if last_source in WeightProcessor.QUESTIONNAIRE_SOURCES:
            return dynamic_config.get('questionnaire_gap_days', 10)
        
        # Default to standard gap
        return kalman_config.get("reset_gap_days", 30)

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
        Internal processing logic with dynamic reset support.
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
        reset_reason = None
        
        if new_state.get('last_timestamp'):
            last_timestamp = new_state['last_timestamp']
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp)
            delta = (timestamp - last_timestamp).total_seconds() / 86400.0
            time_delta_days = max(0.1, min(30.0, delta))
            
            # Get dynamic reset gap based on last source
            reset_gap_days = WeightProcessor._get_dynamic_reset_gap(
                new_state, kalman_config
            )
            
            if delta > reset_gap_days:
                should_reset = True
                reset_reason = f"Gap ({delta:.0f}d) > threshold ({reset_gap_days}d)"
                
                # Add info about dynamic reset if applicable
                if new_state.get('last_source') in WeightProcessor.QUESTIONNAIRE_SOURCES:
                    reset_reason += f" [questionnaire-triggered]"
                
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
                result['reset_reason'] = reset_reason
                return result, new_state
        
        # Rest of the processing logic remains the same...
        # (Copy the rest from original processor.py)
        
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

# Copy all other methods from original processor.py...
# For brevity, I'll just import them
from src.processor import WeightProcessor as OriginalProcessor

# Copy all static methods
for attr_name in dir(OriginalProcessor):
    if attr_name.startswith('_') and attr_name not in ['_process_weight_internal', '_get_dynamic_reset_gap']:
        attr = getattr(OriginalProcessor, attr_name)
        if callable(attr):
            setattr(WeightProcessor, attr_name, attr)

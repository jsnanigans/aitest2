"""
Enhanced validation gate with multiple strategies to handle data quality issues.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class EnhancedValidationGate:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        self.max_deviation_pct = self.config.get('max_deviation_pct', 0.15)
        self.same_day_threshold_kg = self.config.get('same_day_threshold_kg', 5.0)
        self.rapid_change_threshold_pct = self.config.get('rapid_change_threshold_pct', 0.15)
        self.rapid_change_hours = self.config.get('rapid_change_hours', 24)
        self.outlier_z_score = self.config.get('outlier_z_score', 2.5)
        self.min_readings_for_stats = self.config.get('min_readings_for_stats', 5)
        self.future_date_tolerance_days = self.config.get('future_date_tolerance_days', 1)
        self.duplicate_threshold_kg = self.config.get('duplicate_threshold_kg', 0.5)
        
        self.user_contexts = {}
        self.validation_stats = defaultdict(lambda: {'passed': 0, 'rejected': 0, 'reasons': defaultdict(int)})
    
    def validate_reading(self, user_id: str, reading: Dict) -> Tuple[bool, str]:
        """
        Validate a single reading with multiple checks.
        Returns (is_valid, reason_if_invalid)
        """
        
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = UserValidationContext()
        
        context = self.user_contexts[user_id]
        
        reading_date = reading.get('date') or reading.get('date_str')
        if isinstance(reading_date, str):
            try:
                reading_date = datetime.fromisoformat(reading_date.replace(' ', 'T'))
            except:
                # If date parsing fails, allow the reading (don't reject on parse errors)
                return True, "Valid"
        elif reading_date is None:
            # No date provided, allow the reading
            return True, "Valid"
        
        weight = reading.get('weight', 0)
        
        # Reset today_readings if we've moved to a new day
        if context.current_day != reading_date.date():
            context.today_readings = []
            context.current_day = reading_date.date()
        
        if not self._validate_future_date(reading_date):
            self._record_rejection(user_id, 'future_date')
            return False, "Future date"
        
        # Only check extreme deviation if we have recent readings to compare against
        # Use recent average instead of stale baseline
        if len(context.recent_weights) >= 3:
            recent_median = np.median(context.recent_weights[-10:])  # Use last 10 readings
            if not self._validate_extreme_deviation(weight, recent_median):
                self._record_rejection(user_id, 'extreme_deviation')
                return False, f"Extreme deviation from recent trend"
        
        if context.last_reading:
            # Check for long gaps and reset context if needed
            time_since_last = (reading_date - context.last_reading['date']).days
            # Use same gap threshold as baseline establishment (14 days by default)
            gap_threshold = self.config.get('gap_threshold_days', 14)
            if time_since_last > gap_threshold:
                # After a long gap, reset recent weights to avoid false positives
                logger.debug(f"Long gap detected ({time_since_last} days > {gap_threshold}), resetting context for {user_id}")
                context.recent_weights = []
                context.baseline_weight = None  # Baseline may be stale after long gap
                # Add the current weight as the first reading after gap
                context.recent_weights.append(weight)
            elif not self._validate_rapid_change(weight, reading_date, 
                                              context.last_reading['weight'], 
                                              context.last_reading['date']):
                self._record_rejection(user_id, 'rapid_change')
                return False, "Rapid weight change"
        
        if context.today_readings:
            if not self._validate_same_day_consistency(weight, context.today_readings):
                self._record_rejection(user_id, 'same_day_inconsistency')
                return False, "Inconsistent with same-day readings"
        
        if len(context.recent_weights) >= self.min_readings_for_stats:
            if not self._validate_statistical_outlier(weight, context.recent_weights):
                self._record_rejection(user_id, 'statistical_outlier')
                return False, "Statistical outlier"
        
        context.add_reading(reading_date, weight)
        self._record_pass(user_id)
        return True, "Valid"
    
    def _validate_future_date(self, date: datetime) -> bool:
        """Reject readings with future dates"""
        max_allowed = datetime.now() + timedelta(days=self.future_date_tolerance_days)
        return date <= max_allowed
    
    def _validate_extreme_deviation(self, weight: float, baseline: float) -> bool:
        """Reject readings that deviate too much from baseline"""
        if baseline <= 0:
            return True
        deviation = abs(weight - baseline) / baseline
        is_valid = deviation <= self.max_deviation_pct
        if not is_valid:
            logger.debug(f"Extreme deviation: {weight:.1f}kg vs baseline {baseline:.1f}kg ({deviation*100:.1f}% > {self.max_deviation_pct*100:.1f}%)")
        return is_valid
    
    def _validate_rapid_change(self, weight: float, date: datetime, 
                               last_weight: float, last_date: datetime) -> bool:
        """Reject rapid weight changes"""
        time_diff_hours = (date - last_date).total_seconds() / 3600
        if time_diff_hours > self.rapid_change_hours:
            return True
        
        if last_weight <= 0:
            return True
            
        change_pct = abs(weight - last_weight) / last_weight
        return change_pct <= self.rapid_change_threshold_pct
    
    def _validate_same_day_consistency(self, weight: float, today_weights: List[float]) -> bool:
        """Check consistency with other readings from same day"""
        if not today_weights:
            return True
        
        max_diff = max(max(today_weights) - weight, weight - min(today_weights))
        return max_diff <= self.same_day_threshold_kg
    
    def _validate_statistical_outlier(self, weight: float, recent_weights: List[float]) -> bool:
        """Detect statistical outliers using z-score"""
        if len(recent_weights) < self.min_readings_for_stats:
            return True
        
        mean = np.mean(recent_weights)
        std = np.std(recent_weights)
        
        if std == 0:
            return True
        
        z_score = abs((weight - mean) / std)
        return z_score <= self.outlier_z_score
    
    def should_deduplicate(self, user_id: str, reading: Dict) -> bool:
        """Check if this reading should be skipped as duplicate"""
        if user_id not in self.user_contexts:
            return False
        
        context = self.user_contexts[user_id]
        reading_date = reading.get('date') or reading.get('date_str')
        if isinstance(reading_date, str):
            try:
                reading_date = datetime.fromisoformat(reading_date.replace(' ', 'T'))
            except:
                return False  # Don't deduplicate if we can't parse the date
        elif reading_date is None:
            return False
        
        # Reset today_readings if we've moved to a new day
        if context.current_day != reading_date.date():
            context.today_readings = []
            context.current_day = reading_date.date()
        
        weight = reading.get('weight', 0)
        
        for existing_weight in context.today_readings:
            if abs(weight - existing_weight) < self.duplicate_threshold_kg:
                self._record_rejection(user_id, 'duplicate')
                return True
        
        return False
    
    def _record_rejection(self, user_id: str, reason: str):
        """Record validation rejection statistics"""
        self.validation_stats[user_id]['rejected'] += 1
        self.validation_stats[user_id]['reasons'][reason] += 1
        logger.debug(f"Validation rejected for {user_id}: {reason}")
    
    def _record_pass(self, user_id: str):
        """Record validation pass statistics"""
        self.validation_stats[user_id]['passed'] += 1
        logger.debug(f"Validation passed for {user_id}")
    
    def get_stats(self, user_id: str = None) -> Dict:
        """Get validation statistics"""
        if user_id:
            return dict(self.validation_stats[user_id])
        return dict(self.validation_stats)
    
    def reset_user_context(self, user_id: str):
        """Reset context for a user (e.g., after gap detection)"""
        if user_id in self.user_contexts:
            self.user_contexts[user_id].reset()


class UserValidationContext:
    """Maintains validation context for a single user"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.baseline_weight = None
        self.last_reading = None
        self.today_readings = []
        self.current_day = None
        self.recent_weights = []
        self.all_weights = []
    
    def add_reading(self, date: datetime, weight: float):
        """Add a validated reading to context"""
        
        # Day change already handled in validate_reading()
        # Just append to today_readings
        self.today_readings.append(weight)
        self.recent_weights.append(weight)
        self.all_weights.append(weight)
        
        if len(self.recent_weights) > self.window_size:
            self.recent_weights.pop(0)
        
        # Update baseline periodically with recent data (adaptive baseline)
        if len(self.recent_weights) >= 5:
            # Use recent median as adaptive baseline
            self.baseline_weight = np.median(self.recent_weights[-10:])
            logger.debug(f"Baseline updated: {self.baseline_weight:.1f}kg from recent {min(10, len(self.recent_weights))} readings")
        
        self.last_reading = {'date': date, 'weight': weight}
    
    def reset(self):
        """Reset context (e.g., after detecting different person)"""
        self.baseline_weight = None
        self.last_reading = None
        self.today_readings = []
        self.current_day = None
        self.recent_weights = []
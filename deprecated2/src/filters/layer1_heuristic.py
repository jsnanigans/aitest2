"""
Layer 1: Stateless Heuristic Filtering for Real-Time Processing
Fast, physiologically-informed gates that protect Kalman from impossible values
Designed for single-measurement processing without historical context
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
import logging

from src.core.types import WeightMeasurement, OutlierType

logger = logging.getLogger(__name__)


class PhysiologicalFilter:
    """Simple physiological plausibility checks."""

    def __init__(self, min_weight: float = 30.0, max_weight: float = 400.0):
        self.min_weight = min_weight
        self.max_weight = max_weight

    def validate(self, measurement: WeightMeasurement) -> Tuple[bool, Optional[OutlierType]]:
        """Check if weight is physiologically plausible."""
        if measurement.weight < self.min_weight:
            logger.debug(f"Weight {measurement.weight}kg below minimum {self.min_weight}kg")
            return False, OutlierType.PHYSIOLOGICAL_IMPOSSIBLE
        if measurement.weight > self.max_weight:
            logger.debug(f"Weight {measurement.weight}kg above maximum {self.max_weight}kg")
            return False, OutlierType.PHYSIOLOGICAL_IMPOSSIBLE
        return True, None


class StatelessRateOfChangeFilter:
    """
    Stateless rate-of-change validation using previous state from Kalman.
    Suitable for real-time processing where we only have the last known good state.
    """

    def __init__(self,
                 max_daily_change_percent: float = 3.0,
                 medical_mode_percent: float = 5.0):
        self.max_daily_change_percent = max_daily_change_percent
        self.medical_mode_percent = medical_mode_percent

    def validate(self,
                measurement: WeightMeasurement,
                last_known_state: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[OutlierType], Dict]:
        """
        Validate rate of change against last known state.
        For real-time: last_known_state comes from database/cache.
        """
        metadata = {}

        if last_known_state is None:
            return True, None, metadata

        last_weight = last_known_state.get('weight')
        last_timestamp = last_known_state.get('timestamp')

        if last_weight is None or last_timestamp is None:
            return True, None, metadata

        if isinstance(last_timestamp, str):
            last_timestamp = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))

        time_delta_days = (measurement.timestamp - last_timestamp).total_seconds() / 86400.0

        if time_delta_days <= 0:
            metadata['warning'] = 'non_chronological'
            return True, None, metadata

        # For same-day measurements (less than 1 day), be more lenient
        if time_delta_days < 1.0:
            # Allow up to 2kg change for same-day measurements
            max_change = 2.0
            metadata['same_day'] = True
        else:
            # For longer gaps, scale the allowed change
            is_medical_mode = last_known_state.get('medical_intervention_mode', False)
            max_percent = self.medical_mode_percent if is_medical_mode else self.max_daily_change_percent

            # Cap the time delta at 30 days to avoid being too permissive for long gaps
            # This prevents accepting huge changes just because months have passed
            effective_days = min(time_delta_days, 30.0)
            max_change = last_weight * (max_percent / 100.0) * effective_days

        actual_change = abs(measurement.weight - last_weight)
        change_percent = (actual_change / last_weight) * 100

        metadata['time_delta_days'] = time_delta_days
        metadata['actual_change_kg'] = actual_change
        metadata['change_percent'] = change_percent
        metadata['max_allowed_change_kg'] = max_change

        # Check for extreme percentage changes that are likely data errors
        # Weekly safe limits: 0.5-1% normal, 2% medical supervision, 3% post-surgery max
        if change_percent > 10:  # More than 10% change needs scrutiny
            # Medical weight loss context:
            # - Bariatric surgery: typically 1-2% per week (max ~8% per month)
            # - Weight loss drugs: typically 0.5-1% per week (max ~4% per month)
            # - Extreme dieting: rarely exceeds 1% per week sustainably

            if change_percent > 50:
                # >50% change is almost certainly a data error
                logger.debug(f"Extreme change detected: {actual_change:.1f}kg ({change_percent:.1f}%) - data error")
                return False, OutlierType.RATE_VIOLATION, metadata

            elif change_percent > 30:
                # >30% change is highly suspicious
                if time_delta_days < 90:
                    # 30% in less than 3 months is not physiologically possible
                    logger.debug(f"Impossible change: {change_percent:.1f}% in {time_delta_days:.1f} days")
                    return False, OutlierType.RATE_VIOLATION, metadata
                elif time_delta_days < 180:
                    # 30% in 6 months is extreme even for bariatric surgery
                    logger.info(f"Extreme change flagged: {change_percent:.1f}% in {time_delta_days:.1f} days")
                    metadata['warning'] = 'extreme_change_requires_verification'
                    # Accept if rate is reasonable
                    daily_rate = change_percent / time_delta_days
                    if daily_rate > 0.3:  # More than 0.3% per day sustained is unlikely
                        return False, OutlierType.RATE_VIOLATION, metadata

            elif change_percent > 20:
                # 20-30% change
                if time_delta_days < 30:
                    # 20% in a month is too fast
                    logger.debug(f"Rapid change: {change_percent:.1f}% in {time_delta_days:.1f} days")
                    return False, OutlierType.RATE_VIOLATION, metadata
                elif time_delta_days < 60:
                    # 20% in 2 months is suspicious
                    daily_rate = change_percent / time_delta_days
                    if daily_rate > 0.25:  # More than 0.25% per day
                        logger.info(f"High rate: {daily_rate:.2f}%/day")
                        return False, OutlierType.RATE_VIOLATION, metadata

            elif change_percent > 10:
                # 10-20% change
                if time_delta_days < 7:
                    # More than 10% in a week is not possible
                    logger.debug(f"Impossible weekly change: {change_percent:.1f}% in {time_delta_days:.1f} days")
                    return False, OutlierType.RATE_VIOLATION, metadata
                elif time_delta_days < 14:
                    # 10-20% in 2 weeks is very suspicious
                    # Max sustainable is ~4% (2% per week)
                    if change_percent > 15:
                        logger.debug(f"Excessive 2-week change: {change_percent:.1f}%")
                        return False, OutlierType.RATE_VIOLATION, metadata
                    # Check daily rate
                    daily_rate = change_percent / time_delta_days
                    if daily_rate > 0.5:  # More than 0.5% per day (3.5% per week)
                        logger.info(f"High rate: {daily_rate:.2f}%/day")
                        return False, OutlierType.RATE_VIOLATION, metadata

        # Additional weekly rate check for any timeframe
        if time_delta_days >= 7:
            weekly_rate = (change_percent / time_delta_days) * 7
            if weekly_rate > 5.0:  # More than 5% per week averaged
                logger.info(f"Excessive weekly rate: {weekly_rate:.1f}% per week")
                return False, OutlierType.RATE_VIOLATION, metadata

        # For moderate changes, check against our calculated limits
        if actual_change > max_change:
            if time_delta_days < 7 and change_percent > 15:
                # 15% in a week is too much even with medical intervention
                logger.info(f"Rapid change: {change_percent:.1f}% in {time_delta_days:.1f} days")
                return False, OutlierType.RATE_VIOLATION, metadata
            elif time_delta_days >= 7:
                # Check daily rate
                daily_rate = change_percent / time_delta_days
                if daily_rate > max_percent:
                    logger.info(f"Rate violation: {daily_rate:.1f}%/day exceeds {max_percent}% limit")
                    return False, OutlierType.RATE_VIOLATION, metadata

        return True, None, metadata


class StatelessDeviationFilter:
    """
    Stateless deviation check using Kalman's predicted state.
    Replaces MAD filter for real-time processing.
    """

    def __init__(self,
                 normal_threshold_percent: float = 15.0,
                 extreme_threshold_percent: float = 30.0):
        self.normal_threshold_percent = normal_threshold_percent
        self.extreme_threshold_percent = extreme_threshold_percent

    def validate(self,
                measurement: WeightMeasurement,
                predicted_weight: Optional[float] = None) -> Tuple[bool, Optional[OutlierType], Dict]:
        """
        Check deviation from Kalman's prediction.
        This provides a simpler alternative to MAD for real-time.
        """
        metadata = {}

        if predicted_weight is None:
            return True, None, metadata

        deviation = abs(measurement.weight - predicted_weight)
        deviation_percent = (deviation / predicted_weight) * 100

        metadata['deviation_kg'] = deviation
        metadata['deviation_percent'] = deviation_percent
        metadata['predicted_weight'] = predicted_weight

        if deviation_percent > self.extreme_threshold_percent:
            logger.info(f"Extreme deviation: {measurement.weight:.1f}kg is {deviation_percent:.1f}% "
                       f"from predicted {predicted_weight:.1f}kg")
            return False, OutlierType.ADDITIVE, metadata
        elif deviation_percent > self.normal_threshold_percent:
            metadata['warning'] = 'high_deviation'

        return True, None, metadata


class StatelessLayer1Pipeline:
    """
    Stateless Layer 1 pipeline for real-time processing.
    Each measurement processed independently with optional state context.
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}

        self.enabled = config.get('enabled', True)

        self.physiological = PhysiologicalFilter(
            min_weight=config.get('min_weight', 30.0),
            max_weight=config.get('max_weight', 400.0)
        )

        self.rate_filter = StatelessRateOfChangeFilter(
            max_daily_change_percent=config.get('max_daily_change_percent', 3.0),
            medical_mode_percent=config.get('medical_mode_percent', 5.0)
        )

        self.deviation_filter = StatelessDeviationFilter(
            normal_threshold_percent=config.get('normal_threshold_percent', 10.0),
            extreme_threshold_percent=config.get('extreme_threshold_percent', 20.0)
        )

    def process(self,
                measurement: WeightMeasurement,
                state_context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[OutlierType], Dict]:
        """
        Process single measurement with optional state context.
        For real-time: state_context contains last known state and predictions.

        Args:
            measurement: Weight measurement to validate
            state_context: Optional context with:
                - last_known_state: Previous validated state
                - predicted_weight: Kalman prediction for this measurement
                - medical_intervention_mode: Boolean for relaxed limits

        Returns:
            (is_valid, outlier_type, metadata)
        """
        if not self.enabled:
            return True, None, {'layer1_enabled': False}

        metadata = {'layer': 1, 'checks_performed': []}
        state_context = state_context or {}

        # 1. Physiological limits (fastest, most obvious check)
        is_valid, outlier_type = self.physiological.validate(measurement)
        metadata['checks_performed'].append('physiological')

        if not is_valid:
            logger.debug(f"Layer1: Physiological limit violation: {measurement.weight}kg")
            metadata['filter'] = 'physiological'
            metadata['reason'] = f"Weight {measurement.weight}kg outside bounds [{self.physiological.min_weight}, {self.physiological.max_weight}]"
            return False, outlier_type, metadata

        # Check if Kalman will reset - if so, skip rate and deviation checks
        kalman_will_reset = state_context.get('kalman_will_reset', False)
        
        if kalman_will_reset:
            logger.info(f"Layer1: Skipping rate/deviation checks - Kalman will reset ({state_context.get('reset_reason', 'unknown')})")
            metadata['kalman_reset_pending'] = True
            metadata['checks_skipped'] = ['rate_of_change', 'deviation']
            return True, None, metadata
            
        # 2. Rate of change check (if we have previous state)
        last_state = state_context.get('last_known_state')
        if last_state:
            is_valid, outlier_type, rate_meta = self.rate_filter.validate(measurement, last_state)
            metadata['checks_performed'].append('rate_of_change')
            metadata.update(rate_meta)

            if not is_valid:
                logger.debug(f"Layer1: Rate of change violation for {measurement.weight}kg")
                metadata['filter'] = 'rate_of_change'
                metadata['reason'] = f"Change of {rate_meta['actual_change_kg']:.1f}kg ({rate_meta['change_percent']:.1f}%) exceeds limit"
                return False, outlier_type, metadata

        # 3. Deviation from prediction (if available from Kalman)
        predicted_weight = state_context.get('predicted_weight')
        if predicted_weight:
            is_valid, outlier_type, dev_meta = self.deviation_filter.validate(measurement, predicted_weight)
            metadata['checks_performed'].append('deviation')
            metadata.update(dev_meta)

            if not is_valid:
                logger.debug(f"Layer1: Extreme deviation for {measurement.weight}kg")
                metadata['filter'] = 'deviation'
                metadata['reason'] = f"Deviation of {dev_meta['deviation_percent']:.1f}% from predicted {predicted_weight:.1f}kg"
                return False, outlier_type, metadata

        metadata['passed_all_checks'] = True
        return True, None, metadata

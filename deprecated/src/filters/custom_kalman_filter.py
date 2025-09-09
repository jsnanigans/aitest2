import logging
from datetime import datetime
from typing import Any, Optional

import numpy as np
from pykalman import KalmanFilter
from .validation_gate import ValidationGate

logger = logging.getLogger(__name__)


class CustomKalmanFilter:
    """
    Custom Kalman filter for weight tracking with trend analysis.

    Tracks both weight and weight change rate (trend) using a 2D state vector.
    State vector: [weight, weight_change_rate] where change rate is in kg/day.

    Features:
    - Tracks weight and trend simultaneously
    - Velocity-aware adaptive outlier rejection
    - Enhanced time-adaptive process noise for handling gaps
    - Implied velocity detection for impossible changes
    - 7-day weight prediction
    - Trend statistics and change detection
    """

    def __init__(
        self,
        initial_weight: float = 100.0,
        initial_trend: float = 0.0,
        initial_variance: Optional[float] = None,
        process_noise_weight: float = 1.0,
        max_reasonable_trend: float = 0.1,  # 100g/day = ~700g/week max
        process_noise_trend: float = 0.01,  # Moderate trend stability
        measurement_noise: float = 0.5,
        source_trust_config: Optional[dict] = None,
        validation_gamma: float = 3.0,
        enable_validation: bool = True,
        enable_adaptive_validation: bool = True,
    ):
        self.max_reasonable_trend = max_reasonable_trend
        self.base_measurement_noise = measurement_noise if initial_variance is None else np.sqrt(initial_variance)
        self.base_process_noise_weight = process_noise_weight
        self.base_process_noise_trend = process_noise_trend

        self.source_trust_config = source_trust_config or {
            "care-team-upload": {"trust": 0.95, "noise_scale": 0.3},
            "internal-questionnaire": {"trust": 0.8, "noise_scale": 0.5},
            "patient-upload": {"trust": 0.6, "noise_scale": 0.8},
            "https://connectivehealth.io": {"trust": 0.3, "noise_scale": 1.5},
            "https://api.iglucose.com": {"trust": 0.5, "noise_scale": 1.0},
            "patient-device": {"trust": 0.7, "noise_scale": 0.8},
            "unknown": {"trust": 0.5, "noise_scale": 1.0},
        }
        
        initial_weight_variance = initial_variance if initial_variance is not None else measurement_noise

        self.kf = KalmanFilter(
            initial_state_mean=[initial_weight, initial_trend],
            initial_state_covariance=[
                [initial_weight_variance, 0.0],
                [0.0, 0.001]
            ],
            transition_matrices=[
                [1.0, 1.0],
                [0.0, 1.0]
            ],
            observation_matrices=[[1.0, 0.0]],
            transition_covariance=np.array([
                [self.base_process_noise_weight, 0.0],
                [0.0, self.base_process_noise_trend]
            ]),
            observation_covariance=[[measurement_noise]]
        )

        self.current_mean = None
        self.current_covariance = None
        self.measurement_count = 0
        self.last_timestamp = None

        self.trend_history = []
        self.innovation_history = []
        self.weight_history = []
        self.last_trend_update_time = None
        self.peak_trend = 0.0
        
        self.enable_validation = enable_validation
        self.validation_gate = ValidationGate(
            gamma=validation_gamma,
            enable_adaptive=enable_adaptive_validation
        ) if enable_validation else None
        
        self.rejected_measurements = []

    def process_measurement(
        self,
        weight: float,
        timestamp: Optional[datetime] = None,
        time_delta_days: float = 1.0,
        source_type: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Process a weight measurement with optional timestamp.

        Args:
            weight: Measured weight in kg
            timestamp: Optional timestamp for the measurement
            time_delta_days: Time since last measurement in days (used if no timestamp)

        Returns:
            Dictionary with filtered results including trend
        """

        if timestamp and self.last_timestamp:
            time_delta = (timestamp - self.last_timestamp).total_seconds() / 86400.0
            time_delta_days = max(0.1, time_delta)
        
        # Check if we should force reinitialization due to extreme gap or stuck filter
        should_reinit = False
        
        # Case 1: Extreme time gap (>90 days)
        if time_delta_days > 90:
            logger.info(f"Extreme gap detected ({time_delta_days:.0f} days) - forcing reinitialization")
            should_reinit = True
        
        # Case 2: Too many consecutive rejections (filter is stuck)
        elif hasattr(self, 'rejected_measurements') and len(self.rejected_measurements) >= 5:
            # Check if the last 5 measurements were all rejected
            consecutive_count = 0
            for i in range(len(self.rejected_measurements) - 1, -1, -1):
                rej = self.rejected_measurements[i]
                # Check if this rejection was recent (within last 10 measurements)
                if self.measurement_count - rej.get('measurement_count', 0) == consecutive_count:
                    consecutive_count += 1
                else:
                    break
            
            if consecutive_count >= 5:
                logger.info(f"Filter stuck with {consecutive_count} consecutive rejections - forcing reinitialization")
                should_reinit = True
        
        # Case 3: Check if new measurement is dramatically different from current state
        elif self.current_mean is not None:
            weight_diff = abs(weight - self.current_mean[0])
            if weight_diff > 20.0 and time_delta_days > 30:
                logger.info(f"Large weight change ({weight_diff:.1f}kg) after {time_delta_days:.0f} day gap - forcing reinitialization")
                should_reinit = True
        
        if should_reinit:
            # Force reinitialization with new weight as starting point
            logger.info(f"Reinitializing filter with weight={weight:.1f}kg")
            self.current_mean = np.array([weight, 0.0])
            self.current_covariance = np.array([
                [2.0, 0.0],  # Higher initial variance for uncertainty
                [0.0, 0.001]
            ])
            self.rejected_measurements = []  # Clear rejection history
            self.innovation_history = []  # Clear innovation history
            self.trend_history = []  # Clear trend history
            # Don't reset measurement_count to maintain continuity
            self.last_timestamp = timestamp
            
            return {
                'filtered_weight': weight,
                'measured_weight': weight,
                'predicted_weight': weight,
                'trend_kg_per_day': 0.0,
                'trend_kg_per_week': 0.0,
                'uncertainty_weight': np.sqrt(2.0),
                'uncertainty_trend': np.sqrt(0.001),
                'innovation': 0.0,
                'normalized_innovation': 0.0,
                'measurement_count': self.measurement_count,
                'time_delta_days': time_delta_days,
                'measurement_accepted': True,
                'confidence': 1.0,
                'validation_metrics': None,
                'filter_reinitialized': True
            }

        if self.current_mean is None:
            self.current_mean = np.array([weight, 0.0])
            self.current_covariance = np.array([
                [2.0, 0.0],
                [0.0, 0.001]
            ])
            self.measurement_count += 1
            self.last_timestamp = timestamp

            return {
                'filtered_weight': weight,
                'measured_weight': weight,
                'predicted_weight': weight,
                'trend_kg_per_day': 0.0,
                'trend_kg_per_week': 0.0,
                'uncertainty_weight': np.sqrt(2.0),
                'uncertainty_trend': np.sqrt(0.05),
                'innovation': 0.0,
                'normalized_innovation': 0.0,
                'measurement_count': self.measurement_count,
                'time_delta_days': time_delta_days,
                'measurement_accepted': True,
                'confidence': 1.0,
                'validation_metrics': None
            }

        # Standard transition matrix without decay
        # For very long gaps, decay the trend to prevent unrealistic drift
        # People don't gain/lose weight linearly for months
        if time_delta_days > 30:
            # More aggressive decay of trend for long gaps
            # After 30 days, start reducing trend influence
            # At 60 days, trend is ~25% of original
            # At 90 days, trend is ~5% of original
            # Beyond 180 days, trend is essentially zero
            trend_decay = np.exp(-max(0, time_delta_days - 30) / 30.0)
            effective_trend = self.current_mean[1] * trend_decay
            
            # Use modified transition for long gaps
            transition_matrix = np.array([
                [1.0, time_delta_days * trend_decay],
                [0.0, trend_decay]
            ])
            
            # Manually compute prediction with decayed trend
            predicted_weight = float(self.current_mean[0] + effective_trend * time_delta_days)
            predicted_trend = effective_trend
            predicted_mean = np.array([predicted_weight, predicted_trend])
        else:
            # Normal transition for short gaps
            transition_matrix = np.array([
                [1.0, time_delta_days],
                [0.0, 1.0]
            ])
            predicted_mean = transition_matrix @ self.current_mean
            predicted_weight = float(predicted_mean[0])
            predicted_trend = float(predicted_mean[1])
        
        # Initialize recent_exact_values if not present (for backward compatibility)
        if not hasattr(self, 'recent_exact_values'):
            self.recent_exact_values = []
            self.max_exact_history = 20
        
        # Add current measurement to recent values BEFORE checking for duplicates
        self.recent_exact_values.append(weight)
        if len(self.recent_exact_values) > self.max_exact_history:
            self.recent_exact_values.pop(0)

        # Enhanced time-adaptive process noise with measurement history dampening
        # As we get more measurements, we become more confident about the underlying trend
        # Very strong dampening for maximum trend stability
        history_factor = max(0.05, 1.0 / (1.0 + self.measurement_count / 5.0))

        # Calculate data-driven trend stability from recent innovations
        if len(self.innovation_history) > 5:
            recent_innovations = [abs(h['innovation']) for h in self.innovation_history[-20:]]
            avg_innovation = np.mean(recent_innovations)
            # Ultra-conservative with trend changes
            stability_factor = max(0.05, min(0.3, avg_innovation / 10.0))
            
            # Detect persistent bias but respond more aggressively to consistent evidence
            recent_signed_innovations = [h['innovation'] for h in self.innovation_history[-10:]]
            bias = np.mean(recent_signed_innovations)
            
            # Check for consistent direction of error (all recent innovations same sign)
            if len(recent_signed_innovations) >= 3:
                # Check last 3 innovations for consistency
                last_3 = recent_signed_innovations[-3:]
                same_sign = all(i * bias > 0 for i in last_3) if bias != 0 else False
                
                # If we've rejected several measurements in a row, we might be wrong
                recent_rejections = sum(1 for r in self.rejected_measurements[-5:] 
                                      if hasattr(self, 'rejected_measurements') and 
                                      len(self.rejected_measurements) > 0)
                
                # Check if we've had too many consecutive rejections
                consecutive_rejections = 0
                if hasattr(self, 'rejected_measurements') and len(self.rejected_measurements) > 0:
                    for r in reversed(self.rejected_measurements):
                        if hasattr(r, 'get') and r.get('timestamp'):
                            # Check if recent (within last few measurements)
                            if self.measurement_count - r.get('measurement_count', 0) <= 5:
                                consecutive_rejections += 1
                            else:
                                break
                
                # If we've rejected 5+ consecutive measurements, we need to adapt to new reality
                if consecutive_rejections >= 5:
                    logger.debug(f"Detected {consecutive_rejections} consecutive rejections - filter may be stuck")
                    # Force aggressive adaptation
                    stability_factor = 10.0  # Much higher to allow rapid adjustment
                    # Note: process noise will be increased later when q_weight and q_trend are defined
                elif (same_sign and abs(bias) > 1.0) or recent_rejections >= 3:
                    # Strong consistent evidence that we're wrong - adapt much faster
                    logger.debug(f"Consistent bias detected: {bias:.2f}kg, recent rejections: {recent_rejections}")
                    bias_factor = 2.0 + abs(bias) / 1.0  # Very aggressive correction
                    stability_factor = min(5.0, stability_factor * bias_factor)
                    # Note: process noise adjustment will be applied later
                elif abs(bias) > 2.0:  # Very strong bias
                    bias_factor = 1.0 + abs(bias) / 3.0
                    stability_factor = min(2.0, stability_factor * bias_factor)
        else:
            stability_factor = 0.3

        if time_delta_days < 0.1:
            q_weight = self.base_process_noise_weight * time_delta_days
            q_trend = self.base_process_noise_trend * time_delta_days * history_factor * stability_factor
        elif time_delta_days < 1.0:
            q_weight = self.base_process_noise_weight * time_delta_days
            q_trend = self.base_process_noise_trend * np.sqrt(time_delta_days) * history_factor * stability_factor
        elif time_delta_days < 7.0:
            gap_factor = time_delta_days / 7.0
            q_weight = self.base_process_noise_weight * time_delta_days * (1 + gap_factor * 0.5)
            q_trend = self.base_process_noise_trend * (1 + gap_factor * 2) * time_delta_days * history_factor
        else:
            # For large gaps, significantly increase uncertainty
            # Weight can change substantially over months
            if time_delta_days > 180:
                # For very long gaps (6+ months), assume maximum uncertainty
                q_weight = self.base_process_noise_weight * time_delta_days * 2.0
                q_trend = self.base_process_noise_trend * 0.1  # Minimal trend noise since trend is decayed
            elif time_delta_days > 90:
                # For long gaps (3-6 months), high uncertainty
                q_weight = self.base_process_noise_weight * time_delta_days * 1.5
                q_trend = self.base_process_noise_trend * 0.5
            else:
                # For moderate gaps (1-3 months)
                gap_factor = min(time_delta_days / 7.0, 4.0)
                q_weight = self.base_process_noise_weight * np.sqrt(time_delta_days) * (1 + gap_factor)
                q_trend = self.base_process_noise_trend * gap_factor * 5 * np.sqrt(time_delta_days)

        # Check again for consecutive rejections and increase process noise if needed
        consecutive_rejections = 0
        if hasattr(self, 'rejected_measurements') and len(self.rejected_measurements) > 0:
            for r in reversed(self.rejected_measurements):
                if hasattr(r, 'get') and r.get('timestamp'):
                    if self.measurement_count - r.get('measurement_count', 0) <= 5:
                        consecutive_rejections += 1
                    else:
                        break
        
        if consecutive_rejections >= 5:
            # Significantly increase process noise to allow filter to adapt
            q_weight *= 5.0
            q_trend *= 5.0
            logger.debug(f"Increasing process noise due to {consecutive_rejections} consecutive rejections")
        
        # Source-based dynamic parameter adjustment
        source_config = self.source_trust_config.get(
            source_type if source_type else "unknown",
            self.source_trust_config["unknown"]
        )
        source_trust = source_config["trust"]
        noise_scale = source_config["noise_scale"]

        # Adjust process noise based on source trust
        # Even trusted sources should not cause rapid trend changes
        trust_adjusted_q_weight = q_weight * (1.0 + (source_trust - 0.5) * 0.3)
        trust_adjusted_q_trend = q_trend * (1.0 + (source_trust - 0.5) * 0.2)

        Q = np.array([
            [trust_adjusted_q_weight, 0.0],
            [0.0, trust_adjusted_q_trend]
        ])
        predicted_cov = transition_matrix @ self.current_covariance @ transition_matrix.T + Q
        
        # Physiological reality check based on clinical guidelines
        # Daily fluctuations can be 2-3% of body weight (framework section 1.1)
        # Be more permissive to avoid breaking the filter
        max_daily_fluctuation = predicted_weight * 0.04  # 4% of current weight
        max_sustained_daily_change = 0.5  # 500g/day for sustained changes
        
        # For short timeframes, allow normal fluctuations; for longer periods, use sustained rates
        if time_delta_days <= 1.0:
            max_allowed_change = max_daily_fluctuation
        else:
            # More permissive for longer gaps
            max_allowed_change = max_daily_fluctuation + (max_sustained_daily_change * time_delta_days)
        
        # Check for suspicious repeated exact values (common device error pattern)
        # Count exact matches in recent history
        exact_value_count = sum(1 for v in self.recent_exact_values if abs(v - weight) < 0.001)
        
        # Get median of values BEFORE the repeated ones started
        if len(self.weight_history) >= 5:
            # Exclude the repeated value from median calculation
            non_repeated_values = [v for v in self.weight_history[-15:] if abs(v - weight) > 0.001]
            if len(non_repeated_values) >= 3:
                recent_median = np.median(non_repeated_values)
            else:
                recent_median = predicted_weight
        else:
            recent_median = predicted_weight
        
        if exact_value_count >= 5:  # Increased threshold - need 5+ duplicates
            # Multiple exact same values is suspicious for weight data
            # Real weight measurements have natural variation
            deviation_from_median = abs(weight - recent_median)
            deviation_from_predicted = abs(weight - predicted_weight)
            
            # Only reject if it's really suspicious
            if (deviation_from_median > 3.0 and deviation_from_predicted > 3.0) or exact_value_count >= 8:
                # Repeated value that's also different from trend = device error
                logger.info(f"Rejecting repeated device error: {weight:.3f}kg appears {exact_value_count} times, "
                           f"median of non-repeated: {recent_median:.1f}kg, predicted: {predicted_weight:.1f}kg")
                
                # CRITICAL: Still update time progression even when rejecting
                self.measurement_count += 1
                self.last_timestamp = timestamp
                
                # Update state covariance to reflect time passage (prediction only)
                self.current_mean = predicted_mean
                self.current_covariance = predicted_cov
                
                # Return prediction without measurement update
                return {
                    'filtered_weight': predicted_weight,
                    'measured_weight': weight,
                    'predicted_weight': predicted_weight,
                    'weight_variance': float(predicted_cov[0, 0]),
                    'trend': predicted_trend,
                    'trend_kg_per_day': predicted_trend,
                    'trend_kg_per_week': predicted_trend * 7.0,
                    'uncertainty_weight': float(np.sqrt(predicted_cov[0, 0])),
                    'innovation': weight - predicted_weight,
                    'normalized_innovation': abs(weight - predicted_weight) / np.sqrt(predicted_cov[0, 0] + self.base_measurement_noise),
                    'measurement_accepted': False,
                    'rejection_reason': 'repeated_device_error',
                    'confidence': 0.0,
                    'predicted_weight_7d': predicted_weight + predicted_trend * 7.0,
                    'time_delta_days': time_delta_days,
                    'source_trust': source_trust,
                    'measurement_count': self.measurement_count,
                    'validation_metrics': None
                }
        
        # Account for weekly patterns if we have enough history and timestamp available
        weekly_adjustment = 1.0
        if timestamp and self.measurement_count > 14:
            day_of_week = timestamp.weekday()
            # Weekend weights tend to be higher (framework section 1.1)
            if day_of_week in [5, 6, 0]:  # Sat, Sun, Mon
                weekly_adjustment = 1.2  # Allow 20% more deviation on these days
        
        # Check if measurement violates physiological limits
        weight_deviation = abs(weight - predicted_weight)
        adjusted_max_change = max_allowed_change * weekly_adjustment
        
        # Only reject truly impossible values
        if weight < 20 or weight > 500 or weight_deviation > max(15.0, adjusted_max_change * 2):
            # This violates physiological constraints - reject immediately
            rejection_detail = ""
            if weight < 30:
                rejection_detail = f"below minimum (30kg)"
            elif weight > 400:
                rejection_detail = f"above maximum (400kg)"
            else:
                rejection_detail = f"deviation {weight_deviation:.1f}kg exceeds max {adjusted_max_change:.1f}kg"
            
            logger.info(f"Physiologically impossible: {weight:.1f}kg vs predicted {predicted_weight:.1f}kg - {rejection_detail}")
            
            # CRITICAL: Still update time progression even when rejecting
            self.measurement_count += 1
            self.last_timestamp = timestamp
            
            # Update state covariance to reflect time passage (prediction only)
            self.current_mean = predicted_mean
            self.current_covariance = predicted_cov
            
            # Return prediction without measurement update
            return {
                'filtered_weight': predicted_weight,
                'measured_weight': weight,
                'predicted_weight': predicted_weight,
                'weight_variance': float(predicted_cov[0, 0]),
                'trend': predicted_trend,
                'trend_kg_per_day': predicted_trend,
                'trend_kg_per_week': predicted_trend * 7.0,
                'uncertainty_weight': float(np.sqrt(predicted_cov[0, 0])),
                'innovation': weight - predicted_weight,
                'normalized_innovation': weight_deviation / np.sqrt(predicted_cov[0, 0] + self.base_measurement_noise),
                'measurement_accepted': False,
                'rejection_reason': 'physiologically_impossible',
                'confidence': 0.0,
                'predicted_weight_7d': predicted_weight + predicted_trend * 7.0,
                'time_delta_days': time_delta_days,
                'source_trust': self.source_trust_config.get(source_type or "unknown", {}).get("trust", 0.5),
                'measurement_count': self.measurement_count,
                'validation_metrics': None
            }

        innovation = weight - predicted_weight
        innovation_variance = predicted_cov[0, 0] + self.base_measurement_noise
        uncertainty = float(np.sqrt(innovation_variance))
        normalized_innovation = abs(innovation) / uncertainty if uncertainty > 0 else 0
        
        implied_velocity = innovation / time_delta_days if time_delta_days > 0 else 0
        
        self.innovation_history.append({
            'innovation': innovation,
            'normalized': normalized_innovation,
            'implied_velocity': implied_velocity,
            'timestamp': timestamp,
            'normalized_innovation': normalized_innovation
        })
        
        # VALIDATION GATE: Check measurement BEFORE updating state
        validation_passed = True
        confidence = 0.5
        rejection_reason = None
        
        if self.enable_validation and self.validation_gate:
            # Additional check for sudden jumps followed by contradictory evidence
            if len(self.weight_history) >= 3:
                recent_weights = self.weight_history[-3:]
                recent_median = np.median(recent_weights)
                jump_size = abs(weight - recent_median)
                
                # If this is a big jump from recent median, be more conservative
                if jump_size > 3.0:
                    # Temporarily reduce gamma (be more strict)
                    original_gamma = self.validation_gate.gamma
                    self.validation_gate.gamma = min(2.0, original_gamma - 0.5)
                    validation_passed, confidence, rejection_reason = self.validation_gate.validate(
                        measurement=weight,
                        prediction=predicted_weight,
                        innovation_covariance=innovation_variance,
                        user_history=self.innovation_history
                    )
                    self.validation_gate.gamma = original_gamma
                else:
                    validation_passed, confidence, rejection_reason = self.validation_gate.validate(
                        measurement=weight,
                        prediction=predicted_weight,
                        innovation_covariance=innovation_variance,
                        user_history=self.innovation_history
                    )
            
            if not validation_passed:
                # Measurement rejected - DO NOT UPDATE STATE
                self.rejected_measurements.append({
                    'timestamp': timestamp,
                    'measurement': weight,
                    'prediction': predicted_weight,
                    'normalized_innovation': normalized_innovation,
                    'implied_velocity': implied_velocity,
                    'reason': rejection_reason,
                    'confidence': confidence,
                    'measurement_count': self.measurement_count
                })
                
                logger.info(f"Measurement rejected: {weight:.1f}kg (predicted: {predicted_weight:.1f}kg), "
                           f"reason: {rejection_reason}, normalized innovation: {normalized_innovation:.2f}σ")
                
                # CRITICAL: Still update time progression even when rejecting
                self.measurement_count += 1
                self.last_timestamp = timestamp
                
                # Update state covariance to reflect time passage (prediction only)  
                self.current_mean = predicted_mean
                self.current_covariance = predicted_cov
                
                # Return prediction without measurement update
                return {
                    'filtered_weight': predicted_weight,
                    'measured_weight': weight,
                    'predicted_weight': predicted_weight,
                    'weight_variance': float(predicted_cov[0, 0]),
                    'trend': predicted_trend,
                    'trend_kg_per_day': predicted_trend,
                    'trend_kg_per_week': predicted_trend * 7.0,
                    'uncertainty_weight': float(np.sqrt(predicted_cov[0, 0])),
                    'innovation': innovation,
                    'normalized_innovation': normalized_innovation,
                    'measurement_accepted': False,
                    'rejection_reason': rejection_reason,
                    'confidence': confidence,
                    'predicted_weight_7d': predicted_weight + predicted_trend * 7.0,
                    'time_delta_days': time_delta_days,
                    'source_trust': source_trust,
                    'measurement_count': self.measurement_count,
                    'validation_metrics': self.validation_gate.get_metrics() if self.validation_gate else None
                }
        
        # Measurement passed validation or validation disabled - proceed with update
        measurement_noise = self.base_measurement_noise * noise_scale
        
        # Apply adaptive measurement noise based on implied velocity for extreme cases
        if abs(implied_velocity) > self.max_reasonable_trend:
            impossibility_factor = abs(implied_velocity) / self.max_reasonable_trend
            measurement_noise *= min(impossibility_factor ** 2, 10.0)  # Cap the scaling
            logger.debug(f"High velocity {implied_velocity:.3f} kg/day, adjusting measurement noise")
        
        if abs(predicted_trend) > self.max_reasonable_trend:
            logger.debug(f"Unreasonable trend detected: {predicted_trend:.4f} kg/day")
            Q[1, 1] *= 10
        
        temp_kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=[[1.0, 0.0]],
            transition_covariance=Q,
            observation_covariance=[[measurement_noise]]
        )
        
        (self.current_mean,
         self.current_covariance) = temp_kf.filter_update(
            self.current_mean,
            self.current_covariance,
            observation=np.array([weight])
        )
        
        self.measurement_count += 1
        self.last_timestamp = timestamp
        self.weight_history.append(weight)

        filtered_weight = float(self.current_mean[0])
        filtered_trend = float(self.current_mean[1])



        # Apply trend smoothing based on measurement history (very aggressive smoothing)
        if self.measurement_count > 5 and len(self.trend_history) > 0:
            recent_trends = [h['trend'] for h in self.trend_history[-30:]]
            trend_mean = np.mean(recent_trends)
            trend_std = np.std(recent_trends) if len(recent_trends) > 1 else 0.01

            # Dampen trend changes very aggressively (1 std)
            if abs(filtered_trend - trend_mean) > 1.0 * trend_std and trend_std > 0:
                damping_factor = 0.2 + 0.2 * np.exp(-abs(filtered_trend - trend_mean) / (trend_std + 0.01))
                filtered_trend = trend_mean + (filtered_trend - trend_mean) * damping_factor
                self.current_mean[1] = filtered_trend
            
            # Additional physiological constraint on trend
            # Framework suggests sustained weight loss/gain rarely exceeds 1-2 lbs/week (0.45-0.9 kg/week)
            # That's about 0.065-0.13 kg/day - let's use the upper bound
            max_sustainable_trend = 0.13  # 130g/day = ~900g/week sustainable
            if abs(filtered_trend) > max_sustainable_trend:
                # Soft cap - gradually reduce extreme trends
                filtered_trend = np.sign(filtered_trend) * (
                    max_sustainable_trend + (abs(filtered_trend) - max_sustainable_trend) * 0.3
                )
                self.current_mean[1] = filtered_trend

        if abs(filtered_trend) > self.max_reasonable_trend:
            filtered_trend = np.sign(filtered_trend) * self.max_reasonable_trend
            self.current_mean[1] = filtered_trend

        self.trend_history.append({
            'trend': filtered_trend,
            'timestamp': timestamp,
            'measurement_count': self.measurement_count
        })

        uncertainty_weight = float(np.sqrt(self.current_covariance[0, 0]))
        uncertainty_trend = float(np.sqrt(self.current_covariance[1, 1]))

        return {
            'filtered_weight': filtered_weight,
            'measured_weight': weight,
            'predicted_weight': predicted_weight,
            'trend_kg_per_day': filtered_trend,
            'trend_kg_per_week': filtered_trend * 7.0,
            'uncertainty_weight': uncertainty_weight,
            'uncertainty_trend': uncertainty_trend,
            'innovation': innovation,
            'normalized_innovation': normalized_innovation,
            'implied_velocity': implied_velocity,
            'measurement_count': self.measurement_count,
            'time_delta_days': time_delta_days,
            'measurement_noise_used': measurement_noise,
            'source_type': source_type,
            'source_trust': source_trust,
            'measurement_accepted': True,
            'confidence': confidence,
            'validation_metrics': self.validation_gate.get_metrics() if self.validation_gate else None
        }

    def predict_future(self, days_ahead: int = 7) -> Optional[dict[str, Any]]:
        """
        Predict future weight based on current state and trend.

        Args:
            days_ahead: Number of days to predict ahead

        Returns:
            Dictionary with predictions and uncertainty bounds
        """
        if self.current_mean is None or self.current_covariance is None:
            return None

        predictions = []
        uncertainties = []

        mean = self.current_mean.copy()
        cov = self.current_covariance.copy()

        for day in range(1, days_ahead + 1):
            # Standard prediction without decay
            A = np.array([[1.0, 1.0], [0.0, 1.0]])

            mean = A @ mean
            cov = A @ cov @ A.T + self.kf.transition_covariance

            weight_pred = float(mean[0])
            weight_uncertainty = float(np.sqrt(cov[0, 0]))

            predictions.append({
                'day': day,
                'weight': weight_pred,
                'uncertainty': weight_uncertainty,
                'trend_at_day': float(mean[1]),
                'confidence_interval_95': (
                    weight_pred - 1.96 * weight_uncertainty,
                    weight_pred + 1.96 * weight_uncertainty
                )
            })
            uncertainties.append(weight_uncertainty)

        current_weight = float(self.current_mean[0])
        current_trend = float(self.current_mean[1])

        return {
            'current_weight': current_weight,
            'current_trend_kg_per_day': current_trend,
            'predictions': predictions,
            'total_predicted_change': predictions[-1]['weight'] - current_weight,
            'max_uncertainty': max(uncertainties)
        }

    def handle_missing_data(self, time_gap_days: int) -> Optional[dict[str, Any]]:
        """Handle missing measurements by prediction only."""
        if self.current_mean is None or self.current_covariance is None:
            return None

        # Standard prediction without decay
        A = np.array([
            [1.0, float(time_gap_days)],
            [0.0, 1.0]
        ])

        trans_cov = self.kf.transition_covariance
        if trans_cov is None:
            trans_cov = np.array([[0.5, 0.0], [0.0, 0.01]])
        Q = np.array([
            [trans_cov[0, 0] * time_gap_days, 0.0],
            [0.0, trans_cov[1, 1] * time_gap_days]
        ])

        self.current_mean = A @ self.current_mean
        self.current_covariance = A @ self.current_covariance @ A.T + Q

        return {
            'filtered_weight': float(self.current_mean[0]),
            'trend_kg_per_day': float(self.current_mean[1]),
            'uncertainty_weight': float(np.sqrt(self.current_covariance[0, 0])),
            'uncertainty_trend': float(np.sqrt(self.current_covariance[1, 1])),
            'gap_days': time_gap_days
        }

    def get_state(self) -> Optional[dict[str, Any]]:
        """Get current filter state including trend."""
        if self.current_mean is None or self.current_covariance is None:
            return None

        weight = float(self.current_mean[0])
        trend = float(self.current_mean[1])

        trend_stats = self._calculate_trend_statistics()

        return {
            'weight': weight,
            'trend_kg_per_day': trend,
            'trend_kg_per_week': trend * 7.0,
            'uncertainty_weight': float(np.sqrt(self.current_covariance[0, 0])),
            'uncertainty_trend': float(np.sqrt(self.current_covariance[1, 1])),
            'measurement_count': self.measurement_count,
            'trend_direction': 'gaining' if trend > 0.01 else ('losing' if trend < -0.01 else 'stable'),
            'trend_statistics': trend_stats
        }

    def _calculate_trend_statistics(self) -> dict[str, Any]:
        """Calculate statistics about trend history."""
        if not self.trend_history:
            return {}

        trends = [t['trend'] for t in self.trend_history]
        recent_trends = trends[-7:] if len(trends) >= 7 else trends

        return {
            'mean_trend': float(np.mean(trends)),
            'std_trend': float(np.std(trends)),
            'recent_mean_trend': float(np.mean(recent_trends)),
            'max_trend': float(np.max(np.abs(trends))),
            'trend_changes': self._count_trend_changes(trends)
        }

    def _count_trend_changes(self, trends: list) -> int:
        """Count number of times trend changed direction significantly."""
        if len(trends) < 2:
            return 0

        changes = 0
        threshold = 0.01

        for i in range(1, len(trends)):
            if trends[i-1] > threshold and trends[i] < -threshold:
                changes += 1
            elif trends[i-1] < -threshold and trends[i] > threshold:
                changes += 1

        return changes

    def get_innovation_analysis(self) -> dict[str, Any]:
        """Analyze innovation history for diagnostics."""
        if not self.innovation_history:
            return {}

        innovations = [i['innovation'] for i in self.innovation_history]
        normalized = [i['normalized'] for i in self.innovation_history]

        return {
            'mean_innovation': float(np.mean(innovations)),
            'std_innovation': float(np.std(innovations)),
            'mean_normalized': float(np.mean(normalized)),
            'outlier_count': sum(1 for n in normalized if n > 3.0),
            'extreme_outlier_count': sum(1 for n in normalized if n > 6.0),
            'outlier_rate': sum(1 for n in normalized if n > 3.0) / len(normalized) if len(normalized) > 0 else 0
        }
    
    def get_validation_summary(self) -> Optional[dict[str, Any]]:
        """Get summary of validation gate performance."""
        if not self.validation_gate:
            return None
            
        metrics = self.validation_gate.get_metrics()
        
        return {
            'metrics': metrics,
            'rejected_measurements': self.rejected_measurements,
            'total_rejected': len(self.rejected_measurements),
            'should_rebaseline': self.validation_gate.should_rebaseline()
        }
    
    def reset_validation_gate(self):
        """Reset validation gate statistics."""
        if self.validation_gate:
            self.validation_gate.reset()
        self.rejected_measurements = []
        
        # Track weekly patterns (framework mentions 0.35% weekly variation)
        self.weekly_pattern = {}  # Day of week -> typical deviation
        self.weekly_measurements = {i: [] for i in range(7)}  # 0=Monday, 6=Sunday
        
        # Track repeated values (device errors often produce exact duplicates)
        self.recent_exact_values = []
        self.max_exact_history = 20

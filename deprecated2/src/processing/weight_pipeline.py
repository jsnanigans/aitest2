"""
Robust Kalman Filter Weight Processing Pipeline with Layer 1 Pre-filtering
Enhanced pipeline with physiological gates and outlier-resistant Kalman filtering
Layer 1 catches impossible values, Kalman handles statistical outliers
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

from src.core.types import BaselineResult, ProcessedMeasurement, WeightMeasurement
from src.filters.robust_kalman import RobustKalmanFilter, AdaptiveValidationGate
from src.filters.layer1_heuristic import StatelessLayer1Pipeline
from src.processing.robust_baseline import RobustBaselineEstimator

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    INITIALIZATION = "initialization"
    REAL_TIME = "real_time"
    BATCH = "batch"


class WeightProcessingPipeline:
    """
    Robust Kalman filter pipeline with adaptive outlier handling.
    Enhanced resistance to extreme outliers that would corrupt state estimates.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.config = config

        # Source trust configuration
        self.source_trust = config.get('source_trust', {
            "care-team-upload": 0.95,
            "internal-questionnaire": 0.8,
            "patient-upload": 0.6,
            "patient-device": 0.7,
            "unknown": 0.5
        })

        # Initialize Layer 1 pre-filtering
        self.layer1 = StatelessLayer1Pipeline(config.get('layer1', {}))
        
        # Initialize components (lazy loading for some)
        self.baseline_estimator = RobustBaselineEstimator(
            config.get('baseline', {})
        )

        self.kalman = None  # Initialized after baseline
        self.validation_gate = None  # Initialized with Kalman

        # State tracking
        self.baseline_result: Optional[BaselineResult] = None
        self.is_initialized = False
        self.measurements_processed = 0
        self.layer1_rejections = 0
        self.outliers_detected = 0
        self.extreme_outliers_detected = 0
        self.processing_history = []

    def initialize_user(self, measurements: List[WeightMeasurement]) -> BaselineResult:
        """
        Initialize pipeline for a new user.
        Establishes baseline and initializes Kalman filter.
        """
        logger.info(f"Initializing pipeline with {len(measurements)} measurements")

        # Step 1: Establish robust baseline
        self.baseline_result = self.baseline_estimator.establish_baseline(measurements)

        if not self.baseline_result.success:
            logger.error(f"Baseline establishment failed: {self.baseline_result.error}")
            return self.baseline_result

        logger.info(f"Baseline established: {self.baseline_result.baseline_weight:.1f}kg "
                   f"(variance={self.baseline_result.measurement_variance:.3f}, "
                   f"confidence={self.baseline_result.confidence})")

        # Step 2: Initialize Robust Kalman filter with baseline parameters
        kalman_config = self.config.get('kalman', {})
        self.kalman = RobustKalmanFilter(
            initial_weight=self.baseline_result.baseline_weight,
            initial_variance=self.baseline_result.measurement_variance,
            process_noise_weight=kalman_config.get('process_noise_weight', 0.5),
            process_noise_trend=kalman_config.get('process_noise_trend', 0.01),
            measurement_noise=self.baseline_result.measurement_noise_std ** 2,
            outlier_threshold=kalman_config.get('outlier_threshold', 3.0),
            extreme_outlier_threshold=kalman_config.get('extreme_outlier_threshold', 5.0),
            innovation_window_size=kalman_config.get('innovation_window_size', 20),
            reset_gap_days=kalman_config.get('reset_gap_days', 30),
            reset_deviation_threshold=kalman_config.get('reset_deviation_threshold', 0.5),
            physiological_min=kalman_config.get('physiological_min_weight', 40.0),
            physiological_max=kalman_config.get('physiological_max_weight', 300.0)
        )

        # Step 3: Initialize adaptive validation gate
        self.validation_gate = AdaptiveValidationGate(
            base_gamma=self.config.get('validation_gamma', 3.0),
            strict_gamma=self.config.get('strict_gamma', 2.0),
            relaxed_gamma=self.config.get('relaxed_gamma', 4.0)
        )

        # Step 4: Process historical data through pipeline
        if self.config.get('process_historical', True):
            self._process_historical(measurements)

        self.is_initialized = True
        return self.baseline_result

    def process_measurement(self, measurement: WeightMeasurement) -> ProcessedMeasurement:
        """
        Process a single measurement through Layer 1 gates then robust Kalman filter.
        """
        if not self.is_initialized:
            return ProcessedMeasurement(
                measurement=measurement,
                is_valid=False,
                confidence=0.0,
                rejection_reason="Pipeline not initialized"
            )

        # Get source trust factor
        source_trust = self.source_trust.get(
            measurement.source_type or "unknown",
            self.source_trust["unknown"]
        )
        
        # Check if Kalman would reset due to time gap or deviation
        will_reset_kalman = False
        reset_reason = ""
        if self.kalman and self.kalman.last_timestamp:
            time_delta_days = (measurement.timestamp - self.kalman.last_timestamp).total_seconds() / 86400.0
            will_reset_kalman, reset_reason = self.kalman.should_reset_state(measurement.weight, time_delta_days)
        
        # Prepare state context for Layer 1
        state_context = self._prepare_layer1_context()
        
        # If Kalman will reset, inform Layer1 to be more lenient
        if will_reset_kalman:
            state_context['kalman_will_reset'] = True
            state_context['reset_reason'] = reset_reason
            logger.info(f"Kalman will reset: {reset_reason}")

        # LAYER 1 PRE-FILTERING: Check physiological plausibility
        layer1_valid, layer1_outlier_type, layer1_metadata = self.layer1.process(
            measurement, state_context
        )
        
        if not layer1_valid:
            self.layer1_rejections += 1
            logger.info(f"Layer 1 rejection: {measurement.weight}kg - {layer1_metadata.get('reason', 'Unknown')}")
            
            # Use Kalman's prediction as best estimate when Layer 1 rejects
            predicted_weight = state_context.get('predicted_weight', self.kalman.state[0])
            
            return ProcessedMeasurement(
                measurement=measurement,
                is_valid=False,
                confidence=0.0,  # Zero confidence for Layer 1 rejections
                predicted_weight=predicted_weight,
                filtered_weight=predicted_weight,
                trend_kg_per_day=self.kalman.state[1] if self.kalman else 0.0,
                rejection_reason=f"Layer1: {layer1_metadata.get('reason', 'Physiological limit violation')}",
                processing_metadata={
                    'layer1': layer1_metadata,
                    'layer1_outlier_type': layer1_outlier_type.value if layer1_outlier_type else None,
                    'source_trust': source_trust
                }
            )
        
        # ROBUST KALMAN FILTER: Process measurements that passed Layer 1
        kalman_results = self.kalman.process_measurement(
            measurement, 
            source_trust,
            robust_mode=True  # Enable robust outlier handling
        )
        
        predicted_weight = kalman_results['predicted_weight']
        outlier_type = kalman_results.get('outlier_type', 'none')
        normalized_innovation = kalman_results.get('normalized_innovation', 0)

        # Handle extreme outliers
        if outlier_type == 'extreme':
            self.extreme_outliers_detected += 1
            logger.info(f"Extreme outlier rejected: {measurement.weight}kg "
                       f"(predicted={predicted_weight:.1f}, {normalized_innovation:.1f}σ)")
            
            return ProcessedMeasurement(
                measurement=measurement,
                is_valid=False,
                confidence=0.05,
                predicted_weight=predicted_weight,
                filtered_weight=predicted_weight,  # Use prediction as best estimate
                trend_kg_per_day=kalman_results['trend_kg_per_day'],
                rejection_reason=f"Extreme outlier: {normalized_innovation:.1f}σ",
                processing_metadata={
                    'kalman': kalman_results,
                    'outlier_type': 'extreme',
                    'source_trust': source_trust
                }
            )
        
        # Handle moderate outliers (dampened update applied)
        elif outlier_type == 'moderate':
            self.outliers_detected += 1
            confidence = self._calculate_robust_confidence(normalized_innovation, outlier_type)
            
            logger.debug(f"Moderate outlier dampened: {measurement.weight}kg "
                        f"(predicted={predicted_weight:.1f}, {normalized_innovation:.1f}σ)")
        else:
            # Normal measurement
            confidence = self._calculate_robust_confidence(normalized_innovation, outlier_type)

        # Measurement processed (either normal or with dampened update)
        self.measurements_processed += 1

        result = ProcessedMeasurement(
            measurement=measurement,
            is_valid=True,
            confidence=confidence,
            filtered_weight=kalman_results['filtered_weight'],
            predicted_weight=predicted_weight,
            trend_kg_per_day=kalman_results['trend_kg_per_day'],
            processing_metadata={
                'kalman': kalman_results,
                'outlier_type': outlier_type,
                'source_trust': source_trust
            }
        )

        # Store in history for analysis
        self.processing_history.append(result)

        return result

    def _process_historical(self, measurements: List[WeightMeasurement]):
        """
        Process historical measurements to initialize Kalman filter.
        Optionally applies Kalman smoother for optimal trajectory.
        """
        # Sort by timestamp
        sorted_measurements = sorted(measurements, key=lambda m: m.timestamp)

        # Process through pipeline to build history
        for m in sorted_measurements:
            _ = self.process_measurement(m)

        logger.info(f"Processed {len(sorted_measurements)} historical measurements")

        # Optional: Apply Kalman smoother for optimal historical trajectory
        if self.config.get('apply_smoother', False):
            weights = [m.weight for m in sorted_measurements]
            timestamps = [m.timestamp for m in sorted_measurements]
            smoothed = self.kalman.smooth(weights, timestamps)
            logger.info("Applied Kalman smoother to historical data")

    def _calculate_robust_confidence(self, normalized_innovation: float, outlier_type: str) -> float:
        """
        Calculate confidence score with awareness of outlier type.
        """
        if outlier_type == 'extreme':
            return 0.05
        elif outlier_type == 'moderate':
            if normalized_innovation <= 3.5:
                return 0.4
            elif normalized_innovation <= 4.0:
                return 0.3
            else:
                return 0.2
        else:
            # Normal measurements
            if normalized_innovation <= 0.5:
                return 0.99
            elif normalized_innovation <= 1.0:
                return 0.95
            elif normalized_innovation <= 1.5:
                return 0.90
            elif normalized_innovation <= 2.0:
                return 0.80
            elif normalized_innovation <= 2.5:
                return 0.65
            elif normalized_innovation <= 3.0:
                return 0.50
            else:
                return 0.35

    def get_state_summary(self) -> Dict[str, Any]:
        """Get current pipeline state summary."""
        if not self.is_initialized:
            return {'initialized': False}

        kalman_state = self.kalman.get_state()

        return {
            'initialized': True,
            'baseline': {
                'weight': self.baseline_result.baseline_weight,
                'confidence': self.baseline_result.confidence,
                'variance': self.baseline_result.measurement_variance
            },
            'current_state': {
                'weight': kalman_state.weight,
                'trend_kg_per_day': kalman_state.trend,
                'trend_kg_per_week': kalman_state.trend * 7.0,
                'measurement_count': kalman_state.measurement_count
            },
            'measurements_processed': self.measurements_processed,
            'layer1_rejections': self.layer1_rejections,
            'outliers_detected': self.outliers_detected,
            'extreme_outliers_detected': self.extreme_outliers_detected,
            'total_rejections': self.layer1_rejections + self.extreme_outliers_detected,
            'outlier_rate': (self.outliers_detected + self.extreme_outliers_detected) / max(1, self.measurements_processed),
            'acceptance_rate': self._calculate_acceptance_rate()
        }

    def _prepare_layer1_context(self) -> Dict[str, Any]:
        """
        Prepare state context for Layer 1 filtering.
        For real-time processing, this would come from database/cache.
        """
        context = {}
        
        if self.kalman and self.kalman.last_timestamp:
            # Get last known state from Kalman
            context['last_known_state'] = {
                'weight': float(self.kalman.state[0]),
                'timestamp': self.kalman.last_timestamp,
                'measurement_count': self.kalman.measurement_count
            }
            
            # Get Kalman's prediction (if we can calculate it)
            try:
                predicted_weight, _ = self.kalman.predict(1.0)  # Default 1 day prediction
                context['predicted_weight'] = predicted_weight
            except:
                pass
        
        # Check if medical intervention mode should be enabled
        # (This could be a user setting in production)
        context['medical_intervention_mode'] = self.config.get('medical_intervention_mode', False)
        
        return context
    
    def _calculate_acceptance_rate(self) -> float:
        """Calculate measurement acceptance rate."""
        if not self.processing_history:
            return 0.0

        accepted = sum(1 for p in self.processing_history if p.is_valid)
        return accepted / len(self.processing_history)


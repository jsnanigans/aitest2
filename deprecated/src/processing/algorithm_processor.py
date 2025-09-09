#!/usr/bin/env python3

from typing import Dict, Any, Optional, List
import numpy as np
from src.filters import CustomKalmanFilter
from src.utils.prediction_utils import add_daily_prediction
from src.core import get_logger

logger = get_logger(__name__)


class KalmanProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enable_kalman = config.get("enable_kalman", True)
        self.kalman_filter = None
        self.kalman_results = []
        self.filter_initialized = False

        if self.enable_kalman:
            self.source_trust_config = config.get("kalman_source_trust", {
                "care-team-upload": {"trust": 0.95, "noise_scale": 0.3},
                "internal-questionnaire": {"trust": 0.8, "noise_scale": 0.5},
                "patient-upload": {"trust": 0.6, "noise_scale": 0.8},
                "https://connectivehealth.io": {"trust": 0.3, "noise_scale": 1.5},
                "https://api.iglucose.com": {"trust": 0.1, "noise_scale": 3.0},
                "patient-device": {"trust": 0.2, "noise_scale": 2.5},
                "unknown": {"trust": 0.5, "noise_scale": 1.0},
            })

    def initialize_filter(self, baseline_params: Optional[Dict[str, Any]] = None):
        if self.enable_kalman:
            init_params = {}
            if baseline_params and baseline_params.get('success'):
                init_params['initial_weight'] = baseline_params['baseline_weight']
                init_params['initial_variance'] = baseline_params['measurement_variance']
                logger.debug(f"Initializing Kalman with baseline: weight={baseline_params['baseline_weight']:.1f}, "
                           f"variance={baseline_params['measurement_variance']:.3f}")
            
            self.kalman_filter = CustomKalmanFilter(
                source_trust_config=self.source_trust_config,
                enable_validation=self.config.get("enable_kalman_validation", True),
                **init_params
            )
            self.kalman_results = []
            self.filter_initialized = True
    
    def reinitialize_filter(self, baseline_params: Dict[str, Any]):
        if self.enable_kalman and baseline_params and baseline_params.get('success'):
            logger.info(f"Reinitializing Kalman filter with new baseline after gap: weight={baseline_params['baseline_weight']:.1f}")
            
            # Use higher initial variance after gaps to allow adaptation
            # Baseline variance might be too low if established from few consistent readings
            baseline_variance = baseline_params['measurement_variance']
            min_variance_after_gap = 1.0  # Minimum 1 kg² variance after gaps
            initial_variance = max(baseline_variance, min_variance_after_gap)
            
            if initial_variance != baseline_variance:
                logger.info(f"Increased initial variance from {baseline_variance:.4f} to {initial_variance:.4f} for post-gap adaptation")
            
            # Always create a new filter when a gap triggers baseline re-establishment
            # This ensures clean state after significant data gaps
            self.kalman_filter = CustomKalmanFilter(
                initial_weight=baseline_params['baseline_weight'],
                initial_variance=initial_variance,
                initial_trend=0.0,
                source_trust_config=self.source_trust_config,
                enable_validation=self.config.get("enable_kalman_validation", True)
            )
            self.filter_initialized = True
            self.kalman_results = []  # Clear previous results for clean restart
            logger.info("Kalman filter reset with new baseline parameters")
            return True
        return False

    def process_measurement(self, weight: float, date, date_str: str, source_type: str) -> Optional[Dict[str, Any]]:
        if not self.enable_kalman or not self.kalman_filter:
            return None

        try:
            kalman_result = self.kalman_filter.process_measurement(
                weight, timestamp=date, source_type=source_type
            )
            if kalman_result:
                kalman_debug_entry = {
                    "date": date_str,
                    "measured_weight": weight,
                    "filtered_weight": kalman_result["filtered_weight"],
                    "predicted_weight": kalman_result["predicted_weight"],
                    "uncertainty": kalman_result.get(
                        "uncertainty_weight", kalman_result.get("uncertainty")
                    ),
                    "innovation": kalman_result["innovation"],
                    "normalized_innovation": kalman_result["normalized_innovation"],
                    "trend_kg_per_day": kalman_result.get("trend_kg_per_day"),
                    "trend_kg_per_week": kalman_result.get("trend_kg_per_week"),
                    "time_delta_days": kalman_result.get("time_delta_days", 1.0),
                    "measurement_count": kalman_result.get("measurement_count", 0),
                    "measurement_accepted": kalman_result.get("measurement_accepted", True),
                    "confidence": kalman_result.get("confidence", 0.5),
                    "filter_reinitialized": kalman_result.get("filter_reinitialized", False),
                }

                if "rejection_reason" in kalman_result:
                    kalman_debug_entry["rejection_reason"] = kalman_result["rejection_reason"]
                if "implied_velocity" in kalman_result:
                    kalman_debug_entry["implied_velocity"] = kalman_result["implied_velocity"]
                if "measurement_noise_used" in kalman_result:
                    kalman_debug_entry["measurement_noise_used"] = kalman_result["measurement_noise_used"]
                if "uncertainty_trend" in kalman_result:
                    kalman_debug_entry["uncertainty_trend"] = kalman_result["uncertainty_trend"]

                self.kalman_results.append(kalman_debug_entry)
                return kalman_result
        except Exception as e:
            logger.warning(f"Kalman filter error: {e}")
            return None

    def get_current_prediction(self, date_str) -> Dict[str, Any]:
        """Get Kalman prediction for a date without updating the filter state."""
        if not self.filter_initialized or not self.kalman_filter:
            return None
        
        try:
            # Parse date
            if isinstance(date_str, str):
                from datetime import datetime
                if 'T' in date_str:
                    measurement_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                else:
                    measurement_date = datetime.fromisoformat(date_str.replace(' ', 'T'))
            else:
                measurement_date = date_str
            
            # Get prediction from Kalman filter
            state = self.kalman_filter.get_state()
            if state:
                # For now, return the current state as prediction
                # In a more sophisticated implementation, we'd extrapolate based on time delta
                return {
                    'predicted_weight': state.get('weight', 0),
                    'uncertainty': state.get('uncertainty_weight', state.get('uncertainty', 0))
                }
        except Exception as e:
            logger.debug(f"Error getting prediction: {e}")
            return None
        
        return None
    
    def add_predictions(self, time_series: List[Dict[str, Any]], max_gap_days: int = 3) -> Dict[str, Any]:
        # Gap filling removed - predictions are no longer added
        return {}

    def create_summary(self) -> Dict[str, Any]:
        if not self.kalman_results:
            return {}

        innovations = [r["innovation"] for r in self.kalman_results]
        normalized = [r["normalized_innovation"] for r in self.kalman_results]
        uncertainties = [r["uncertainty"] for r in self.kalman_results]

        summary = {
            "filter_initialized": self.filter_initialized,
            "filter_type": "Custom 2D Trend",
            "filter_parameters": {
                "process_noise_weight": self.kalman_filter.base_process_noise_weight if self.kalman_filter else None,
                "process_noise_trend": self.kalman_filter.base_process_noise_trend if self.kalman_filter else None,
                "measurement_noise": self.kalman_filter.base_measurement_noise if self.kalman_filter else None,
                "max_reasonable_trend": self.kalman_filter.max_reasonable_trend if self.kalman_filter else None
            },
            "total_measurements": len(self.kalman_results),
            "innovation_stats": {
                "mean": round(float(np.mean(innovations)), 3),
                "std": round(float(np.std(innovations)), 3),
                "min": round(float(np.min(innovations)), 3),
                "max": round(float(np.max(innovations)), 3),
                "median": round(float(np.median(innovations)), 3),
                "abs_mean": round(float(np.mean(np.abs(innovations))), 3)
            },
            "normalized_innovation_stats": {
                "mean": round(float(np.mean(normalized)), 3),
                "std": round(float(np.std(normalized)), 3),
                "max": round(float(np.max(normalized)), 3),
                "percentile_95": round(float(np.percentile(normalized, 95)), 3),
                "outliers_3sigma": sum(1 for n in normalized if n > 3.0),
                "outliers_5sigma": sum(1 for n in normalized if n > 5.0)
            },
            "uncertainty_stats": {
                "mean": round(float(np.mean(uncertainties)), 3),
                "std": round(float(np.std(uncertainties)), 3),
                "min": round(float(np.min(uncertainties)), 3),
                "max": round(float(np.max(uncertainties)), 3),
                "trend": "increasing" if uncertainties[-1] > uncertainties[0] else "decreasing" if len(uncertainties) > 1 else "stable"
            }
        }

        if self.kalman_results and "trend_kg_per_day" in self.kalman_results[0]:
            trends_per_day = [
                r.get("trend_kg_per_day", 0) for r in self.kalman_results if "trend_kg_per_day" in r
            ]
            if trends_per_day:
                summary["mean_trend_kg_per_day"] = round(float(np.mean(trends_per_day)), 4)
                summary["std_trend_kg_per_day"] = round(float(np.std(trends_per_day)), 4)
                summary["max_trend_kg_per_day"] = round(float(np.max(np.abs(trends_per_day))), 4)

        if self.kalman_filter:
            final_state = self.kalman_filter.get_state()
            if final_state:
                summary["final_filtered_weight"] = round(final_state["weight"], 2)
                if "uncertainty_weight" in final_state:
                    summary["final_uncertainty"] = round(final_state["uncertainty_weight"], 3)
                else:
                    summary["final_uncertainty"] = round(final_state.get("uncertainty", 0), 3)
            
            # Add validation metrics if available
            validation_summary = self.kalman_filter.get_validation_summary()
            if validation_summary:
                summary["validation"] = {
                    "metrics": validation_summary["metrics"],
                    "total_rejected": validation_summary["total_rejected"],
                    "should_rebaseline": validation_summary["should_rebaseline"]
                }

                if "trend_kg_per_day" in final_state:
                    summary["final_trend_kg_per_day"] = round(final_state["trend_kg_per_day"], 4)
                    summary["final_trend_kg_per_week"] = round(final_state["trend_kg_per_week"], 3)
                    summary["trend_direction"] = final_state.get("trend_direction", "unknown")
                    if "trend_statistics" in final_state:
                        summary["trend_statistics"] = final_state["trend_statistics"]

        outlier_count = sum(1 for n in normalized if n > 3.0)
        summary["kalman_outliers"] = outlier_count
        summary["kalman_outlier_rate"] = round(outlier_count / len(normalized), 3) if normalized else 0

        return summary

    def get_filter_debug_state(self) -> Dict[str, Any]:
        debug_state = {}

        if self.kalman_filter:
            filter_state = self.kalman_filter.get_state()
            if filter_state:
                debug_state["kalman_final_state"] = filter_state

            if hasattr(self.kalman_filter, 'innovation_history'):
                debug_state["kalman_innovation_history"] = [
                    {
                        "innovation": round(ih["innovation"], 3),
                        "normalized": round(ih["normalized"], 3),
                        "implied_velocity": round(ih["implied_velocity"], 4)
                    }
                    for ih in self.kalman_filter.innovation_history[-10:]
                ]

            if hasattr(self.kalman_filter, 'trend_history'):
                debug_state["kalman_trend_history"] = [
                    {
                        "trend": round(th["trend"], 4),
                        "measurement_count": th["measurement_count"]
                    }
                    for th in self.kalman_filter.trend_history[-10:]
                ]

            future_prediction = self.kalman_filter.predict_future(days_ahead=7)
            if future_prediction:
                debug_state["kalman_7day_prediction"] = {
                    "current_weight": round(future_prediction["current_weight"], 2),
                    "predicted_weight_7d": round(future_prediction["predictions"][-1]["weight"], 2),
                    "predicted_change_7d": round(future_prediction["total_predicted_change"], 2),
                    "max_uncertainty": round(future_prediction["max_uncertainty"], 3),
                    "trend_kg_per_week": round(future_prediction["current_trend_kg_per_day"] * 7, 3)
                }

        return debug_state

#!/usr/bin/env python3

from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
from src.core import get_logger
from src.processing.baseline_establishment import RobustBaselineEstimator

logger = get_logger(__name__)


class UserProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Parse date filters from config
        min_date_str = config.get("min_date", None)
        if min_date_str:
            try:
                self.min_date = datetime.strptime(min_date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                self.min_date = None
        else:
            self.min_date = None
            
        max_date_str = config.get("max_date", None)
        if max_date_str:
            try:
                self.max_date = datetime.strptime(max_date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                self.max_date = None
        else:
            self.max_date = None
            
        self.source_type_trust_scores = config.get(
            "source_type_trust_scores",
            {
                "care-team-upload": 0.5,
                "internal-questionnaire": 0.3,
                "patient-upload": 0.3,
                "https://connectivehealth.io": 0.1,
                "https://api.iglucose.com": 0.05,
                "patient-device": 0.05,
                "unknown": 0.5,
            },
        )

        self.confidence_thresholds = config.get(
            "confidence_thresholds",
            {
                "normal": 3.0,
                "small": 5.0,
                "moderate": 10.0,
                "significant": 15.0,
                "large": 20.0,
                "major": 30.0,
                "extreme": 50.0,
            },
        )

        self.weight_validation = config.get(
            "weight_validation",
            {"reasonable_min": 30, "reasonable_max": 250, "possible_min": 20, "possible_max": 300},
        )

        self.config = config
        self.baseline_estimator = RobustBaselineEstimator(config)
        self.baseline_enabled = config.get("enable_baseline", True)
        self.baseline_gap_threshold_days = config.get("baseline_gap_threshold_days", 30)
        self.enable_baseline_gaps = config.get("enable_baseline_gaps", True)

        self.reset()

    def reset(self):
        self.user_id = None
        self.readings = []
        self.stats = {}
        self.signup_date = None
        self.baseline_window_readings = []
        self.baseline_result = None
        self.last_reading_date = None
        self.baseline_state = 'NORMAL'
        self.baseline_window_start = None
        self.baseline_history = []
        self.current_gap_days = 0
        self.baseline_attempts = 0
        self.max_baseline_attempts = 10

    def start_user(self, user_id: str):
        self.reset()
        self.user_id = user_id
        self.stats = {
            "user_id": user_id,
            "total_readings": 0,
            "outliers": 0,
            "confidence_scores": [],
            "min_weight": None,
            "max_weight": None,
            "first_date": None,
            "last_date": None,
            "processing_timestamp": datetime.now().isoformat(),
        }

    def validate_weight(self, weight_str: str) -> Optional[float]:
        if not weight_str or weight_str == "NULL" or weight_str == "None":
            return None

        try:
            weight = float(weight_str)
        except (ValueError, TypeError):
            return None

        if weight < 30 or weight > 300:
            return None

        return weight

    def validate_date(self, date_str: str) -> Optional[datetime]:
        date = self._parse_datetime(date_str)
        
        # Apply date filters if configured
        if self.min_date and date < self.min_date:
            return None
        if self.max_date and date > self.max_date:
            return None
            
        return date

    def detect_gap(self, current_date: datetime, last_date: Optional[datetime]) -> bool:
        if not last_date:
            self.current_gap_days = 0
            return False
        gap_days = (current_date - last_date).days
        self.current_gap_days = gap_days
        
        # Log significant gaps even if below threshold
        if gap_days >= 14:
            logger.info(f"Significant gap detected: {gap_days} days")
        
        return gap_days >= self.baseline_gap_threshold_days

    def process_reading(self, reading: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        weight = self.validate_weight(reading.get("weight", ""))
        if weight is None:
            return None

        date_str = reading["effectivDateTime"]
        date = self.validate_date(date_str)
        if date is None:
            return None

        source_type = reading.get("source_type", "unknown")

        if self.enable_baseline_gaps and self.baseline_enabled:
            if self.baseline_state == 'NORMAL':
                if self.detect_gap(date, self.last_reading_date):
                    self.baseline_state = 'COLLECTING_BASELINE'
                    self.baseline_window_start = date
                    self.baseline_window_readings = []
                    self.baseline_attempts = 0
                    logger.info(f"User {self.user_id}: Gap of {self.current_gap_days} days detected, starting baseline collection")

            if self.baseline_state in ['COLLECTING_BASELINE', 'BASELINE_PENDING']:
                self.baseline_window_readings.append({
                    'weight': weight,
                    'date': date,
                    'source_type': source_type
                })

        if not self.signup_date:
            self.signup_date = date
            self.stats["signup_date"] = date_str
            self.stats["signup_weight"] = weight

        self.stats["total_readings"] += 1

        if self.stats["min_weight"] is None or weight < self.stats["min_weight"]:
            self.stats["min_weight"] = weight
        if self.stats["max_weight"] is None or weight > self.stats["max_weight"]:
            self.stats["max_weight"] = weight

        if self.stats["first_date"] is None:
            self.stats["first_date"] = date_str
        self.stats["last_date"] = date_str

        confidence = self._calculate_confidence(weight, source_type)
        self.stats["confidence_scores"].append(confidence)

        if confidence < 0.5:
            self.stats["outliers"] += 1
            if "outlier_details" not in self.stats:
                self.stats["outlier_details"] = []
            if len(self.stats["outlier_details"]) < 10:
                outlier_info = {
                    "date": date_str,
                    "weight": weight,
                    "confidence": confidence,
                    "source_type": source_type
                }
                if len(self.readings) > 0:
                    prev_weight = self.readings[-1]["weight"]
                    outlier_info["change_from_prev"] = round(weight - prev_weight, 2)
                    outlier_info["change_pct"] = round((weight - prev_weight) / prev_weight * 100, 2)
                self.stats["outlier_details"].append(outlier_info)

        reading_data = {
            "weight": weight,
            "date": date_str,
            "confidence": confidence,
            "source_type": source_type,
        }

        self.readings.append(reading_data)

        self.last_reading_date = date

        processed_reading = {
            "weight": weight,
            "date": date,
            "date_str": date_str,
            "source_type": source_type,
            "confidence": confidence,
            "gap_days": self.current_gap_days  # Include gap information
        }

        if self.baseline_state in ['COLLECTING_BASELINE', 'BASELINE_PENDING']:
            window_days = (date - self.baseline_window_start).days
            min_readings = self.config.get('baseline_min_readings', 3)

            # Try to establish baseline if we have enough readings
            should_try_baseline = False

            if self.baseline_state == 'BASELINE_PENDING':
                # In pending state, try with each new reading
                should_try_baseline = True
                self.baseline_attempts += 1
            elif len(self.baseline_window_readings) >= min_readings:
                # Have minimum readings, try to establish
                should_try_baseline = True
            elif window_days >= self.config.get('baseline_window_days', 7) and len(self.baseline_window_readings) >= 2:
                # Window time reached, try with what we have (but need at least 2)
                should_try_baseline = True

            if should_try_baseline:
                baseline_result = self._establish_gap_baseline()
                if baseline_result and baseline_result.get('success'):
                    processed_reading['new_baseline'] = baseline_result
                    self.baseline_state = 'NORMAL'
                    logger.info(f"User {self.user_id}: New baseline established after {self.current_gap_days} day gap (attempt {self.baseline_attempts + 1})")
                elif len(self.baseline_window_readings) >= self.config.get('baseline_max_readings', 30):
                    # Max readings reached, give up
                    self.baseline_state = 'NORMAL'
                    logger.debug(f"User {self.user_id}: Failed to establish baseline after {len(self.baseline_window_readings)} readings, continuing without it")
                elif window_days >= self.config.get('baseline_max_window_days', 14):
                    # Max window reached, give up
                    self.baseline_state = 'NORMAL'
                    logger.debug(f"User {self.user_id}: Failed to establish baseline after {window_days} days, continuing without it")
                elif self.baseline_attempts >= self.max_baseline_attempts:
                    # Too many attempts, give up
                    self.baseline_state = 'NORMAL'
                    logger.debug(f"User {self.user_id}: Failed to establish baseline after {self.baseline_attempts} attempts, continuing without it")
                else:
                    # Keep trying - enter or stay in pending state
                    if self.baseline_state == 'COLLECTING_BASELINE':
                        self.baseline_state = 'BASELINE_PENDING'
                        logger.info(f"User {self.user_id}: Baseline failed with {len(self.baseline_window_readings)} readings, entering pending state for retry")

        return processed_reading

    def _calculate_confidence(self, weight: float, source_type: str) -> float:
        bonus = 0.1 if source_type == "internal-questionnaire" else 0

        if len(self.readings) >= 2:
            recent = [r["weight"] for r in self.readings[-10:]]
            ref_weight = sum(recent) / len(recent)
        else:
            reasonable_min = self.weight_validation["reasonable_min"]
            reasonable_max = self.weight_validation["reasonable_max"]
            possible_min = self.weight_validation["possible_min"]
            possible_max = self.weight_validation["possible_max"]

            if reasonable_min <= weight <= reasonable_max:
                return min(0.9 + bonus, 1.0)
            elif possible_min <= weight <= possible_max:
                return min(0.6 + bonus, 1.0)
            else:
                return 0.2

        if ref_weight == 0:
            return 0.5

        change_percent = abs(weight - ref_weight) / ref_weight * 100

        thresholds = self.confidence_thresholds
        if change_percent < thresholds["normal"]:
            base_conf = 0.95
        elif change_percent < thresholds["small"]:
            base_conf = 0.90
        elif change_percent < thresholds["moderate"]:
            base_conf = 0.75
        elif change_percent < thresholds["significant"]:
            base_conf = 0.60
        elif change_percent < thresholds["large"]:
            base_conf = 0.45
        elif change_percent < thresholds["major"]:
            base_conf = 0.30
        elif change_percent < thresholds["extreme"]:
            base_conf = 0.15
        else:
            base_conf = 0.05

        return min(base_conf + bonus, 1.0)

    def _parse_datetime(self, dt_string: str) -> datetime:
        try:
            return datetime.fromisoformat(dt_string.replace("Z", "+00:00"))
        except:
            try:
                return datetime.strptime(dt_string, "%Y-%m-%d %H:%M:%S")
            except:
                return datetime.strptime(dt_string.split(".")[0], "%Y-%m-%dT%H:%M:%S")

    def _establish_gap_baseline(self) -> Optional[Dict[str, Any]]:
        if not self.baseline_window_readings:
            return None

        baseline_result = self.baseline_estimator.establish_baseline(
            self.baseline_window_readings,
            window_start_date=self.baseline_window_start
        )

        if baseline_result.get('success'):
            baseline_entry = {
                'timestamp': datetime.now().isoformat(),
                'trigger_date': self.baseline_window_start.isoformat() if self.baseline_window_start else None,
                'gap_days': self.current_gap_days,
                'weight': baseline_result['baseline_weight'],
                'variance': baseline_result['measurement_variance'],
                'std': baseline_result['measurement_noise_std'],
                'readings_used': baseline_result['readings_used'],
                'confidence': baseline_result['confidence']
            }
            self.baseline_history.append(baseline_entry)

            if not self.stats.get('baseline_history'):
                self.stats['baseline_history'] = []
            self.stats['baseline_history'].append(baseline_entry)

            return baseline_result
        return None

    def _establish_baseline(self):

        readings_for_baseline = []
        for reading in self.readings:
            readings_for_baseline.append({
                'weight': reading['weight'],
                'date': self._parse_datetime(reading['date']),
                'source_type': reading['source_type'],
                'confidence': reading['confidence']
            })

        # First try with signup date if available
        baseline_result = self.baseline_estimator.establish_baseline(
            readings_for_baseline,
            signup_date=self.signup_date
        )

        # If failed and we have readings, try alternative approaches
        if not baseline_result.get('success') and len(readings_for_baseline) >= 3:
            error_msg = baseline_result.get('error', '')

            # If failed due to insufficient readings in window, try without signup date
            # This will use the first N readings instead
            if 'Insufficient readings' in error_msg or 'No readings provided' in error_msg:
                logger.info(f"User {self.user_id}: Initial baseline failed, trying with first readings approach")
                baseline_result = self.baseline_estimator.establish_baseline(
                    readings_for_baseline,
                    signup_date=None  # This makes it use first N readings
                )

        if baseline_result.get('success'):
            self.baseline_result = baseline_result
            self.stats['baseline_established'] = True
            self.stats['baseline_weight'] = baseline_result['baseline_weight']
            self.stats['baseline_variance'] = baseline_result['measurement_variance']
            self.stats['baseline_std'] = baseline_result['measurement_noise_std']
            self.stats['baseline_mad'] = baseline_result['mad']
            self.stats['baseline_readings_count'] = baseline_result['readings_used']
            self.stats['baseline_confidence'] = baseline_result['confidence']
            self.stats['baseline_outliers_removed'] = baseline_result['outliers_removed']
            self.stats['baseline_method'] = baseline_result['method']

            if 'percentiles' in baseline_result:
                self.stats['baseline_percentiles'] = baseline_result['percentiles']

            quality_check = self.baseline_estimator.validate_baseline_quality(baseline_result)
            self.stats['baseline_quality'] = quality_check

            if baseline_result.get('iqr_fences'):
                self.stats['baseline_iqr_fences'] = baseline_result['iqr_fences']

            if self.readings:
                deviations = []
                for reading in self.readings:
                    deviation = abs(reading['weight'] - baseline_result['baseline_weight'])
                    deviations.append(deviation)
                self.stats['average_deviation_from_baseline'] = round(np.mean(deviations), 2)
                self.stats['max_deviation_from_baseline'] = round(max(deviations), 2)

            logger.info(f"User {self.user_id}: Baseline established at {baseline_result['baseline_weight']:.1f} kg "
                       f"(confidence: {baseline_result['confidence']}, readings: {baseline_result['readings_used']})")
        else:
            self.stats['baseline_established'] = False
            self.stats['baseline_error'] = baseline_result.get('error', 'Unknown error')
            logger.debug(f"User {self.user_id}: Baseline not established - {baseline_result.get('error')}")

    def finalize_stats(self) -> Dict[str, Any]:
        if not self.user_id:
            return {"total_readings": 0}

        if self.baseline_enabled and self.readings and len(self.readings) >= 3:
            self._establish_baseline()

        if self.stats.get("confidence_scores"):
            self.stats["average_confidence"] = round(
                sum(self.stats["confidence_scores"]) / len(self.stats["confidence_scores"]), 3
            )
            self.stats["confidence_std"] = round(float(np.std(self.stats["confidence_scores"])), 3)
            self.stats["confidence_min"] = round(min(self.stats["confidence_scores"]), 3)
            self.stats["confidence_max"] = round(max(self.stats["confidence_scores"]), 3)

        if self.stats["min_weight"] and self.stats["max_weight"]:
            self.stats["weight_range"] = round(
                self.stats["max_weight"] - self.stats["min_weight"], 2
            )

        if self.readings:
            weights = [r["weight"] for r in self.readings]
            self.stats["weight_mean"] = round(float(np.mean(weights)), 2)
            self.stats["weight_std"] = round(float(np.std(weights)), 2)
            self.stats["weight_variance"] = round(float(np.var(weights)), 3)
            self.stats["weight_cv"] = round(float(np.std(weights) / np.mean(weights) * 100), 2) if np.mean(weights) > 0 else 0

        source_counts = {}
        for reading in self.readings:
            src = reading.get("source_type", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        self.stats["source_type_breakdown"] = source_counts

        if self.stats["first_date"] and self.stats["last_date"]:
            first_date = self._parse_datetime(self.stats["first_date"])
            last_date = self._parse_datetime(self.stats["last_date"])
            self.stats["days_spanned"] = (last_date - first_date).days
            self.stats["readings_per_day"] = round(
                self.stats["total_readings"] / max(1, self.stats["days_spanned"]), 2
            )

        self.stats["time_series"] = []
        for i, reading in enumerate(self.readings):
            ts_point = {
                "index": i,
                "date": reading["date"],
                "weight": reading["weight"],
                "confidence": reading["confidence"],
                "source": reading["source_type"],
            }

            if i > 0:
                prev_weight = self.readings[i-1]["weight"]
                ts_point["weight_change"] = round(reading["weight"] - prev_weight, 2)
                ts_point["weight_change_pct"] = round((reading["weight"] - prev_weight) / prev_weight * 100, 2)

                prev_date = self._parse_datetime(self.readings[i-1]["date"])
                curr_date = self._parse_datetime(reading["date"])
                ts_point["days_since_last"] = round((curr_date - prev_date).total_seconds() / 86400, 2)

            self.stats["time_series"].append(ts_point)

        if self.readings:
            weights = [r["weight"] for r in self.readings]
            sorted_weights = sorted(weights)
            n = len(sorted_weights)
            self.stats["percentiles"] = {
                "p5": sorted_weights[int(n * 0.05)] if n > 20 else sorted_weights[0],
                "p25": sorted_weights[int(n * 0.25)] if n > 4 else sorted_weights[0],
                "p50": sorted_weights[int(n * 0.50)],
                "p75": sorted_weights[int(n * 0.75)] if n > 4 else sorted_weights[-1],
                "p95": sorted_weights[int(n * 0.95)] if n > 20 else sorted_weights[-1],
            }

        if self.stats["time_series"]:
            actual_weights = [ts["weight"] for ts in self.stats["time_series"]]
            self.stats["moving_averages"] = []
            window_size = min(7, len(actual_weights))
            for i in range(len(actual_weights)):
                start_idx = max(0, i - window_size + 1)
                window = actual_weights[start_idx : i + 1]
                ma = sum(window) / len(window)
                self.stats["moving_averages"].append(round(ma, 2))

        return self.stats

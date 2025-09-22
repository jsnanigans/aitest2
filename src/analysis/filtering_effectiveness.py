"""
Filtering effectiveness analysis module.
Analyzes the impact of data filtering on weight measurements.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats
from scipy.stats import kurtosis, skew
import pandas as pd

from ..constants import PHYSIOLOGICAL_LIMITS, SOURCE_PROFILES
from ..processing.unified_quality_scorer import UnifiedQualityScorer, MeasurementHistory
from ..processing.outlier_detection import OutlierDetector
from ..processing.kalman import KalmanFilterManager, ResetManager

logger = logging.getLogger(__name__)


@dataclass
class DistributionMetrics:
    """Statistical distribution metrics for data analysis."""

    # Central tendency
    mean: float
    median: float
    mode: Optional[float]

    # Dispersion
    std: float
    variance: float
    iqr: float
    mad: float  # Median Absolute Deviation
    cv: float  # Coefficient of Variation

    # Shape
    skewness: float
    kurtosis: float

    # Range
    min_val: float
    max_val: float
    range_val: float

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "mean": self.mean,
            "median": self.median,
            "mode": self.mode,
            "std": self.std,
            "variance": self.variance,
            "iqr": self.iqr,
            "mad": self.mad,
            "cv": self.cv,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "min": self.min_val,
            "max": self.max_val,
            "range": self.range_val
        }


@dataclass
class OutlierMetrics:
    """Metrics for outlier detection effectiveness."""

    total_points: int
    outliers_removed: int
    outlier_rate: float

    # By method
    iqr_outliers: int
    z_score_outliers: int
    mad_outliers: int
    temporal_outliers: int

    # Characteristics
    outlier_magnitudes: List[float] = field(default_factory=list)
    outlier_timestamps: List[datetime] = field(default_factory=list)
    outlier_sources: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "total_points": self.total_points,
            "outliers_removed": self.outliers_removed,
            "outlier_rate": self.outlier_rate,
            "by_method": {
                "iqr": self.iqr_outliers,
                "z_score": self.z_score_outliers,
                "mad": self.mad_outliers,
                "temporal": self.temporal_outliers
            },
            "magnitude_stats": {
                "mean": np.mean(self.outlier_magnitudes) if self.outlier_magnitudes else 0,
                "std": np.std(self.outlier_magnitudes) if self.outlier_magnitudes else 0,
                "max": max(self.outlier_magnitudes) if self.outlier_magnitudes else 0
            },
            "source_distribution": self.outlier_sources
        }


@dataclass
class TemporalMetrics:
    """Temporal consistency metrics."""

    # Daily changes
    max_daily_change: float
    impossible_changes: int  # >2kg/day
    daily_volatility: float  # Rolling std of daily changes

    # Trend analysis
    trend_correlation: float  # Between raw and filtered
    smoothness_score: float  # Second derivative analysis
    inflection_points: int  # Trend reversals

    # Gap handling
    gap_count: int
    avg_gap_duration: float
    max_gap_duration: float

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "daily_change": {
                "max": self.max_daily_change,
                "impossible_count": self.impossible_changes,
                "volatility": self.daily_volatility
            },
            "trend": {
                "correlation": self.trend_correlation,
                "smoothness": self.smoothness_score,
                "inflection_points": self.inflection_points
            },
            "gaps": {
                "count": self.gap_count,
                "avg_duration_days": self.avg_gap_duration,
                "max_duration_days": self.max_gap_duration
            }
        }


@dataclass
class MedicalImpactMetrics:
    """Metrics for medical decision impact."""

    # Weight change accuracy
    start_point_variance: float
    end_point_variance: float
    total_change_delta: float

    # Clinical thresholds
    misclassification_rate: float  # % crossing 5% threshold differently
    direction_errors: int  # Gain/loss direction changes

    # Magnitude errors
    minor_errors: int  # <1kg difference
    moderate_errors: int  # 1-3kg difference
    severe_errors: int  # >3kg difference

    # Confidence
    confidence_interval_reduction: float  # % reduction in CI width

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "accuracy": {
                "start_variance": self.start_point_variance,
                "end_variance": self.end_point_variance,
                "change_delta": self.total_change_delta
            },
            "clinical": {
                "misclassification_rate": self.misclassification_rate,
                "direction_errors": self.direction_errors
            },
            "magnitude_errors": {
                "minor": self.minor_errors,
                "moderate": self.moderate_errors,
                "severe": self.severe_errors
            },
            "confidence": {
                "ci_reduction": self.confidence_interval_reduction
            }
        }


@dataclass
class ReportingMetrics:
    """Quarterly reporting impact metrics."""

    # Cohort statistics
    raw_cohort_mean: float
    filtered_cohort_mean: float
    mean_difference: float
    mean_percent_change: float

    # Success rates
    pct_losing_5pct_raw: float
    pct_losing_5pct_filtered: float
    pct_losing_10pct_raw: float
    pct_losing_10pct_filtered: float

    # User inclusion
    valid_baseline_raw: int
    valid_baseline_filtered: int
    valid_endpoint_raw: int
    valid_endpoint_filtered: int

    # Statistical power
    variance_reduction: float
    effect_size_improvement: float

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "cohort": {
                "raw_mean": self.raw_cohort_mean,
                "filtered_mean": self.filtered_cohort_mean,
                "difference": self.mean_difference,
                "percent_change": self.mean_percent_change
            },
            "success_rates": {
                "5pct_loss": {
                    "raw": self.pct_losing_5pct_raw,
                    "filtered": self.pct_losing_5pct_filtered
                },
                "10pct_loss": {
                    "raw": self.pct_losing_10pct_raw,
                    "filtered": self.pct_losing_10pct_filtered
                }
            },
            "inclusion": {
                "baseline": {
                    "raw": self.valid_baseline_raw,
                    "filtered": self.valid_baseline_filtered
                },
                "endpoint": {
                    "raw": self.valid_endpoint_raw,
                    "filtered": self.valid_endpoint_filtered
                }
            },
            "power": {
                "variance_reduction": self.variance_reduction,
                "effect_size_improvement": self.effect_size_improvement
            }
        }


class FilteringAnalyzer:
    """
    Comprehensive filtering effectiveness analyzer.
    Compares raw and filtered weight data to quantify improvements.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize analyzer with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.quality_scorer = UnifiedQualityScorer(config)
        self.outlier_detector = OutlierDetector(config)

    def analyze_user_data(
        self,
        user_id: str,
        raw_data: pd.DataFrame,
        filtered_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze filtering effectiveness for a single user.

        Args:
            user_id: User identifier
            raw_data: DataFrame with raw measurements
            filtered_data: DataFrame with filtered measurements

        Returns:
            Dictionary with all metrics
        """
        results = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "data_summary": self._get_data_summary(raw_data, filtered_data)
        }

        # Calculate all metric categories
        distribution_metrics = self._calculate_distribution_metrics(raw_data, filtered_data)
        outlier_metrics = self._calculate_outlier_metrics(raw_data, filtered_data)
        temporal_metrics = self._calculate_temporal_metrics(raw_data, filtered_data)
        medical_impact_metrics = self._calculate_medical_impact(raw_data, filtered_data)
        source_analysis = self._analyze_by_source(raw_data, filtered_data)

        # Convert to dictionaries for JSON serialization
        results["distribution"] = {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in distribution_metrics.items()}
        results["outliers"] = outlier_metrics.to_dict()
        results["temporal"] = temporal_metrics.to_dict()
        results["medical_impact"] = medical_impact_metrics.to_dict()
        results["source_analysis"] = source_analysis

        return results

    def analyze_cohort_data(
        self,
        cohort_raw: Dict[str, pd.DataFrame],
        cohort_filtered: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Analyze filtering effectiveness for a cohort of users.

        Args:
            cohort_raw: Dictionary of user_id -> raw DataFrame
            cohort_filtered: Dictionary of user_id -> filtered DataFrame

        Returns:
            Dictionary with cohort-level metrics
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "cohort_size": len(cohort_raw),
            "users": []
        }

        # Analyze each user
        for user_id in cohort_raw:
            if user_id in cohort_filtered:
                user_metrics = self.analyze_user_data(
                    user_id,
                    cohort_raw[user_id],
                    cohort_filtered[user_id]
                )
                results["users"].append(user_metrics)

        # Calculate cohort-level metrics
        results["reporting"] = self._calculate_reporting_metrics(cohort_raw, cohort_filtered)
        results["aggregate"] = self._aggregate_user_metrics(results["users"])

        return results

    def _get_data_summary(self, raw_df: pd.DataFrame, filtered_df: pd.DataFrame) -> Dict:
        """Get basic data summary statistics."""
        return {
            "raw": {
                "count": len(raw_df),
                "date_range": {
                    "start": raw_df['timestamp'].min().isoformat() if not raw_df.empty else None,
                    "end": raw_df['timestamp'].max().isoformat() if not raw_df.empty else None
                }
            },
            "filtered": {
                "count": len(filtered_df),
                "date_range": {
                    "start": filtered_df['timestamp'].min().isoformat() if not filtered_df.empty else None,
                    "end": filtered_df['timestamp'].max().isoformat() if not filtered_df.empty else None
                }
            },
            "removal_rate": 1 - (len(filtered_df) / len(raw_df)) if len(raw_df) > 0 else 0
        }

    def _calculate_distribution_metrics(
        self,
        raw_df: pd.DataFrame,
        filtered_df: pd.DataFrame
    ) -> Dict[str, DistributionMetrics]:
        """Calculate distribution metrics for raw and filtered data."""
        metrics = {}

        for name, df in [("raw", raw_df), ("filtered", filtered_df)]:
            if df.empty:
                continue

            weights = df['weight'].values

            # Calculate mode (most frequent value)
            mode_result = stats.mode(weights, keepdims=False)
            mode_val = mode_result.mode if hasattr(mode_result, 'mode') else None

            # Calculate IQR
            q1, q3 = np.percentile(weights, [25, 75])
            iqr = q3 - q1

            # Calculate MAD
            mad = np.median(np.abs(weights - np.median(weights)))

            # Calculate metrics
            mean_val = np.mean(weights)
            std_val = np.std(weights)

            metrics[name] = DistributionMetrics(
                mean=mean_val,
                median=np.median(weights),
                mode=float(mode_val) if mode_val is not None else None,
                std=std_val,
                variance=np.var(weights),
                iqr=iqr,
                mad=mad,
                cv=std_val / mean_val if mean_val != 0 else 0,
                skewness=skew(weights),
                kurtosis=kurtosis(weights),
                min_val=np.min(weights),
                max_val=np.max(weights),
                range_val=np.max(weights) - np.min(weights)
            )

        # Calculate improvements
        if "raw" in metrics and "filtered" in metrics:
            metrics["improvement"] = {
                "std_reduction": (metrics["raw"].std - metrics["filtered"].std) / metrics["raw"].std if metrics["raw"].std > 0 else 0,
                "iqr_compression": (metrics["raw"].iqr - metrics["filtered"].iqr) / metrics["raw"].iqr if metrics["raw"].iqr > 0 else 0,
                "mad_improvement": (metrics["raw"].mad - metrics["filtered"].mad) / metrics["raw"].mad if metrics["raw"].mad > 0 else 0,
                "cv_reduction": (metrics["raw"].cv - metrics["filtered"].cv) / metrics["raw"].cv if metrics["raw"].cv > 0 else 0,
                "skewness_correction": abs(metrics["filtered"].skewness) - abs(metrics["raw"].skewness),
                "kurtosis_normalization": abs(metrics["filtered"].kurtosis - 3) - abs(metrics["raw"].kurtosis - 3)
            }

        return metrics

    def _calculate_outlier_metrics(
        self,
        raw_df: pd.DataFrame,
        filtered_df: pd.DataFrame
    ) -> OutlierMetrics:
        """Calculate outlier detection metrics."""
        if raw_df.empty:
            return OutlierMetrics(
                total_points=0,
                outliers_removed=0,
                outlier_rate=0,
                iqr_outliers=0,
                z_score_outliers=0,
                mad_outliers=0,
                temporal_outliers=0
            )

        # Identify removed points
        if not filtered_df.empty:
            # Find measurements that were removed
            raw_set = set(zip(raw_df['timestamp'], raw_df['weight']))
            filtered_set = set(zip(filtered_df['timestamp'], filtered_df['weight']))
            removed = raw_set - filtered_set
        else:
            removed = set(zip(raw_df['timestamp'], raw_df['weight']))

        outliers_removed = len(removed)

        # Analyze outliers by different methods
        weights = raw_df['weight'].values

        # IQR method
        q1, q3 = np.percentile(weights, [25, 75])
        iqr = q3 - q1
        iqr_outliers = np.sum((weights < q1 - 1.5 * iqr) | (weights > q3 + 1.5 * iqr))

        # Z-score method
        z_scores = np.abs(stats.zscore(weights))
        z_score_outliers = np.sum(z_scores > 3)

        # MAD method
        mad = np.median(np.abs(weights - np.median(weights)))
        if mad > 0:
            modified_z = 0.6745 * (weights - np.median(weights)) / mad
            mad_outliers = np.sum(np.abs(modified_z) > 3.5)
        else:
            mad_outliers = 0

        # Temporal outliers (large day-to-day changes)
        if len(raw_df) > 1:
            raw_sorted = raw_df.sort_values('timestamp')
            daily_changes = np.abs(np.diff(raw_sorted['weight'].values))
            temporal_outliers = np.sum(daily_changes > PHYSIOLOGICAL_LIMITS['MAX_DAILY_CHANGE_KG'])
        else:
            temporal_outliers = 0

        # Analyze outlier characteristics
        outlier_magnitudes = []
        outlier_timestamps = []
        outlier_sources = {}

        if not filtered_df.empty and outliers_removed > 0:
            median_weight = np.median(filtered_df['weight'].values)
            for ts, w in removed:
                # Find the original row
                row = raw_df[(raw_df['timestamp'] == ts) & (raw_df['weight'] == w)]
                if not row.empty:
                    outlier_magnitudes.append(abs(w - median_weight))
                    outlier_timestamps.append(ts)
                    source = row.iloc[0].get('source', 'unknown')
                    outlier_sources[source] = outlier_sources.get(source, 0) + 1

        return OutlierMetrics(
            total_points=len(raw_df),
            outliers_removed=outliers_removed,
            outlier_rate=outliers_removed / len(raw_df) if len(raw_df) > 0 else 0,
            iqr_outliers=int(iqr_outliers),
            z_score_outliers=int(z_score_outliers),
            mad_outliers=int(mad_outliers),
            temporal_outliers=int(temporal_outliers),
            outlier_magnitudes=outlier_magnitudes,
            outlier_timestamps=outlier_timestamps,
            outlier_sources=outlier_sources
        )

    def _calculate_temporal_metrics(
        self,
        raw_df: pd.DataFrame,
        filtered_df: pd.DataFrame
    ) -> TemporalMetrics:
        """Calculate temporal consistency metrics."""
        # Initialize with defaults
        metrics = TemporalMetrics(
            max_daily_change=0,
            impossible_changes=0,
            daily_volatility=0,
            trend_correlation=0,
            smoothness_score=0,
            inflection_points=0,
            gap_count=0,
            avg_gap_duration=0,
            max_gap_duration=0
        )

        # Calculate daily changes for raw data
        if len(raw_df) > 1:
            raw_sorted = raw_df.sort_values('timestamp')
            time_diffs = np.diff(raw_sorted['timestamp'].values) / np.timedelta64(1, 'D')
            weight_diffs = np.abs(np.diff(raw_sorted['weight'].values))

            # Daily change metrics
            if len(time_diffs) > 0:
                daily_changes = weight_diffs / np.maximum(time_diffs, 1)
                metrics.max_daily_change = np.max(daily_changes)
                metrics.impossible_changes = np.sum(daily_changes > PHYSIOLOGICAL_LIMITS['MAX_DAILY_CHANGE_KG'])

                # Rolling volatility (7-day window)
                if len(daily_changes) >= 7:
                    volatilities = [np.std(daily_changes[i:i+7]) for i in range(len(daily_changes)-6)]
                    metrics.daily_volatility = np.mean(volatilities)
                else:
                    metrics.daily_volatility = np.std(daily_changes)

        # Trend correlation between raw and filtered
        if len(raw_df) > 1 and len(filtered_df) > 1:
            # Align timestamps for correlation
            common_timestamps = sorted(set(raw_df['timestamp']) & set(filtered_df['timestamp']))
            if len(common_timestamps) > 1:
                # Get unique values for each timestamp (handle duplicates)
                raw_weights = []
                filtered_weights = []

                for ts in common_timestamps:
                    raw_at_ts = raw_df[raw_df['timestamp'] == ts]['weight'].values
                    filtered_at_ts = filtered_df[filtered_df['timestamp'] == ts]['weight'].values

                    # Take mean if multiple values at same timestamp
                    if len(raw_at_ts) > 0 and len(filtered_at_ts) > 0:
                        raw_weights.append(np.mean(raw_at_ts))
                        filtered_weights.append(np.mean(filtered_at_ts))

                if len(raw_weights) > 1 and len(filtered_weights) > 1:
                    try:
                        corr_matrix = np.corrcoef(raw_weights, filtered_weights)
                        metrics.trend_correlation = corr_matrix[0, 1]
                    except Exception as e:
                        logger.warning(f"Could not calculate correlation: {e}")
                        metrics.trend_correlation = None

        # Smoothness score (second derivative)
        if len(filtered_df) > 2:
            filtered_sorted = filtered_df.sort_values('timestamp')
            weights = filtered_sorted['weight'].values
            second_derivative = np.diff(weights, n=2)
            metrics.smoothness_score = 1 / (1 + np.std(second_derivative))

            # Count inflection points
            first_derivative = np.diff(weights)
            sign_changes = np.diff(np.sign(first_derivative))
            metrics.inflection_points = np.sum(sign_changes != 0)

        # Gap analysis
        if len(raw_df) > 1:
            raw_sorted = raw_df.sort_values('timestamp')
            time_gaps = np.diff(raw_sorted['timestamp'].values) / np.timedelta64(1, 'D')

            # Consider gaps > 3 days as significant
            significant_gaps = time_gaps[time_gaps > 3]
            metrics.gap_count = len(significant_gaps)

            if len(significant_gaps) > 0:
                metrics.avg_gap_duration = np.mean(significant_gaps)
                metrics.max_gap_duration = np.max(significant_gaps)

        return metrics

    def _calculate_medical_impact(
        self,
        raw_df: pd.DataFrame,
        filtered_df: pd.DataFrame
    ) -> MedicalImpactMetrics:
        """Calculate medical decision impact metrics."""
        metrics = MedicalImpactMetrics(
            start_point_variance=0,
            end_point_variance=0,
            total_change_delta=0,
            misclassification_rate=0,
            direction_errors=0,
            minor_errors=0,
            moderate_errors=0,
            severe_errors=0,
            confidence_interval_reduction=0
        )

        if raw_df.empty or filtered_df.empty:
            return metrics

        # Calculate start/end point selection variance
        # Use first 14 days for start, last 14 days for end
        raw_sorted = raw_df.sort_values('timestamp')
        filtered_sorted = filtered_df.sort_values('timestamp')

        # Start point analysis (first 14 days)
        start_date = raw_sorted['timestamp'].min()
        start_window = start_date + timedelta(days=14)

        raw_start = raw_sorted[raw_sorted['timestamp'] <= start_window]['weight']
        filtered_start = filtered_sorted[filtered_sorted['timestamp'] <= start_window]['weight']

        if len(raw_start) > 0 and len(filtered_start) > 0:
            raw_start_weight = np.mean(raw_start)
            filtered_start_weight = np.mean(filtered_start)
            metrics.start_point_variance = abs(raw_start_weight - filtered_start_weight)

        # End point analysis (last 14 days)
        end_date = raw_sorted['timestamp'].max()
        end_window = end_date - timedelta(days=14)

        raw_end = raw_sorted[raw_sorted['timestamp'] >= end_window]['weight']
        filtered_end = filtered_sorted[filtered_sorted['timestamp'] >= end_window]['weight']

        if len(raw_end) > 0 and len(filtered_end) > 0:
            raw_end_weight = np.mean(raw_end)
            filtered_end_weight = np.mean(filtered_end)
            metrics.end_point_variance = abs(raw_end_weight - filtered_end_weight)

        # Total change calculation
        if len(raw_start) > 0 and len(raw_end) > 0 and len(filtered_start) > 0 and len(filtered_end) > 0:
            raw_change = raw_end_weight - raw_start_weight
            filtered_change = filtered_end_weight - filtered_start_weight
            metrics.total_change_delta = abs(raw_change - filtered_change)

            # Direction error
            if np.sign(raw_change) != np.sign(filtered_change):
                metrics.direction_errors = 1

            # Magnitude errors
            diff = abs(raw_change - filtered_change)
            if diff < 1:
                metrics.minor_errors = 1
            elif diff < 3:
                metrics.moderate_errors = 1
            else:
                metrics.severe_errors = 1

            # Clinical threshold analysis (5% weight loss)
            if raw_start_weight > 0 and filtered_start_weight > 0:
                raw_pct_change = (raw_change / raw_start_weight) * 100
                filtered_pct_change = (filtered_change / filtered_start_weight) * 100

                # Check if classification differs for 5% threshold
                raw_meets_5pct = raw_pct_change <= -5
                filtered_meets_5pct = filtered_pct_change <= -5

                if raw_meets_5pct != filtered_meets_5pct:
                    metrics.misclassification_rate = 1

        # Confidence interval reduction
        if len(raw_df) > 1 and len(filtered_df) > 1:
            raw_std = np.std(raw_df['weight'])
            filtered_std = np.std(filtered_df['weight'])

            # CI width is proportional to standard error
            raw_ci_width = 1.96 * raw_std / np.sqrt(len(raw_df))
            filtered_ci_width = 1.96 * filtered_std / np.sqrt(len(filtered_df))

            if raw_ci_width > 0:
                metrics.confidence_interval_reduction = (raw_ci_width - filtered_ci_width) / raw_ci_width

        return metrics

    def _analyze_by_source(
        self,
        raw_df: pd.DataFrame,
        filtered_df: pd.DataFrame
    ) -> Dict[str, Dict]:
        """Analyze filtering effectiveness by data source."""
        results = {}

        if 'source' not in raw_df.columns:
            return results

        sources = raw_df['source'].unique()

        for source in sources:
            raw_source = raw_df[raw_df['source'] == source]
            filtered_source = filtered_df[filtered_df['source'] == source] if not filtered_df.empty else pd.DataFrame()

            source_metrics = {
                "total_measurements": len(raw_source),
                "filtered_measurements": len(filtered_source),
                "removal_rate": 1 - (len(filtered_source) / len(raw_source)) if len(raw_source) > 0 else 0,
                "expected_reliability": SOURCE_PROFILES.get(source, {}).get("reliability", "unknown"),
                "expected_outlier_rate": SOURCE_PROFILES.get(source, {}).get("outlier_rate", 0)
            }

            # Calculate actual outlier rate
            if len(raw_source) > 1:
                weights = raw_source['weight'].values
                q1, q3 = np.percentile(weights, [25, 75])
                iqr = q3 - q1
                outliers = np.sum((weights < q1 - 1.5 * iqr) | (weights > q3 + 1.5 * iqr))
                source_metrics["actual_outlier_rate"] = outliers / len(raw_source)

            results[source] = source_metrics

        return results

    def _calculate_reporting_metrics(
        self,
        cohort_raw: Dict[str, pd.DataFrame],
        cohort_filtered: Dict[str, pd.DataFrame]
    ) -> ReportingMetrics:
        """Calculate quarterly reporting impact metrics."""
        # Initialize tracking variables
        raw_changes = []
        filtered_changes = []

        valid_baseline_raw = 0
        valid_baseline_filtered = 0
        valid_endpoint_raw = 0
        valid_endpoint_filtered = 0

        for user_id in cohort_raw:
            raw_df = cohort_raw[user_id]
            filtered_df = cohort_filtered.get(user_id, pd.DataFrame())

            if raw_df.empty:
                continue

            # Calculate weight changes for raw data
            raw_sorted = raw_df.sort_values('timestamp')
            start_date = raw_sorted['timestamp'].min()
            end_date = raw_sorted['timestamp'].max()

            # Check baseline validity (measurements in first 14 days)
            baseline_window = start_date + timedelta(days=14)
            raw_baseline = raw_sorted[raw_sorted['timestamp'] <= baseline_window]

            if len(raw_baseline) >= 2:  # At least 2 measurements in baseline
                valid_baseline_raw += 1

                # Check endpoint validity (measurements in last 14 days)
                endpoint_window = end_date - timedelta(days=14)
                raw_endpoint = raw_sorted[raw_sorted['timestamp'] >= endpoint_window]

                if len(raw_endpoint) >= 2:
                    valid_endpoint_raw += 1

                    # Calculate weight change
                    start_weight = np.mean(raw_baseline['weight'])
                    end_weight = np.mean(raw_endpoint['weight'])
                    pct_change = ((end_weight - start_weight) / start_weight) * 100
                    raw_changes.append(pct_change)

            # Same for filtered data
            if not filtered_df.empty:
                filtered_sorted = filtered_df.sort_values('timestamp')

                filtered_baseline = filtered_sorted[filtered_sorted['timestamp'] <= baseline_window]
                if len(filtered_baseline) >= 2:
                    valid_baseline_filtered += 1

                    filtered_endpoint = filtered_sorted[filtered_sorted['timestamp'] >= endpoint_window]
                    if len(filtered_endpoint) >= 2:
                        valid_endpoint_filtered += 1

                        start_weight = np.mean(filtered_baseline['weight'])
                        end_weight = np.mean(filtered_endpoint['weight'])
                        pct_change = ((end_weight - start_weight) / start_weight) * 100
                        filtered_changes.append(pct_change)

        # Calculate cohort statistics
        raw_mean_change = np.mean(raw_changes) if raw_changes else 0
        filtered_mean_change = np.mean(filtered_changes) if filtered_changes else 0

        # Success rates
        pct_5_raw = np.sum(np.array(raw_changes) <= -5) / len(raw_changes) * 100 if raw_changes else 0
        pct_5_filtered = np.sum(np.array(filtered_changes) <= -5) / len(filtered_changes) * 100 if filtered_changes else 0
        pct_10_raw = np.sum(np.array(raw_changes) <= -10) / len(raw_changes) * 100 if raw_changes else 0
        pct_10_filtered = np.sum(np.array(filtered_changes) <= -10) / len(filtered_changes) * 100 if filtered_changes else 0

        # Statistical power metrics
        raw_var = np.var(raw_changes) if raw_changes else 0
        filtered_var = np.var(filtered_changes) if filtered_changes else 0
        variance_reduction = (raw_var - filtered_var) / raw_var if raw_var > 0 else 0

        # Cohen's d effect size
        pooled_std = np.sqrt((raw_var + filtered_var) / 2) if raw_var > 0 or filtered_var > 0 else 1
        effect_size_improvement = abs(filtered_mean_change) / pooled_std if pooled_std > 0 else 0

        return ReportingMetrics(
            raw_cohort_mean=raw_mean_change,
            filtered_cohort_mean=filtered_mean_change,
            mean_difference=filtered_mean_change - raw_mean_change,
            mean_percent_change=(filtered_mean_change - raw_mean_change) / abs(raw_mean_change) * 100 if raw_mean_change != 0 else 0,
            pct_losing_5pct_raw=pct_5_raw,
            pct_losing_5pct_filtered=pct_5_filtered,
            pct_losing_10pct_raw=pct_10_raw,
            pct_losing_10pct_filtered=pct_10_filtered,
            valid_baseline_raw=valid_baseline_raw,
            valid_baseline_filtered=valid_baseline_filtered,
            valid_endpoint_raw=valid_endpoint_raw,
            valid_endpoint_filtered=valid_endpoint_filtered,
            variance_reduction=variance_reduction,
            effect_size_improvement=effect_size_improvement
        )

    def _aggregate_user_metrics(self, user_metrics: List[Dict]) -> Dict:
        """Aggregate individual user metrics to cohort level."""
        if not user_metrics:
            return {}

        aggregate = {
            "total_users": len(user_metrics),
            "avg_removal_rate": np.mean([m["data_summary"]["removal_rate"] for m in user_metrics]),
            "outlier_summary": {
                "avg_outlier_rate": np.mean([m["outliers"]["outlier_rate"] for m in user_metrics if "outliers" in m]),
                "total_outliers": sum([m["outliers"]["outliers_removed"] for m in user_metrics if "outliers" in m])
            },
            "temporal_summary": {
                "avg_daily_volatility": np.mean([m["temporal"]["daily_change"]["volatility"] for m in user_metrics if "temporal" in m and m["temporal"]["daily_change"]["volatility"] > 0]),
                "total_impossible_changes": sum([m["temporal"]["daily_change"]["impossible_count"] for m in user_metrics if "temporal" in m])
            },
            "medical_summary": {
                "total_direction_errors": sum([m["medical_impact"]["clinical"]["direction_errors"] for m in user_metrics if "medical_impact" in m]),
                "avg_confidence_improvement": np.mean([m["medical_impact"]["confidence"]["ci_reduction"] for m in user_metrics if "medical_impact" in m])
            }
        }

        return aggregate
#!/usr/bin/env python3
"""
Comprehensive Filtering Analysis Script
Analyzes the effectiveness of weight data filtering by comparing raw vs filtered data.
Focuses on medical decision safety and quarterly reporting accuracy.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import shapiro, levene, ks_2samp, wilcoxon
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Visualization imports
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Some visualizations will be skipped.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Some visualizations will be skipped.")


class ComprehensiveFilteringAnalyzer:
    """
    Comprehensive analysis of filtering effectiveness for weight data.
    """

    def __init__(self, output_dir: str = "reports"):
        """Initialize the analyzer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = {}
        self.visualizations = []

    def analyze_user_data(
        self, raw_df: pd.DataFrame, filtered_df: pd.DataFrame, user_id: str
    ) -> Dict[str, Any]:
        """
        Analyze filtering effectiveness for a single user.

        Args:
            raw_df: Raw measurement data
            filtered_df: Filtered measurement data
            user_id: User identifier

        Returns:
            Dictionary of user-specific metrics
        """
        if raw_df.empty or filtered_df.empty:
            return {}

        # Ensure datetime sorting
        raw_df = raw_df.sort_values("effectiveDateTime")
        filtered_df = filtered_df.sort_values("effectiveDateTime")

        metrics = {
            "user_id": user_id,
            "raw_count": len(raw_df),
            "filtered_count": len(filtered_df),
            "removal_rate": 1 - (len(filtered_df) / len(raw_df))
            if len(raw_df) > 0
            else 0,
        }

        # Statistical metrics
        metrics.update(self._calculate_statistical_metrics(raw_df, filtered_df))

        # Temporal consistency
        metrics.update(self._calculate_temporal_consistency(raw_df, filtered_df))

        # Medical decision impact
        metrics.update(self._calculate_medical_impact(raw_df, filtered_df))

        # Source reliability
        if "source" in raw_df.columns:
            metrics.update(self._analyze_source_reliability(raw_df, filtered_df))

        return metrics

    def _calculate_statistical_metrics(
        self, raw_df: pd.DataFrame, filtered_df: pd.DataFrame
    ) -> Dict:
        """Calculate statistical distribution metrics."""
        metrics = {}

        raw_weights = raw_df["weight"].values
        filtered_weights = filtered_df["weight"].values

        # Central tendency
        metrics["mean_raw"] = np.mean(raw_weights)
        metrics["mean_filtered"] = np.mean(filtered_weights)
        metrics["mean_shift"] = metrics["mean_filtered"] - metrics["mean_raw"]

        metrics["median_raw"] = np.median(raw_weights)
        metrics["median_filtered"] = np.median(filtered_weights)
        metrics["median_shift"] = metrics["median_filtered"] - metrics["median_raw"]

        # Dispersion
        metrics["std_raw"] = np.std(raw_weights)
        metrics["std_filtered"] = np.std(filtered_weights)
        metrics["std_reduction"] = (
            (metrics["std_raw"] - metrics["std_filtered"]) / metrics["std_raw"]
            if metrics["std_raw"] > 0
            else 0
        )

        # IQR analysis
        q1_raw, q3_raw = np.percentile(raw_weights, [25, 75])
        q1_filt, q3_filt = np.percentile(filtered_weights, [25, 75])
        metrics["iqr_raw"] = q3_raw - q1_raw
        metrics["iqr_filtered"] = q3_filt - q1_filt
        metrics["iqr_reduction"] = (
            (metrics["iqr_raw"] - metrics["iqr_filtered"]) / metrics["iqr_raw"]
            if metrics["iqr_raw"] > 0
            else 0
        )

        # Coefficient of variation
        metrics["cv_raw"] = (
            metrics["std_raw"] / metrics["mean_raw"] if metrics["mean_raw"] > 0 else 0
        )
        metrics["cv_filtered"] = (
            metrics["std_filtered"] / metrics["mean_filtered"]
            if metrics["mean_filtered"] > 0
            else 0
        )

        # Distribution shape
        if len(raw_weights) > 3:
            metrics["skewness_raw"] = stats.skew(raw_weights)
            metrics["kurtosis_raw"] = stats.kurtosis(raw_weights)
        if len(filtered_weights) > 3:
            metrics["skewness_filtered"] = stats.skew(filtered_weights)
            metrics["kurtosis_filtered"] = stats.kurtosis(filtered_weights)

        # Outlier detection
        metrics.update(self._detect_outliers(raw_weights, filtered_weights))

        return metrics

    def _detect_outliers(
        self, raw_weights: np.ndarray, filtered_weights: np.ndarray
    ) -> Dict:
        """Detect outliers using multiple methods."""
        metrics = {}

        # IQR method
        q1, q3 = np.percentile(raw_weights, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        iqr_outliers = (raw_weights < lower_bound) | (raw_weights > upper_bound)
        metrics["outliers_iqr"] = np.sum(iqr_outliers)
        metrics["outlier_rate_iqr"] = np.mean(iqr_outliers)

        # Z-score method (for raw data)
        if len(raw_weights) > 2:
            z_scores = np.abs(stats.zscore(raw_weights))
            z_outliers = z_scores > 3
            metrics["outliers_zscore"] = np.sum(z_outliers)
            metrics["outlier_rate_zscore"] = np.mean(z_outliers)

        # MAD method
        median = np.median(raw_weights)
        mad = np.median(np.abs(raw_weights - median))
        if mad > 0:
            modified_z_scores = 0.6745 * (raw_weights - median) / mad
            mad_outliers = np.abs(modified_z_scores) > 3.5
            metrics["outliers_mad"] = np.sum(mad_outliers)
            metrics["outlier_rate_mad"] = np.mean(mad_outliers)

        # Isolation Forest (if enough data)
        if len(raw_weights) > 10:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(raw_weights.reshape(-1, 1))
            iso_outliers = outlier_labels == -1
            metrics["outliers_isolation"] = np.sum(iso_outliers)
            metrics["outlier_rate_isolation"] = np.mean(iso_outliers)

        # Points removed by filtering
        metrics["points_removed"] = len(raw_weights) - len(filtered_weights)
        metrics["removal_rate"] = (
            metrics["points_removed"] / len(raw_weights) if len(raw_weights) > 0 else 0
        )

        return metrics

    def _calculate_temporal_consistency(
        self, raw_df: pd.DataFrame, filtered_df: pd.DataFrame
    ) -> Dict:
        """Calculate temporal consistency metrics."""
        metrics = {}

        # Daily changes analysis
        if len(raw_df) > 1:
            raw_df = raw_df.sort_values("effectiveDateTime")
            raw_changes = []
            for i in range(1, len(raw_df)):
                days_diff = (
                    raw_df.iloc[i]["effectiveDateTime"]
                    - raw_df.iloc[i - 1]["effectiveDateTime"]
                ).days
                if days_diff > 0:
                    weight_change = (
                        abs(raw_df.iloc[i]["weight"] - raw_df.iloc[i - 1]["weight"])
                        / days_diff
                    )
                    raw_changes.append(weight_change)

            if raw_changes:
                metrics["max_daily_change_raw"] = np.max(raw_changes)
                metrics["mean_daily_change_raw"] = np.mean(raw_changes)
                metrics["impossible_changes_raw"] = sum(
                    c > 2.0 for c in raw_changes
                )  # >2kg/day

        if len(filtered_df) > 1:
            filtered_df = filtered_df.sort_values("effectiveDateTime")
            filtered_changes = []
            for i in range(1, len(filtered_df)):
                days_diff = (
                    filtered_df.iloc[i]["effectiveDateTime"]
                    - filtered_df.iloc[i - 1]["effectiveDateTime"]
                ).days
                if days_diff > 0:
                    weight_change = (
                        abs(
                            filtered_df.iloc[i]["weight"]
                            - filtered_df.iloc[i - 1]["weight"]
                        )
                        / days_diff
                    )
                    filtered_changes.append(weight_change)

            if filtered_changes:
                metrics["max_daily_change_filtered"] = np.max(filtered_changes)
                metrics["mean_daily_change_filtered"] = np.mean(filtered_changes)
                metrics["impossible_changes_filtered"] = sum(
                    c > 2.0 for c in filtered_changes
                )

        # Volatility index (rolling standard deviation)
        if len(raw_df) > 7:
            raw_df["weight_rolling_std"] = (
                raw_df["weight"].rolling(window=7, min_periods=3).std()
            )
            metrics["volatility_raw"] = raw_df["weight_rolling_std"].mean()

        if len(filtered_df) > 7:
            filtered_df["weight_rolling_std"] = (
                filtered_df["weight"].rolling(window=7, min_periods=3).std()
            )
            metrics["volatility_filtered"] = filtered_df["weight_rolling_std"].mean()

        # Trend preservation (if enough data)
        if len(raw_df) > 10 and len(filtered_df) > 10:
            # Create daily averages for comparison
            raw_daily = (
                raw_df.set_index("effectiveDateTime").resample("D")["weight"].mean()
            )
            filtered_daily = (
                filtered_df.set_index("effectiveDateTime")
                .resample("D")["weight"]
                .mean()
            )

            # Find overlapping dates
            common_dates = raw_daily.index.intersection(filtered_daily.index)
            if len(common_dates) > 5:
                correlation = np.corrcoef(
                    raw_daily[common_dates].values, filtered_daily[common_dates].values
                )[0, 1]
                metrics["trend_correlation"] = correlation

        return metrics

    def _calculate_medical_impact(
        self, raw_df: pd.DataFrame, filtered_df: pd.DataFrame
    ) -> Dict:
        """Calculate medical decision impact metrics."""
        metrics = {}

        if raw_df.empty or filtered_df.empty:
            return metrics

        # Weight change calculation accuracy (30-day, 60-day, 90-day)
        for days in [30, 60, 90]:
            metrics.update(
                self._calculate_weight_change_accuracy(
                    raw_df, filtered_df, days, f"{days}d"
                )
            )

        # Clinical threshold analysis (5% and 10% weight loss)
        initial_raw = raw_df.iloc[0]["weight"]
        final_raw = raw_df.iloc[-1]["weight"]
        raw_change_pct = 100 * (initial_raw - final_raw) / initial_raw

        initial_filtered = filtered_df.iloc[0]["weight"]
        final_filtered = filtered_df.iloc[-1]["weight"]
        filtered_change_pct = (
            100 * (initial_filtered - final_filtered) / initial_filtered
        )

        # Check if classification changes at key thresholds
        metrics["crosses_5pct_threshold"] = (
            raw_change_pct < 5 and filtered_change_pct >= 5
        ) or (raw_change_pct >= 5 and filtered_change_pct < 5)
        metrics["crosses_10pct_threshold"] = (
            raw_change_pct < 10 and filtered_change_pct >= 10
        ) or (raw_change_pct >= 10 and filtered_change_pct < 10)

        # Direction agreement
        metrics["direction_agreement"] = np.sign(raw_change_pct) == np.sign(
            filtered_change_pct
        )

        # Magnitude difference
        metrics["change_magnitude_diff"] = abs(raw_change_pct - filtered_change_pct)

        return metrics

    def _calculate_weight_change_accuracy(
        self, raw_df: pd.DataFrame, filtered_df: pd.DataFrame, days: int, label: str
    ) -> Dict:
        """Calculate weight change accuracy for a specific time period."""
        metrics = {}

        # Find measurements close to start and end of period
        if len(raw_df) < 2 or len(filtered_df) < 2:
            return metrics

        start_date = raw_df.iloc[0]["effectiveDateTime"]
        target_date = start_date + timedelta(days=days)

        # Find closest measurements to target date
        raw_end = raw_df.iloc[
            (raw_df["effectiveDateTime"] - target_date).abs().argsort()[:1]
        ]
        filtered_end = filtered_df.iloc[
            (filtered_df["effectiveDateTime"] - target_date).abs().argsort()[:1]
        ]

        if not raw_end.empty and not filtered_end.empty:
            raw_change = raw_end.iloc[0]["weight"] - raw_df.iloc[0]["weight"]
            filtered_change = (
                filtered_end.iloc[0]["weight"] - filtered_df.iloc[0]["weight"]
            )

            metrics[f"weight_change_raw_{label}"] = raw_change
            metrics[f"weight_change_filtered_{label}"] = filtered_change
            metrics[f"weight_change_diff_{label}"] = abs(raw_change - filtered_change)

        return metrics

    def _analyze_source_reliability(
        self, raw_df: pd.DataFrame, filtered_df: pd.DataFrame
    ) -> Dict:
        """Analyze reliability by data source."""
        metrics = {}

        if "source" not in raw_df.columns:
            return metrics

        # Calculate removal rate by source
        source_counts = raw_df["source"].value_counts()
        for source in source_counts.index:
            raw_source = raw_df[raw_df["source"] == source]
            filtered_source = (
                filtered_df[filtered_df["source"] == source]
                if "source" in filtered_df.columns
                else pd.DataFrame()
            )

            removal_rate = (
                1 - (len(filtered_source) / len(raw_source))
                if len(raw_source) > 0
                else 0
            )
            metrics[f"removal_rate_{source}"] = removal_rate
            metrics[f"count_{source}"] = len(raw_source)

        return metrics

    def generate_population_metrics(self, user_metrics: List[Dict]) -> Dict:
        """
        Generate population-level metrics from individual user analyses.

        Args:
            user_metrics: List of individual user metric dictionaries

        Returns:
            Population-level metrics dictionary
        """
        if not user_metrics:
            return {}

        df = pd.DataFrame(user_metrics)
        population_metrics = {}

        # Aggregate statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col != "user_id" and not df[col].isna().all():
                population_metrics[f"{col}_mean"] = df[col].mean()
                population_metrics[f"{col}_median"] = df[col].median()
                population_metrics[f"{col}_std"] = df[col].std()
                population_metrics[f"{col}_min"] = df[col].min()
                population_metrics[f"{col}_max"] = df[col].max()

        # Overall effectiveness metrics
        population_metrics["total_users"] = len(user_metrics)
        population_metrics["avg_removal_rate"] = (
            df["removal_rate"].mean() if "removal_rate" in df else 0
        )
        population_metrics["avg_std_reduction"] = (
            df["std_reduction"].mean() if "std_reduction" in df else 0
        )

        # Medical impact summary
        if "crosses_5pct_threshold" in df:
            population_metrics["pct_crossing_5pct"] = (
                df["crosses_5pct_threshold"].mean() * 100
            )
        if "crosses_10pct_threshold" in df:
            population_metrics["pct_crossing_10pct"] = (
                df["crosses_10pct_threshold"].mean() * 100
            )
        if "direction_agreement" in df:
            population_metrics["pct_direction_agreement"] = (
                df["direction_agreement"].mean() * 100
            )

        return population_metrics

    def analyze_quarterly_reporting_impact(
        self,
        raw_cohort_df: pd.DataFrame,
        filtered_cohort_df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict:
        """
        Analyze impact on quarterly reporting metrics.

        Args:
            raw_cohort_df: Raw data for entire cohort
            filtered_cohort_df: Filtered data for entire cohort
            start_date: Report start date
            end_date: Report end date

        Returns:
            Dictionary of reporting impact metrics
        """
        metrics = {}

        # User inclusion analysis
        raw_users = raw_cohort_df["user_id"].unique()
        filtered_users = filtered_cohort_df["user_id"].unique()

        metrics["total_raw_users"] = len(raw_users)
        metrics["total_filtered_users"] = len(filtered_users)

        # Analyze users with valid baselines and endpoints
        valid_baseline_raw = 0
        valid_baseline_filtered = 0
        valid_endpoint_raw = 0
        valid_endpoint_filtered = 0

        for user_id in raw_users:
            user_raw = raw_cohort_df[raw_cohort_df["user_id"] == user_id]

            # Check baseline (first 7 days)
            baseline_window = user_raw[
                (user_raw["effectiveDateTime"] >= start_date)
                & (user_raw["effectiveDateTime"] <= start_date + timedelta(days=7))
            ]
            if not baseline_window.empty:
                valid_baseline_raw += 1

            # Check endpoint (last 7 days)
            endpoint_window = user_raw[
                (user_raw["effectiveDateTime"] >= end_date - timedelta(days=7))
                & (user_raw["effectiveDateTime"] <= end_date)
            ]
            if not endpoint_window.empty:
                valid_endpoint_raw += 1

        for user_id in filtered_users:
            user_filtered = filtered_cohort_df[filtered_cohort_df["user_id"] == user_id]

            # Check baseline
            baseline_window = user_filtered[
                (user_filtered["effectiveDateTime"] >= start_date)
                & (user_filtered["effectiveDateTime"] <= start_date + timedelta(days=7))
            ]
            if not baseline_window.empty:
                valid_baseline_filtered += 1

            # Check endpoint
            endpoint_window = user_filtered[
                (user_filtered["effectiveDateTime"] >= end_date - timedelta(days=7))
                & (user_filtered["effectiveDateTime"] <= end_date)
            ]
            if not endpoint_window.empty:
                valid_endpoint_filtered += 1

        metrics["valid_baseline_raw"] = valid_baseline_raw
        metrics["valid_baseline_filtered"] = valid_baseline_filtered
        metrics["valid_endpoint_raw"] = valid_endpoint_raw
        metrics["valid_endpoint_filtered"] = valid_endpoint_filtered

        # Calculate cohort-level statistics
        cohort_stats_raw = self._calculate_cohort_statistics(
            raw_cohort_df, start_date, end_date
        )
        cohort_stats_filtered = self._calculate_cohort_statistics(
            filtered_cohort_df, start_date, end_date
        )

        metrics["cohort_mean_loss_raw"] = cohort_stats_raw.get("mean_weight_loss", 0)
        metrics["cohort_mean_loss_filtered"] = cohort_stats_filtered.get(
            "mean_weight_loss", 0
        )
        metrics["cohort_std_raw"] = cohort_stats_raw.get("std_weight_loss", 0)
        metrics["cohort_std_filtered"] = cohort_stats_filtered.get("std_weight_loss", 0)

        # Success rates
        metrics["pct_5pct_loss_raw"] = cohort_stats_raw.get("pct_5pct_loss", 0)
        metrics["pct_5pct_loss_filtered"] = cohort_stats_filtered.get(
            "pct_5pct_loss", 0
        )
        metrics["pct_10pct_loss_raw"] = cohort_stats_raw.get("pct_10pct_loss", 0)
        metrics["pct_10pct_loss_filtered"] = cohort_stats_filtered.get(
            "pct_10pct_loss", 0
        )

        # Statistical power improvement
        if metrics["cohort_std_raw"] > 0 and metrics["cohort_std_filtered"] > 0:
            # Calculate effect sizes (Cohen's d)
            effect_size_raw = (
                abs(metrics["cohort_mean_loss_raw"]) / metrics["cohort_std_raw"]
            )
            effect_size_filtered = (
                abs(metrics["cohort_mean_loss_filtered"])
                / metrics["cohort_std_filtered"]
            )
            metrics["effect_size_raw"] = effect_size_raw
            metrics["effect_size_filtered"] = effect_size_filtered
            metrics["effect_size_improvement"] = effect_size_filtered - effect_size_raw

        return metrics

    def _calculate_cohort_statistics(
        self, cohort_df: pd.DataFrame, start_date: datetime, end_date: datetime
    ) -> Dict:
        """Calculate cohort-level weight loss statistics."""
        stats = {}

        if cohort_df.empty:
            return stats

        weight_changes = []
        pct_changes = []

        for user_id in cohort_df["user_id"].unique():
            user_df = cohort_df[cohort_df["user_id"] == user_id]

            # Get baseline weight (first week average)
            baseline = user_df[
                (user_df["effectiveDateTime"] >= start_date)
                & (user_df["effectiveDateTime"] <= start_date + timedelta(days=7))
            ]

            # Get endpoint weight (last week average)
            endpoint = user_df[
                (user_df["effectiveDateTime"] >= end_date - timedelta(days=7))
                & (user_df["effectiveDateTime"] <= end_date)
            ]

            if not baseline.empty and not endpoint.empty:
                baseline_weight = baseline["weight"].mean()
                endpoint_weight = endpoint["weight"].mean()
                weight_change = baseline_weight - endpoint_weight
                pct_change = 100 * weight_change / baseline_weight

                weight_changes.append(weight_change)
                pct_changes.append(pct_change)

        if weight_changes:
            stats["mean_weight_loss"] = np.mean(weight_changes)
            stats["std_weight_loss"] = np.std(weight_changes)
            stats["median_weight_loss"] = np.median(weight_changes)
            stats["pct_5pct_loss"] = (
                sum(p >= 5 for p in pct_changes) / len(pct_changes) * 100
            )
            stats["pct_10pct_loss"] = (
                sum(p >= 10 for p in pct_changes) / len(pct_changes) * 100
            )

        return stats

    def perform_statistical_tests(
        self, raw_data: pd.DataFrame, filtered_data: pd.DataFrame
    ) -> Dict:
        """
        Perform comprehensive statistical tests comparing distributions.

        Args:
            raw_data: Combined raw data from all users
            filtered_data: Combined filtered data from all users

        Returns:
            Dictionary of test results
        """
        tests = {}

        if raw_data.empty or filtered_data.empty:
            return tests

        raw_weights = raw_data["weight"].values
        filtered_weights = filtered_data["weight"].values

        # Normality tests
        if len(raw_weights) > 3:
            stat, p_value = shapiro(
                raw_weights[: min(5000, len(raw_weights))]
            )  # Shapiro has sample size limit
            tests["shapiro_raw"] = {
                "statistic": stat,
                "p_value": p_value,
                "is_normal": p_value > 0.05,
            }

        if len(filtered_weights) > 3:
            stat, p_value = shapiro(
                filtered_weights[: min(5000, len(filtered_weights))]
            )
            tests["shapiro_filtered"] = {
                "statistic": stat,
                "p_value": p_value,
                "is_normal": p_value > 0.05,
            }

        # Variance equality test
        if len(raw_weights) > 1 and len(filtered_weights) > 1:
            stat, p_value = levene(raw_weights, filtered_weights)
            tests["levene"] = {
                "statistic": stat,
                "p_value": p_value,
                "equal_variance": p_value > 0.05,
            }

        # Distribution comparison
        if len(raw_weights) > 1 and len(filtered_weights) > 1:
            # Kolmogorov-Smirnov test
            stat, p_value = ks_2samp(raw_weights, filtered_weights)
            tests["ks_test"] = {
                "statistic": stat,
                "p_value": p_value,
                "same_distribution": p_value > 0.05,
            }

            # Paired test if we can match observations
            if len(raw_weights) == len(filtered_weights):
                stat, p_value = wilcoxon(raw_weights, filtered_weights)
                tests["wilcoxon"] = {
                    "statistic": stat,
                    "p_value": p_value,
                    "significant_difference": p_value < 0.05,
                }

        return tests

    def generate_comprehensive_report(
        self,
        user_metrics: List[Dict],
        population_metrics: Dict,
        reporting_metrics: Dict,
        statistical_tests: Dict,
    ) -> str:
        """
        Generate comprehensive markdown report.

        Args:
            user_metrics: Individual user analysis results
            population_metrics: Population-level metrics
            reporting_metrics: Quarterly reporting impact metrics
            statistical_tests: Statistical test results

        Returns:
            Markdown formatted report
        """
        report = []

        # Header
        report.append("# Comprehensive Filtering Effectiveness Report")
        report.append(
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report.append(
            f"\n**Total Users Analyzed**: {population_metrics.get('total_users', 0)}"
        )

        # Executive Summary
        report.append("\n## Executive Summary")
        report.append("\n### Key Findings")

        avg_removal = population_metrics.get("avg_removal_rate", 0) * 100
        avg_std_reduction = population_metrics.get("avg_std_reduction", 0) * 100
        direction_agreement = population_metrics.get("pct_direction_agreement", 0)

        report.append(
            f"- **Outlier Removal Rate**: {avg_removal:.1f}% of measurements identified as outliers"
        )
        report.append(
            f"- **Data Quality Improvement**: {avg_std_reduction:.1f}% reduction in standard deviation"
        )
        report.append(
            f"- **Medical Decision Consistency**: {direction_agreement:.1f}% agreement in weight change direction"
        )

        # Data Cleanliness Metrics
        report.append("\n## 1. Data Cleanliness Metrics")

        report.append("\n### Statistical Distribution Improvements")
        report.append("\n| Metric | Raw Data | Filtered Data | Improvement |")
        report.append("|--------|----------|---------------|-------------|")

        metrics_to_show = [
            ("Mean Weight", "mean_raw_mean", "mean_filtered_mean"),
            ("Std Deviation", "std_raw_mean", "std_filtered_mean"),
            ("IQR", "iqr_raw_mean", "iqr_filtered_mean"),
            ("CV", "cv_raw_mean", "cv_filtered_mean"),
        ]

        for label, raw_key, filt_key in metrics_to_show:
            raw_val = population_metrics.get(raw_key, 0)
            filt_val = population_metrics.get(filt_key, 0)
            improvement = ((raw_val - filt_val) / raw_val * 100) if raw_val != 0 else 0
            report.append(
                f"| {label} | {raw_val:.2f} | {filt_val:.2f} | {improvement:+.1f}% |"
            )

        # Outlier Detection
        report.append("\n### Outlier Detection Effectiveness")
        report.append(f"- **Average Removal Rate**: {avg_removal:.2f}%")

        outlier_methods = ["iqr", "zscore", "mad", "isolation"]
        for method in outlier_methods:
            key = f"outlier_rate_{method}_mean"
            if key in population_metrics:
                rate = population_metrics[key] * 100
                report.append(
                    f"- **{method.upper()} Method Detection Rate**: {rate:.2f}%"
                )

        # Temporal Consistency
        report.append("\n### Temporal Consistency")

        temporal_metrics = [
            (
                "Max Daily Change",
                "max_daily_change_raw_mean",
                "max_daily_change_filtered_mean",
            ),
            (
                "Mean Daily Change",
                "mean_daily_change_raw_mean",
                "mean_daily_change_filtered_mean",
            ),
            (
                "Impossible Changes (>2kg/day)",
                "impossible_changes_raw_mean",
                "impossible_changes_filtered_mean",
            ),
        ]

        report.append("\n| Metric | Raw Data | Filtered Data | Improvement |")
        report.append("|--------|----------|---------------|-------------|")

        for label, raw_key, filt_key in temporal_metrics:
            raw_val = population_metrics.get(raw_key, 0)
            filt_val = population_metrics.get(filt_key, 0)
            if "Impossible" in label:
                report.append(
                    f"| {label} | {raw_val:.0f} | {filt_val:.0f} | {raw_val - filt_val:.0f} fewer |"
                )
            else:
                improvement = (
                    ((raw_val - filt_val) / raw_val * 100) if raw_val != 0 else 0
                )
                report.append(
                    f"| {label} | {raw_val:.2f} kg | {filt_val:.2f} kg | {improvement:+.1f}% |"
                )

        # Medical Decision Impact
        report.append("\n## 2. Medical Decision Impact")

        report.append("\n### Weight Change Calculation Accuracy")

        for period in ["30d", "60d", "90d"]:
            diff_key = f"weight_change_diff_{period}_mean"
            if diff_key in population_metrics:
                diff = population_metrics[diff_key]
                report.append(
                    f"- **{period} Period**: Average difference of {diff:.2f} kg"
                )

        report.append("\n### Clinical Threshold Analysis")
        pct_5 = population_metrics.get("pct_crossing_5pct", 0)
        pct_10 = population_metrics.get("pct_crossing_10pct", 0)

        report.append(f"- **Users crossing 5% threshold differently**: {pct_5:.1f}%")
        report.append(f"- **Users crossing 10% threshold differently**: {pct_10:.1f}%")
        report.append(f"- **Direction agreement**: {direction_agreement:.1f}%")

        # Quarterly Reporting Impact
        if reporting_metrics:
            report.append("\n## 3. Quarterly Reporting Impact")

            report.append("\n### User Inclusion Analysis")
            report.append(
                f"- **Valid Baseline (Raw)**: {reporting_metrics.get('valid_baseline_raw', 0)} users"
            )
            report.append(
                f"- **Valid Baseline (Filtered)**: {reporting_metrics.get('valid_baseline_filtered', 0)} users"
            )
            report.append(
                f"- **Valid Endpoint (Raw)**: {reporting_metrics.get('valid_endpoint_raw', 0)} users"
            )
            report.append(
                f"- **Valid Endpoint (Filtered)**: {reporting_metrics.get('valid_endpoint_filtered', 0)} users"
            )

            report.append("\n### Cohort Statistics")
            report.append("\n| Metric | Raw Data | Filtered Data | Difference |")
            report.append("|--------|----------|---------------|------------|")

            cohort_metrics = [
                (
                    "Mean Weight Loss",
                    "cohort_mean_loss_raw",
                    "cohort_mean_loss_filtered",
                    "kg",
                ),
                ("Std Deviation", "cohort_std_raw", "cohort_std_filtered", "kg"),
                ("5% Loss Success", "pct_5pct_loss_raw", "pct_5pct_loss_filtered", "%"),
                (
                    "10% Loss Success",
                    "pct_10pct_loss_raw",
                    "pct_10pct_loss_filtered",
                    "%",
                ),
            ]

            for label, raw_key, filt_key, unit in cohort_metrics:
                raw_val = reporting_metrics.get(raw_key, 0)
                filt_val = reporting_metrics.get(filt_key, 0)
                diff = filt_val - raw_val
                report.append(
                    f"| {label} | {raw_val:.2f} {unit} | {filt_val:.2f} {unit} | {diff:+.2f} {unit} |"
                )

            # Statistical power
            if "effect_size_improvement" in reporting_metrics:
                improvement = reporting_metrics["effect_size_improvement"]
                report.append(f"\n### Statistical Power")
                report.append(
                    f"- **Effect Size Improvement**: {improvement:+.3f} (Cohen's d)"
                )

        # Statistical Tests
        if statistical_tests:
            report.append("\n## 4. Statistical Testing Results")

            report.append("\n### Distribution Tests")
            report.append("\n| Test | Result | P-Value | Interpretation |")
            report.append("|------|--------|---------|----------------|")

            if "shapiro_raw" in statistical_tests:
                test = statistical_tests["shapiro_raw"]
                normality = "Normal" if test["is_normal"] else "Not Normal"
                report.append(
                    f"| Shapiro-Wilk (Raw) | {test['statistic']:.4f} | {test['p_value']:.4f} | {normality} |"
                )

            if "shapiro_filtered" in statistical_tests:
                test = statistical_tests["shapiro_filtered"]
                normality = "Normal" if test["is_normal"] else "Not Normal"
                report.append(
                    f"| Shapiro-Wilk (Filtered) | {test['statistic']:.4f} | {test['p_value']:.4f} | {normality} |"
                )

            if "levene" in statistical_tests:
                test = statistical_tests["levene"]
                variance = "Equal" if test["equal_variance"] else "Unequal"
                report.append(
                    f"| Levene's Test | {test['statistic']:.4f} | {test['p_value']:.4f} | {variance} Variance |"
                )

            if "ks_test" in statistical_tests:
                test = statistical_tests["ks_test"]
                dist = "Same" if test["same_distribution"] else "Different"
                report.append(
                    f"| Kolmogorov-Smirnov | {test['statistic']:.4f} | {test['p_value']:.4f} | {dist} Distribution |"
                )

        # Conclusions
        report.append("\n## 5. Conclusions and Recommendations")

        report.append("\n### Key Achievements")
        if avg_std_reduction > 20:
            report.append(
                f"✅ **Excellent variance reduction**: {avg_std_reduction:.1f}% exceeds 20% target"
            )
        elif avg_std_reduction > 10:
            report.append(f"✓ **Good variance reduction**: {avg_std_reduction:.1f}%")
        else:
            report.append(
                f"⚠️ **Limited variance reduction**: {avg_std_reduction:.1f}% below expectations"
            )

        if direction_agreement > 95:
            report.append("✅ **High medical safety**: >95% direction agreement")
        elif direction_agreement > 90:
            report.append("✓ **Good medical safety**: >90% direction agreement")
        else:
            report.append("⚠️ **Medical safety concern**: <90% direction agreement")

        # Success criteria evaluation
        report.append("\n### Success Criteria Evaluation")

        success_criteria = [
            ("Variance Reduction >20%", avg_std_reduction > 20),
            (
                "Outlier Detection >95%",
                avg_removal > 5,
            ),  # Assuming 5-10% outlier rate is good
            ("Medical Safety >99%", direction_agreement > 99),
            ("Direction Agreement >95%", direction_agreement > 95),
        ]

        for criterion, met in success_criteria:
            status = "✅ Met" if met else "❌ Not Met"
            report.append(f"- {criterion}: {status}")

        # Recommendations
        report.append("\n### Recommendations")

        if avg_removal < 5:
            report.append(
                "1. Consider adjusting outlier detection thresholds - removal rate may be too conservative"
            )
        elif avg_removal > 20:
            report.append(
                "1. Review outlier detection thresholds - removal rate may be too aggressive"
            )

        if direction_agreement < 95:
            report.append(
                "2. Investigate cases with direction disagreement for potential algorithm improvements"
            )

        if (
            "effect_size_improvement" in reporting_metrics
            and reporting_metrics["effect_size_improvement"] < 0
        ):
            report.append(
                "3. Filtering may be reducing statistical power - review filtering criteria"
            )

        report.append("\n---")
        report.append(
            f"\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        )

        return "\n".join(report)

    def save_report(self, report: str, filename: str = None):
        """Save report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"filtering_effectiveness_report_{timestamp}.md"

        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            f.write(report)

        print(f"Report saved to: {filepath}")
        return filepath


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive filtering effectiveness analysis"
    )
    parser.add_argument("--raw-data", required=True, help="Path to raw data CSV")
    parser.add_argument(
        "--filtered-data", required=True, help="Path to filtered data CSV"
    )
    parser.add_argument(
        "--output-dir", default="reports", help="Output directory for reports"
    )
    parser.add_argument(
        "--user-sample", type=int, help="Number of users to sample for analysis"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualizations"
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ComprehensiveFilteringAnalyzer(args.output_dir)

    # Load data
    print("Loading data...")
    raw_df = pd.read_csv(args.raw_data)
    filtered_df = pd.read_csv(args.filtered_data)

    # Convert datetime columns
    raw_df["effectiveDateTime"] = pd.to_datetime(raw_df["effectiveDateTime"])
    filtered_df["effectiveDateTime"] = pd.to_datetime(filtered_df["effectiveDateTime"])

    print(f"Loaded {len(raw_df):,} raw measurements")
    print(f"Loaded {len(filtered_df):,} filtered measurements")

    # Get unique users
    all_users = set(raw_df["user_id"].unique()) | set(filtered_df["user_id"].unique())

    # Sample users if requested
    if args.user_sample and args.user_sample < len(all_users):
        import random

        all_users = random.sample(list(all_users), args.user_sample)
        print(f"Sampling {args.user_sample} users for analysis")

    # Analyze individual users
    print("\nAnalyzing individual users...")
    user_metrics = []

    for i, user_id in enumerate(all_users):
        if i % 100 == 0:
            print(f"  Processing user {i + 1}/{len(all_users)}...")

        user_raw = raw_df[raw_df["user_id"] == user_id]
        user_filtered = filtered_df[filtered_df["user_id"] == user_id]

        user_metric = analyzer.analyze_user_data(user_raw, user_filtered, user_id)
        if user_metric:
            user_metrics.append(user_metric)

    # Generate population metrics
    print("\nGenerating population-level metrics...")
    population_metrics = analyzer.generate_population_metrics(user_metrics)

    # Analyze quarterly reporting impact (using full dataset)
    print("\nAnalyzing quarterly reporting impact...")
    # Define reporting period (last 90 days of data)
    end_date = max(
        raw_df["effectiveDateTime"].max(), filtered_df["effectiveDateTime"].max()
    )
    start_date = end_date - timedelta(days=90)

    reporting_metrics = analyzer.analyze_quarterly_reporting_impact(
        raw_df, filtered_df, start_date, end_date
    )

    # Perform statistical tests
    print("\nPerforming statistical tests...")
    statistical_tests = analyzer.perform_statistical_tests(raw_df, filtered_df)

    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    report = analyzer.generate_comprehensive_report(
        user_metrics, population_metrics, reporting_metrics, statistical_tests
    )

    # Save report
    report_path = analyzer.save_report(report)

    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Report saved to: {report_path}")
    print(f"\nKey Findings:")
    print(
        f"  - Outlier Removal Rate: {population_metrics.get('avg_removal_rate', 0) * 100:.1f}%"
    )
    print(
        f"  - Variance Reduction: {population_metrics.get('avg_std_reduction', 0) * 100:.1f}%"
    )
    print(
        f"  - Direction Agreement: {population_metrics.get('pct_direction_agreement', 0):.1f}%"
    )

    # Generate visualizations if requested
    if args.visualize and PLOTLY_AVAILABLE:
        print("\nGenerating visualizations...")
        # TODO: Add visualization generation
        print("Visualization generation not yet implemented")


if __name__ == "__main__":
    main()

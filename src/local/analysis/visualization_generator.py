"""
Visualization generator for filtering effectiveness analysis.
Creates comprehensive visualizations comparing raw and filtered data.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

# Set style for professional plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class FilteringVisualizationGenerator:
    """
    Generates visualizations for filtering effectiveness analysis.
    Creates both individual user and population-level plots.
    """

    def __init__(self, output_dir: str = "reports/visualizations"):
        """
        Initialize visualization generator.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set consistent color scheme
        self.colors = {
            "raw": "#808080",  # Gray for raw data
            "filtered": "#1f77b4",  # Blue for filtered data
            "outlier": "#d62728",  # Red for outliers
            "good": "#2ca02c",  # Green for good quality
            "warning": "#ff7f0e",  # Orange for warning
            "confidence": "#9467bd",  # Purple for confidence bands
        }

    def generate_user_visualization_suite(
        self,
        user_id: str,
        raw_df: pd.DataFrame,
        filtered_df: pd.DataFrame,
        metrics: Dict[str, Any],
    ) -> List[str]:
        """
        Generate complete visualization suite for a single user.

        Args:
            user_id: User identifier
            raw_df: Raw measurement data
            filtered_df: Filtered measurement data
            metrics: Calculated metrics dictionary

        Returns:
            List of paths to generated visualization files
        """
        saved_files = []

        # Create user-specific directory
        user_dir = self.output_dir / f"user_{user_id}"
        user_dir.mkdir(exist_ok=True)

        # 1. Dual-axis time series
        file_path = self._create_dual_time_series(
            user_id, raw_df, filtered_df, user_dir
        )
        if file_path:
            saved_files.append(file_path)

        # 2. Residual plot
        file_path = self._create_residual_plot(user_id, raw_df, filtered_df, user_dir)
        if file_path:
            saved_files.append(file_path)

        # 3. Daily change histogram
        file_path = self._create_daily_change_histogram(
            user_id, raw_df, filtered_df, user_dir
        )
        if file_path:
            saved_files.append(file_path)

        # 4. Quality score heatmap
        file_path = self._create_quality_heatmap(user_id, raw_df, filtered_df, user_dir)
        if file_path:
            saved_files.append(file_path)

        # 5. Comprehensive summary dashboard
        file_path = self._create_user_dashboard(
            user_id, raw_df, filtered_df, metrics, user_dir
        )
        if file_path:
            saved_files.append(file_path)

        return saved_files

    def generate_cohort_visualization_suite(
        self,
        cohort_raw: Dict[str, pd.DataFrame],
        cohort_filtered: Dict[str, pd.DataFrame],
        cohort_metrics: Dict[str, Any],
    ) -> List[str]:
        """
        Generate population-level visualizations.

        Args:
            cohort_raw: Dictionary of user_id -> raw DataFrame
            cohort_filtered: Dictionary of user_id -> filtered DataFrame
            cohort_metrics: Cohort-level metrics

        Returns:
            List of paths to generated visualization files
        """
        saved_files = []

        # 1. Distribution overlay
        file_path = self._create_distribution_overlay(cohort_raw, cohort_filtered)
        if file_path:
            saved_files.append(file_path)

        # 2. Outlier clustering map
        file_path = self._create_outlier_map(cohort_raw, cohort_filtered)
        if file_path:
            saved_files.append(file_path)

        # 3. Source reliability matrix
        file_path = self._create_source_reliability_matrix(cohort_raw, cohort_filtered)
        if file_path:
            saved_files.append(file_path)

        # 4. Cohort trajectory fans
        file_path = self._create_trajectory_fans(cohort_raw, cohort_filtered)
        if file_path:
            saved_files.append(file_path)

        # 5. Impact visualizations
        file_path = self._create_impact_dashboard(cohort_metrics)
        if file_path:
            saved_files.append(file_path)

        return saved_files

    def _create_dual_time_series(
        self,
        user_id: str,
        raw_df: pd.DataFrame,
        filtered_df: pd.DataFrame,
        output_dir: Path,
    ) -> Optional[str]:
        """Create dual-axis time series plot comparing raw and filtered data."""
        try:
            fig, ax = plt.subplots(figsize=(14, 7))

            if not raw_df.empty:
                # Plot raw data as gray background
                ax.plot(
                    raw_df["timestamp"],
                    raw_df["weight"],
                    "o",
                    color=self.colors["raw"],
                    alpha=0.4,
                    markersize=4,
                    label="Raw measurements",
                )

            if not filtered_df.empty:
                # Plot filtered data as blue line
                ax.plot(
                    filtered_df["timestamp"],
                    filtered_df["weight"],
                    "-o",
                    color=self.colors["filtered"],
                    linewidth=2,
                    markersize=6,
                    label="Filtered data",
                )

                # Add confidence band
                weights = filtered_df["weight"].values
                timestamps = filtered_df["timestamp"].values

                # Simple rolling confidence band
                if len(weights) > 3:
                    window = min(7, len(weights))
                    rolling_std = pd.Series(weights).rolling(window, center=True).std()
                    rolling_mean = (
                        pd.Series(weights).rolling(window, center=True).mean()
                    )

                    ax.fill_between(
                        timestamps,
                        rolling_mean - 1.96 * rolling_std,
                        rolling_mean + 1.96 * rolling_std,
                        color=self.colors["confidence"],
                        alpha=0.2,
                        label="95% CI",
                    )

            # Highlight outliers
            if not raw_df.empty and not filtered_df.empty:
                # Find removed points
                raw_set = set(zip(raw_df["timestamp"], raw_df["weight"]))
                filtered_set = set(zip(filtered_df["timestamp"], filtered_df["weight"]))
                outliers = list(raw_set - filtered_set)

                if outliers:
                    outlier_times = [o[0] for o in outliers]
                    outlier_weights = [o[1] for o in outliers]

                    ax.scatter(
                        outlier_times,
                        outlier_weights,
                        color=self.colors["outlier"],
                        s=80,
                        marker="x",
                        linewidth=2,
                        label=f"Outliers ({len(outliers)})",
                        zorder=5,
                    )

            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Weight (kg)", fontsize=12)
            ax.set_title(
                f"Weight Measurements Over Time - User {user_id[:8]}",
                fontsize=14,
                fontweight="bold",
            )
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()

            # Save figure
            file_path = output_dir / f"time_series_{user_id[:8]}.png"
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close()

            return str(file_path)

        except Exception as e:
            logger.error(f"Error creating time series plot for user {user_id}: {e}")
            plt.close()
            return None

    def _create_residual_plot(
        self,
        user_id: str,
        raw_df: pd.DataFrame,
        filtered_df: pd.DataFrame,
        output_dir: Path,
    ) -> Optional[str]:
        """Create residual plot showing removed points relative to filtered trend."""
        try:
            if raw_df.empty or filtered_df.empty or len(filtered_df) < 2:
                return None

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

            # Top plot: Data with fitted trend
            filtered_sorted = filtered_df.sort_values("timestamp")

            # Fit polynomial trend to filtered data
            x_numeric = (
                filtered_sorted["timestamp"] - filtered_sorted["timestamp"].min()
            ).dt.total_seconds() / 86400
            y = filtered_sorted["weight"].values

            # Fit a 3rd degree polynomial
            if len(x_numeric) > 3:
                coeffs = np.polyfit(x_numeric, y, min(3, len(x_numeric) - 1))
                trend_poly = np.poly1d(coeffs)

                # Calculate residuals for all raw data
                raw_sorted = raw_df.sort_values("timestamp")
                raw_x = (
                    raw_sorted["timestamp"] - filtered_sorted["timestamp"].min()
                ).dt.total_seconds() / 86400
                raw_trend = trend_poly(raw_x)
                residuals = raw_sorted["weight"].values - raw_trend

                # Plot original data and trend
                ax1.plot(
                    raw_sorted["timestamp"],
                    raw_sorted["weight"],
                    "o",
                    color=self.colors["raw"],
                    alpha=0.4,
                    markersize=4,
                    label="Raw data",
                )

                trend_times = pd.date_range(
                    start=raw_sorted["timestamp"].min(),
                    end=raw_sorted["timestamp"].max(),
                    periods=100,
                )
                trend_x = (
                    trend_times - filtered_sorted["timestamp"].min()
                ).total_seconds() / 86400
                trend_y = trend_poly(trend_x)

                ax1.plot(
                    trend_times,
                    trend_y,
                    "--",
                    color=self.colors["filtered"],
                    linewidth=2,
                    label="Filtered trend",
                )

                # Bottom plot: Residuals
                ax2.scatter(
                    raw_sorted["timestamp"],
                    residuals,
                    c=abs(residuals),
                    cmap="RdYlGn_r",
                    alpha=0.6,
                    s=30,
                )

                ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
                ax2.axhline(
                    y=2, color="red", linestyle="--", alpha=0.5, label="±2kg threshold"
                )
                ax2.axhline(y=-2, color="red", linestyle="--", alpha=0.5)

                ax1.set_ylabel("Weight (kg)", fontsize=12)
                ax1.set_title(
                    f"Residual Analysis - User {user_id[:8]}",
                    fontsize=14,
                    fontweight="bold",
                )
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                ax2.set_xlabel("Date", fontsize=12)
                ax2.set_ylabel("Residual (kg)", fontsize=12)
                ax2.set_title("Deviations from Filtered Trend", fontsize=12)
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Format x-axis
                for ax in [ax1, ax2]:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

                plt.tight_layout()

                # Save figure
                file_path = output_dir / f"residual_plot_{user_id[:8]}.png"
                plt.savefig(file_path, dpi=150, bbox_inches="tight")
                plt.close()

                return str(file_path)

        except Exception as e:
            logger.error(f"Error creating residual plot for user {user_id}: {e}")
            plt.close()
            return None

    def _create_daily_change_histogram(
        self,
        user_id: str,
        raw_df: pd.DataFrame,
        filtered_df: pd.DataFrame,
        output_dir: Path,
    ) -> Optional[str]:
        """Create side-by-side histograms of daily weight changes."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Calculate daily changes for raw data
            if len(raw_df) > 1:
                raw_sorted = raw_df.sort_values("timestamp")
                raw_changes = np.diff(raw_sorted["weight"].values)

                ax1.hist(
                    raw_changes,
                    bins=30,
                    color=self.colors["raw"],
                    alpha=0.7,
                    edgecolor="black",
                )
                ax1.axvline(x=0, color="black", linestyle="-", linewidth=1)
                ax1.axvline(
                    x=2, color="red", linestyle="--", alpha=0.5, label="±2kg limit"
                )
                ax1.axvline(x=-2, color="red", linestyle="--", alpha=0.5)

                ax1.set_xlabel("Daily Weight Change (kg)", fontsize=12)
                ax1.set_ylabel("Frequency", fontsize=12)
                ax1.set_title(
                    f"Raw Data (std={np.std(raw_changes):.2f}kg)", fontsize=12
                )
                ax1.legend()

            # Calculate daily changes for filtered data
            if len(filtered_df) > 1:
                filtered_sorted = filtered_df.sort_values("timestamp")
                filtered_changes = np.diff(filtered_sorted["weight"].values)

                ax2.hist(
                    filtered_changes,
                    bins=30,
                    color=self.colors["filtered"],
                    alpha=0.7,
                    edgecolor="black",
                )
                ax2.axvline(x=0, color="black", linestyle="-", linewidth=1)
                ax2.axvline(
                    x=2, color="red", linestyle="--", alpha=0.5, label="±2kg limit"
                )
                ax2.axvline(x=-2, color="red", linestyle="--", alpha=0.5)

                ax2.set_xlabel("Daily Weight Change (kg)", fontsize=12)
                ax2.set_ylabel("Frequency", fontsize=12)
                ax2.set_title(
                    f"Filtered Data (std={np.std(filtered_changes):.2f}kg)", fontsize=12
                )
                ax2.legend()

            fig.suptitle(
                f"Daily Weight Change Distribution - User {user_id[:8]}",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()

            # Save figure
            file_path = output_dir / f"daily_changes_{user_id[:8]}.png"
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close()

            return str(file_path)

        except Exception as e:
            logger.error(
                f"Error creating daily change histogram for user {user_id}: {e}"
            )
            plt.close()
            return None

    def _create_quality_heatmap(
        self,
        user_id: str,
        raw_df: pd.DataFrame,
        filtered_df: pd.DataFrame,
        output_dir: Path,
    ) -> Optional[str]:
        """Create quality score heatmap over time."""
        try:
            if "quality_score" not in raw_df.columns:
                return None

            fig, ax = plt.subplots(figsize=(14, 6))

            # Prepare data for heatmap
            raw_sorted = raw_df.sort_values("timestamp")

            # Group by date and average quality scores
            raw_sorted["date"] = raw_sorted["timestamp"].dt.date
            daily_quality = raw_sorted.groupby("date")["quality_score"].mean()

            # Create a matrix for the heatmap (one row, many columns for dates)
            quality_matrix = daily_quality.values.reshape(1, -1)

            # Create heatmap
            im = ax.imshow(quality_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

            # Set labels
            dates = daily_quality.index
            if len(dates) > 30:
                # Show fewer labels if many dates
                step = len(dates) // 30
                ax.set_xticks(np.arange(0, len(dates), step))
                ax.set_xticklabels(
                    [str(dates[i]) for i in range(0, len(dates), step)], rotation=45
                )
            else:
                ax.set_xticks(np.arange(len(dates)))
                ax.set_xticklabels([str(d) for d in dates], rotation=45)

            ax.set_yticks([0])
            ax.set_yticklabels(["Quality Score"])

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Quality Score", rotation=270, labelpad=20)

            # Mark filtered points
            if not filtered_df.empty:
                filtered_dates = set(filtered_df["timestamp"].dt.date)
                for i, date in enumerate(dates):
                    if date not in filtered_dates:
                        ax.scatter(i, 0, marker="x", color="red", s=100)

            ax.set_title(
                f"Daily Quality Scores - User {user_id[:8]}",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel("Date", fontsize=12)

            plt.tight_layout()

            # Save figure
            file_path = output_dir / f"quality_heatmap_{user_id[:8]}.png"
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close()

            return str(file_path)

        except Exception as e:
            logger.error(f"Error creating quality heatmap for user {user_id}: {e}")
            plt.close()
            return None

    def _create_user_dashboard(
        self,
        user_id: str,
        raw_df: pd.DataFrame,
        filtered_df: pd.DataFrame,
        metrics: Dict[str, Any],
        output_dir: Path,
    ) -> Optional[str]:
        """Create comprehensive dashboard for single user."""
        try:
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

            # 1. Time series (top, spanning 2 columns)
            ax1 = fig.add_subplot(gs[0, :2])
            if not raw_df.empty:
                ax1.plot(
                    raw_df["timestamp"],
                    raw_df["weight"],
                    "o",
                    color=self.colors["raw"],
                    alpha=0.4,
                    markersize=3,
                    label="Raw",
                )
            if not filtered_df.empty:
                ax1.plot(
                    filtered_df["timestamp"],
                    filtered_df["weight"],
                    "-o",
                    color=self.colors["filtered"],
                    linewidth=2,
                    markersize=5,
                    label="Filtered",
                )
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Weight (kg)")
            ax1.set_title("Weight Over Time")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Key metrics summary (top right)
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.axis("off")

            if "data_summary" in metrics:
                removal_rate = metrics["data_summary"].get("removal_rate", 0)
                raw_count = metrics["data_summary"]["raw"].get("count", 0)
                filtered_count = metrics["data_summary"]["filtered"].get("count", 0)

                text = (
                    f"Data Summary\n"
                    f"{'=' * 20}\n"
                    f"Raw Points: {raw_count}\n"
                    f"Filtered Points: {filtered_count}\n"
                    f"Removal Rate: {removal_rate:.1%}\n"
                )

                if "outliers" in metrics:
                    outlier_rate = metrics["outliers"].get("outlier_rate", 0)
                    text += f"\nOutlier Rate: {outlier_rate:.1%}"

                if "temporal" in metrics and "daily_change" in metrics["temporal"]:
                    volatility = metrics["temporal"]["daily_change"].get(
                        "volatility", 0
                    )
                    text += f"\nDaily Volatility: {volatility:.2f}kg"

                ax2.text(
                    0.1,
                    0.9,
                    text,
                    transform=ax2.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

            # 3. Distribution comparison (middle left)
            ax3 = fig.add_subplot(gs[1, 0])
            if not raw_df.empty:
                ax3.hist(
                    raw_df["weight"],
                    bins=20,
                    alpha=0.5,
                    color=self.colors["raw"],
                    label="Raw",
                    orientation="horizontal",
                )
            if not filtered_df.empty:
                ax3.hist(
                    filtered_df["weight"],
                    bins=20,
                    alpha=0.5,
                    color=self.colors["filtered"],
                    label="Filtered",
                    orientation="horizontal",
                )
            ax3.set_xlabel("Frequency")
            ax3.set_ylabel("Weight (kg)")
            ax3.set_title("Weight Distribution")
            ax3.legend()

            # 4. Daily changes box plot (middle center)
            ax4 = fig.add_subplot(gs[1, 1])
            changes_data = []
            labels = []

            if len(raw_df) > 1:
                raw_sorted = raw_df.sort_values("timestamp")
                raw_changes = np.diff(raw_sorted["weight"].values)
                changes_data.append(raw_changes)
                labels.append("Raw")

            if len(filtered_df) > 1:
                filtered_sorted = filtered_df.sort_values("timestamp")
                filtered_changes = np.diff(filtered_sorted["weight"].values)
                changes_data.append(filtered_changes)
                labels.append("Filtered")

            if changes_data:
                bp = ax4.boxplot(changes_data, labels=labels, patch_artist=True)
                for patch, color in zip(
                    bp["boxes"],
                    [self.colors["raw"], self.colors["filtered"]][: len(bp["boxes"])],
                ):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

            ax4.set_ylabel("Daily Change (kg)")
            ax4.set_title("Daily Change Distribution")
            ax4.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax4.axhline(y=2, color="red", linestyle="--", alpha=0.3)
            ax4.axhline(y=-2, color="red", linestyle="--", alpha=0.3)

            # 5. Source breakdown (middle right)
            ax5 = fig.add_subplot(gs[1, 2])
            if "source_analysis" in metrics and metrics["source_analysis"]:
                sources = list(metrics["source_analysis"].keys())
                removal_rates = [
                    metrics["source_analysis"][s].get("removal_rate", 0)
                    for s in sources
                ]

                bars = ax5.bar(range(len(sources)), removal_rates)
                ax5.set_xticks(range(len(sources)))
                ax5.set_xticklabels(sources, rotation=45, ha="right")
                ax5.set_ylabel("Removal Rate")
                ax5.set_title("Removal Rate by Source")

                # Color bars by reliability
                for i, bar in enumerate(bars):
                    if removal_rates[i] > 0.2:
                        bar.set_color(self.colors["outlier"])
                    elif removal_rates[i] > 0.1:
                        bar.set_color(self.colors["warning"])
                    else:
                        bar.set_color(self.colors["good"])

            # 6. Medical impact summary (bottom left)
            ax6 = fig.add_subplot(gs[2, 0])
            ax6.axis("off")

            if "medical_impact" in metrics and "accuracy" in metrics["medical_impact"]:
                medical = metrics["medical_impact"]
                text = (
                    f"Medical Impact\n"
                    f"{'=' * 20}\n"
                    f"Start Variance: {medical['accuracy']['start_variance']:.2f}kg\n"
                    f"End Variance: {medical['accuracy']['end_variance']:.2f}kg\n"
                    f"Change Delta: {medical['accuracy']['change_delta']:.2f}kg\n"
                    f"Direction Errors: {medical['clinical']['direction_errors']}\n"
                    f"CI Reduction: {medical['confidence']['ci_reduction']:.1%}"
                )

                ax6.text(
                    0.1,
                    0.9,
                    text,
                    transform=ax6.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
                )

            # 7. Temporal metrics (bottom center)
            ax7 = fig.add_subplot(gs[2, 1])
            ax7.axis("off")

            if "temporal" in metrics:
                temporal = metrics["temporal"]
                text = f"Temporal Analysis\n{'=' * 20}\n"

                if "daily_change" in temporal:
                    text += (
                        f"Max Daily: {temporal['daily_change']['max']:.2f}kg\n"
                        f"Impossible: {temporal['daily_change']['impossible_count']}\n"
                        f"Volatility: {temporal['daily_change']['volatility']:.2f}kg\n"
                    )

                if "trend" in temporal:
                    text += (
                        f"Correlation: {temporal['trend']['correlation']:.3f}\n"
                        f"Smoothness: {temporal['trend']['smoothness']:.3f}"
                    )

                ax7.text(
                    0.1,
                    0.9,
                    text,
                    transform=ax7.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
                )

            # 8. Distribution metrics (bottom right)
            ax8 = fig.add_subplot(gs[2, 2])
            ax8.axis("off")

            if "distribution" in metrics and "improvement" in metrics["distribution"]:
                improvements = metrics["distribution"]["improvement"]
                text = (
                    f"Statistical Improvements\n"
                    f"{'=' * 20}\n"
                    f"Std Reduction: {improvements.get('std_reduction', 0):.1%}\n"
                    f"IQR Compression: {improvements.get('iqr_compression', 0):.1%}\n"
                    f"MAD Improvement: {improvements.get('mad_improvement', 0):.1%}\n"
                    f"CV Reduction: {improvements.get('cv_reduction', 0):.1%}"
                )

                ax8.text(
                    0.1,
                    0.9,
                    text,
                    transform=ax8.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
                )

            fig.suptitle(
                f"Filtering Analysis Dashboard - User {user_id[:8]}",
                fontsize=16,
                fontweight="bold",
            )

            # Save figure
            file_path = output_dir / f"dashboard_{user_id[:8]}.png"
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close()

            return str(file_path)

        except Exception as e:
            logger.error(f"Error creating dashboard for user {user_id}: {e}")
            plt.close()
            return None

    def _create_distribution_overlay(
        self,
        cohort_raw: Dict[str, pd.DataFrame],
        cohort_filtered: Dict[str, pd.DataFrame],
    ) -> Optional[str]:
        """Create kernel density plot overlay for raw vs filtered distributions."""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Collect all weights
            raw_weights = []
            filtered_weights = []

            for user_id in cohort_raw:
                if not cohort_raw[user_id].empty:
                    raw_weights.extend(cohort_raw[user_id]["weight"].values)

                if user_id in cohort_filtered and not cohort_filtered[user_id].empty:
                    filtered_weights.extend(cohort_filtered[user_id]["weight"].values)

            # Create KDE plots
            if raw_weights:
                sns.kdeplot(
                    raw_weights,
                    ax=ax,
                    color=self.colors["raw"],
                    label=f"Raw (n={len(raw_weights)})",
                    linewidth=2,
                    alpha=0.7,
                )

            if filtered_weights:
                sns.kdeplot(
                    filtered_weights,
                    ax=ax,
                    color=self.colors["filtered"],
                    label=f"Filtered (n={len(filtered_weights)})",
                    linewidth=2,
                    alpha=0.7,
                )

            ax.set_xlabel("Weight (kg)", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.set_title(
                "Population Weight Distribution: Raw vs Filtered",
                fontsize=14,
                fontweight="bold",
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add statistics annotations
            if raw_weights and filtered_weights:
                raw_mean = np.mean(raw_weights)
                raw_std = np.std(raw_weights)
                filtered_mean = np.mean(filtered_weights)
                filtered_std = np.std(filtered_weights)

                stats_text = (
                    f"Raw: μ={raw_mean:.1f}kg, σ={raw_std:.1f}kg\n"
                    f"Filtered: μ={filtered_mean:.1f}kg, σ={filtered_std:.1f}kg\n"
                    f"Std Reduction: {(raw_std - filtered_std) / raw_std * 100:.1f}%"
                )

                ax.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            plt.tight_layout()

            # Save figure
            file_path = self.output_dir / "cohort_distribution_overlay.png"
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close()

            return str(file_path)

        except Exception as e:
            logger.error(f"Error creating distribution overlay: {e}")
            plt.close()
            return None

    def _create_outlier_map(
        self,
        cohort_raw: Dict[str, pd.DataFrame],
        cohort_filtered: Dict[str, pd.DataFrame],
    ) -> Optional[str]:
        """Create improved outlier analysis visualization."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

            # Collect outlier data
            outlier_data = []
            source_outlier_rates = {}

            for user_id in cohort_raw:
                raw_df = cohort_raw[user_id]
                filtered_df = cohort_filtered.get(user_id, pd.DataFrame())

                if raw_df.empty:
                    continue

                # Find outliers
                if not filtered_df.empty:
                    raw_set = set(raw_df["timestamp"])
                    filtered_set = set(filtered_df["timestamp"])
                    outlier_timestamps = list(raw_set - filtered_set)

                    for ts in outlier_timestamps:
                        outlier_row = raw_df[raw_df["timestamp"] == ts].iloc[0]

                        # Calculate deviation from rolling median
                        window = raw_df[
                            (raw_df["timestamp"] >= ts - pd.Timedelta(days=7))
                            & (raw_df["timestamp"] <= ts + pd.Timedelta(days=7))
                        ]["weight"]

                        if len(window) > 1:
                            median_weight = window.median()
                            deviation = outlier_row["weight"] - median_weight

                            outlier_data.append(
                                {
                                    "user_id": user_id,
                                    "timestamp": ts,
                                    "weight": outlier_row["weight"],
                                    "deviation": deviation,
                                    "source": outlier_row.get("source", "unknown"),
                                }
                            )

                            # Track source statistics
                            source = outlier_row.get("source", "unknown")
                            if source not in source_outlier_rates:
                                source_outlier_rates[source] = {
                                    "outliers": 0,
                                    "total": 0,
                                }
                            source_outlier_rates[source]["outliers"] += 1

                # Count total by source
                if "source" in raw_df.columns:
                    for source in raw_df["source"].unique():
                        if source not in source_outlier_rates:
                            source_outlier_rates[source] = {"outliers": 0, "total": 0}
                        source_outlier_rates[source]["total"] += len(
                            raw_df[raw_df["source"] == source]
                        )

            if not outlier_data:
                plt.close()
                return None

            outlier_df = pd.DataFrame(outlier_data)

            # 1. Distribution of outlier magnitudes
            ax1 = axes[0, 0]
            ax1.hist(
                np.abs(outlier_df["deviation"]),
                bins=30,
                color=self.colors["outlier"],
                alpha=0.7,
                edgecolor="black",
            )
            ax1.set_xlabel("Absolute Deviation (kg)", fontsize=11)
            ax1.set_ylabel("Count", fontsize=11)
            ax1.set_title("Distribution of Outlier Magnitudes", fontsize=12)
            ax1.axvline(
                x=5, color="orange", linestyle="--", alpha=0.5, label="5kg threshold"
            )
            ax1.axvline(
                x=10, color="red", linestyle="--", alpha=0.5, label="10kg threshold"
            )
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Outlier rate by data source
            ax2 = axes[0, 1]
            if source_outlier_rates:
                sources = list(source_outlier_rates.keys())
                rates = [
                    source_outlier_rates[s]["outliers"]
                    / source_outlier_rates[s]["total"]
                    * 100
                    if source_outlier_rates[s]["total"] > 0
                    else 0
                    for s in sources
                ]

                # Sort by rate
                sorted_pairs = sorted(
                    zip(sources, rates), key=lambda x: x[1], reverse=True
                )
                sources, rates = zip(*sorted_pairs) if sorted_pairs else ([], [])

                bars = ax2.barh(range(len(sources)), rates)
                ax2.set_yticks(range(len(sources)))
                ax2.set_yticklabels(sources, fontsize=9)
                ax2.set_xlabel("Outlier Rate (%)", fontsize=11)
                ax2.set_title("Outlier Rate by Data Source", fontsize=12)

                # Color bars by rate
                for bar, rate in zip(bars, rates):
                    if rate > 20:
                        bar.set_color(self.colors["outlier"])
                    elif rate > 10:
                        bar.set_color(self.colors["warning"])
                    else:
                        bar.set_color(self.colors["good"])

                # Add value labels
                for i, (bar, rate) in enumerate(zip(bars, rates)):
                    ax2.text(
                        rate + 0.5,
                        bar.get_y() + bar.get_height() / 2,
                        f"{rate:.1f}%",
                        va="center",
                        fontsize=9,
                    )

            # 3. Temporal distribution of outliers
            ax3 = axes[1, 0]
            # Group by week
            outlier_df["week"] = pd.to_datetime(outlier_df["timestamp"]).dt.to_period(
                "W"
            )
            weekly_counts = outlier_df.groupby("week").size()

            if len(weekly_counts) > 0:
                ax3.bar(
                    range(len(weekly_counts)),
                    weekly_counts.values,
                    color=self.colors["outlier"],
                    alpha=0.7,
                )
                ax3.set_xlabel("Week Number", fontsize=11)
                ax3.set_ylabel("Outlier Count", fontsize=11)
                ax3.set_title("Temporal Distribution of Outliers", fontsize=12)
                ax3.grid(True, alpha=0.3, axis="y")

            # 4. Summary statistics
            ax4 = axes[1, 1]
            ax4.axis("off")

            total_measurements = sum(len(df) for df in cohort_raw.values())
            total_outliers = len(outlier_df)
            outlier_rate = (
                total_outliers / total_measurements * 100
                if total_measurements > 0
                else 0
            )

            stats_text = f"""Outlier Analysis Summary
{"=" * 30}

Total Measurements: {total_measurements:,}
Total Outliers: {total_outliers:,}
Overall Outlier Rate: {outlier_rate:.2f}%

Deviation Statistics:
Mean Absolute Deviation: {np.abs(outlier_df["deviation"]).mean():.2f} kg
Median Absolute Deviation: {np.abs(outlier_df["deviation"]).median():.2f} kg
Max Deviation: {np.abs(outlier_df["deviation"]).max():.2f} kg

Impact:
Users Affected: {outlier_df["user_id"].nunique()}
Avg Outliers per Affected User: {total_outliers / outlier_df["user_id"].nunique():.1f}"""

            ax4.text(
                0.1,
                0.9,
                stats_text,
                transform=ax4.transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            plt.suptitle(
                "Outlier Analysis: Characteristics and Distribution",
                fontsize=14,
                fontweight="bold",
            )

            # Save figure
            file_path = self.output_dir / "outlier_analysis.png"
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close()

            return str(file_path)

        except Exception as e:
            logger.error(f"Error creating outlier analysis: {e}")
            plt.close()
            return None

    def _create_source_reliability_matrix(
        self,
        cohort_raw: Dict[str, pd.DataFrame],
        cohort_filtered: Dict[str, pd.DataFrame],
    ) -> Optional[str]:
        """Create heatmap of outlier rates by data source."""
        try:
            # Collect outlier rates by source
            source_stats = {}

            for user_id in cohort_raw:
                raw_df = cohort_raw[user_id]
                filtered_df = cohort_filtered.get(user_id, pd.DataFrame())

                if raw_df.empty or "source" not in raw_df.columns:
                    continue

                sources = raw_df["source"].unique()

                for source in sources:
                    if source not in source_stats:
                        source_stats[source] = {
                            "total": 0,
                            "removed": 0,
                            "users": set(),
                        }

                    source_raw = raw_df[raw_df["source"] == source]
                    source_stats[source]["total"] += len(source_raw)
                    source_stats[source]["users"].add(user_id)

                    if not filtered_df.empty:
                        source_filtered = filtered_df[filtered_df["source"] == source]
                        source_stats[source]["removed"] += len(source_raw) - len(
                            source_filtered
                        )

            if not source_stats:
                return None

            # Prepare data for heatmap
            sources = list(source_stats.keys())
            metrics = ["Total Count", "Removal Rate", "User Count", "Avg per User"]

            matrix = np.zeros((len(sources), len(metrics)))

            for i, source in enumerate(sources):
                stats = source_stats[source]
                matrix[i, 0] = stats["total"]
                matrix[i, 1] = (
                    stats["removed"] / stats["total"] if stats["total"] > 0 else 0
                )
                matrix[i, 2] = len(stats["users"])
                matrix[i, 3] = (
                    stats["total"] / len(stats["users"]) if stats["users"] else 0
                )

            # Normalize columns for better visualization
            matrix_norm = np.zeros_like(matrix)
            for j in range(matrix.shape[1]):
                col_max = matrix[:, j].max()
                if col_max > 0:
                    matrix_norm[:, j] = matrix[:, j] / col_max

            fig, ax = plt.subplots(figsize=(10, 8))

            im = ax.imshow(matrix_norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

            # Set labels
            ax.set_xticks(np.arange(len(metrics)))
            ax.set_yticks(np.arange(len(sources)))
            ax.set_xticklabels(metrics, rotation=45, ha="right")
            ax.set_yticklabels(sources)

            # Add text annotations
            for i in range(len(sources)):
                for j in range(len(metrics)):
                    if j == 1:  # Removal rate as percentage
                        text = f"{matrix[i, j]:.1%}"
                    else:
                        text = f"{matrix[i, j]:.0f}"

                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        color="white" if matrix_norm[i, j] > 0.5 else "black",
                    )

            ax.set_title(
                "Data Source Reliability Matrix", fontsize=14, fontweight="bold"
            )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Normalized Value", rotation=270, labelpad=20)

            plt.tight_layout()

            # Save figure
            file_path = self.output_dir / "source_reliability_matrix.png"
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close()

            return str(file_path)

        except Exception as e:
            logger.error(f"Error creating source reliability matrix: {e}")
            plt.close()
            return None

    def _create_trajectory_fans(
        self,
        cohort_raw: Dict[str, pd.DataFrame],
        cohort_filtered: Dict[str, pd.DataFrame],
    ) -> Optional[str]:
        """Create improved trajectory comparison with focus on key time points."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 1. Side-by-side comparison with limited time range (0-180 days)
            ax1 = axes[0, 0]
            self._plot_trajectories_limited(
                ax1,
                cohort_raw,
                cohort_filtered,
                max_days=180,
                title="0-180 Day Trajectories",
            )

            # 2. Difference plot showing filtering impact
            ax2 = axes[0, 1]
            self._plot_trajectory_difference(ax2, cohort_raw, cohort_filtered)

            # 3. Success rate over time
            ax3 = axes[1, 0]
            self._plot_success_rate_progression(ax3, cohort_raw, cohort_filtered)

            # 4. Distribution at key checkpoints
            ax4 = axes[1, 1]
            self._plot_checkpoint_distributions(ax4, cohort_raw, cohort_filtered)

            fig.suptitle(
                "Weight Loss Trajectory Analysis: Impact of Filtering",
                fontsize=16,
                fontweight="bold",
            )
            plt.tight_layout()

            # Save figure
            file_path = self.output_dir / "trajectory_analysis.png"
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close()

            return str(file_path)

        except Exception as e:
            logger.error(f"Error creating trajectory analysis: {e}")
            plt.close()
            return None

    def _plot_trajectories_limited(
        self,
        ax,
        cohort_raw: Dict[str, pd.DataFrame],
        cohort_filtered: Dict[str, pd.DataFrame],
        max_days: int = 180,
        title: str = "Trajectory Comparison",
    ):
        """Plot trajectories with limited time range for better visibility."""

        # Calculate mean trajectories for both datasets
        def get_mean_trajectory(cohort_data, color, label_prefix):
            all_trajectories = []

            for user_id in cohort_data:
                df = cohort_data[user_id]
                if df.empty or len(df) < 2:
                    continue

                sorted_df = df.sort_values("timestamp")
                baseline = sorted_df["weight"].iloc[0]
                if baseline > 0:
                    days = (
                        sorted_df["timestamp"] - sorted_df["timestamp"].min()
                    ).dt.total_seconds() / 86400
                    # Limit to max_days
                    mask = days <= max_days
                    days = days[mask]
                    weights = sorted_df["weight"].values[mask]

                    if len(days) > 1:
                        pct_change = ((weights - baseline) / baseline) * 100
                        all_trajectories.append((days.values, pct_change))

            if all_trajectories:
                # Create common time grid
                time_grid = np.linspace(0, max_days, 50)
                interpolated = []

                for days, pct in all_trajectories:
                    if len(days) > 1:
                        interp_pct = np.interp(time_grid, days, pct)
                        interpolated.append(interp_pct)

                if interpolated:
                    mean_trajectory = np.mean(interpolated, axis=0)
                    std_trajectory = np.std(interpolated, axis=0)
                    sem_trajectory = std_trajectory / np.sqrt(len(interpolated))

                    # Plot mean with SEM (standard error)
                    ax.plot(
                        time_grid,
                        mean_trajectory,
                        color=color,
                        linewidth=2.5,
                        label=f"{label_prefix} (n={len(interpolated)})",
                    )
                    ax.fill_between(
                        time_grid,
                        mean_trajectory - 1.96 * sem_trajectory,
                        mean_trajectory + 1.96 * sem_trajectory,
                        color=color,
                        alpha=0.3,
                    )

                    return time_grid, mean_trajectory
            return None, None

        # Plot both trajectories
        get_mean_trajectory(cohort_raw, self.colors["raw"], "Raw")
        get_mean_trajectory(cohort_filtered, self.colors["filtered"], "Filtered")

        ax.set_xlabel("Days from Start", fontsize=12)
        ax.set_ylabel("Weight Change (%)", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax.axhline(
            y=-5, color="green", linestyle="--", alpha=0.5, label="5% loss target"
        )
        ax.axhline(
            y=-10, color="blue", linestyle="--", alpha=0.5, label="10% loss target"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_days)

    def _plot_trajectory_difference(
        self,
        ax,
        cohort_raw: Dict[str, pd.DataFrame],
        cohort_filtered: Dict[str, pd.DataFrame],
    ):
        """Plot the difference between raw and filtered trajectories."""

        checkpoints = [30, 60, 90, 120, 150, 180]
        raw_means = []
        filtered_means = []

        for checkpoint in checkpoints:
            raw_values = []
            filtered_values = []

            for user_id in cohort_raw:
                if user_id not in cohort_filtered:
                    continue

                raw_df = cohort_raw[user_id]
                filtered_df = cohort_filtered[user_id]

                if raw_df.empty or filtered_df.empty:
                    continue

                # Get weight change at checkpoint for raw data
                raw_sorted = raw_df.sort_values("timestamp")
                days = (
                    raw_sorted["timestamp"] - raw_sorted["timestamp"].min()
                ).dt.total_seconds() / 86400

                # Find measurement closest to checkpoint
                idx = np.argmin(np.abs(days - checkpoint))
                if abs(days.iloc[idx] - checkpoint) < 15:  # Within 15 days
                    baseline = raw_sorted["weight"].iloc[0]
                    if baseline > 0:
                        pct_change = (
                            (raw_sorted["weight"].iloc[idx] - baseline) / baseline
                        ) * 100
                        raw_values.append(pct_change)

                # Get weight change at checkpoint for filtered data
                filtered_sorted = filtered_df.sort_values("timestamp")
                days = (
                    filtered_sorted["timestamp"] - filtered_sorted["timestamp"].min()
                ).dt.total_seconds() / 86400

                if len(days) > 0:
                    idx = np.argmin(np.abs(days - checkpoint))
                    if abs(days.iloc[idx] - checkpoint) < 15:
                        baseline = filtered_sorted["weight"].iloc[0]
                        if baseline > 0:
                            pct_change = (
                                (filtered_sorted["weight"].iloc[idx] - baseline)
                                / baseline
                            ) * 100
                            filtered_values.append(pct_change)

            raw_means.append(np.mean(raw_values) if raw_values else 0)
            filtered_means.append(np.mean(filtered_values) if filtered_values else 0)

        # Plot difference
        differences = [f - r for r, f in zip(raw_means, filtered_means)]

        bars = ax.bar(
            checkpoints,
            differences,
            width=20,
            color=[
                self.colors["good"] if d < 0 else self.colors["warning"]
                for d in differences
            ],
        )

        ax.set_xlabel("Days in Program", fontsize=12)
        ax.set_ylabel("Difference in Mean Weight Loss (%)", fontsize=12)
        ax.set_title("Filtering Impact on Weight Loss (Filtered - Raw)", fontsize=12)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, diff in zip(bars, differences):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{diff:.1f}%",
                ha="center",
                va="bottom" if height > 0 else "top",
            )

    def _plot_success_rate_progression(
        self,
        ax,
        cohort_raw: Dict[str, pd.DataFrame],
        cohort_filtered: Dict[str, pd.DataFrame],
    ):
        """Plot success rate progression over time."""

        checkpoints = [30, 60, 90, 120, 150, 180]
        thresholds = [5, 10]  # 5% and 10% weight loss

        results = {threshold: {"raw": [], "filtered": []} for threshold in thresholds}

        for checkpoint in checkpoints:
            for threshold in thresholds:
                # Count successes for raw data
                raw_success = 0
                raw_total = 0

                for user_id in cohort_raw:
                    raw_df = cohort_raw[user_id]
                    if raw_df.empty:
                        continue

                    raw_sorted = raw_df.sort_values("timestamp")
                    days = (
                        raw_sorted["timestamp"] - raw_sorted["timestamp"].min()
                    ).dt.total_seconds() / 86400

                    # Find measurement closest to checkpoint
                    if len(days) > 0:
                        idx = np.argmin(np.abs(days - checkpoint))
                        if abs(days.iloc[idx] - checkpoint) < 15:  # Within 15 days
                            baseline = raw_sorted["weight"].iloc[0]
                            if baseline > 0:
                                pct_change = (
                                    (raw_sorted["weight"].iloc[idx] - baseline)
                                    / baseline
                                ) * 100
                                raw_total += 1
                                if pct_change <= -threshold:
                                    raw_success += 1

                # Count successes for filtered data
                filtered_success = 0
                filtered_total = 0

                for user_id in cohort_filtered:
                    filtered_df = cohort_filtered[user_id]
                    if filtered_df.empty:
                        continue

                    filtered_sorted = filtered_df.sort_values("timestamp")
                    days = (
                        filtered_sorted["timestamp"]
                        - filtered_sorted["timestamp"].min()
                    ).dt.total_seconds() / 86400

                    if len(days) > 0:
                        idx = np.argmin(np.abs(days - checkpoint))
                        if abs(days.iloc[idx] - checkpoint) < 15:
                            baseline = filtered_sorted["weight"].iloc[0]
                            if baseline > 0:
                                pct_change = (
                                    (filtered_sorted["weight"].iloc[idx] - baseline)
                                    / baseline
                                ) * 100
                                filtered_total += 1
                                if pct_change <= -threshold:
                                    filtered_success += 1

                results[threshold]["raw"].append(
                    (raw_success / raw_total * 100) if raw_total > 0 else 0
                )
                results[threshold]["filtered"].append(
                    (filtered_success / filtered_total * 100)
                    if filtered_total > 0
                    else 0
                )

        # Plot success rates
        for threshold, color in zip(thresholds, ["green", "blue"]):
            ax.plot(
                checkpoints,
                results[threshold]["raw"],
                "o--",
                color=color,
                alpha=0.5,
                label=f"{threshold}% Loss (Raw)",
            )
            ax.plot(
                checkpoints,
                results[threshold]["filtered"],
                "s-",
                color=color,
                label=f"{threshold}% Loss (Filtered)",
            )

        ax.set_xlabel("Days in Program", fontsize=12)
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_title("Clinical Success Rates Over Time", fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(20, 190)

    def _plot_checkpoint_distributions(
        self,
        ax,
        cohort_raw: Dict[str, pd.DataFrame],
        cohort_filtered: Dict[str, pd.DataFrame],
    ):
        """Plot weight change distributions at key checkpoints."""

        checkpoint_day = 90  # Focus on 90-day checkpoint

        raw_values = []
        filtered_values = []

        for user_id in cohort_raw:
            raw_df = cohort_raw[user_id]
            if raw_df.empty:
                continue

            raw_sorted = raw_df.sort_values("timestamp")
            days = (
                raw_sorted["timestamp"] - raw_sorted["timestamp"].min()
            ).dt.total_seconds() / 86400

            # Find measurement closest to 90 days
            if len(days) > 0:
                idx = np.argmin(np.abs(days - checkpoint_day))
                if abs(days.iloc[idx] - checkpoint_day) < 15:  # Within 15 days
                    baseline = raw_sorted["weight"].iloc[0]
                    if baseline > 0:
                        pct_change = (
                            (raw_sorted["weight"].iloc[idx] - baseline) / baseline
                        ) * 100
                        raw_values.append(pct_change)

        for user_id in cohort_filtered:
            filtered_df = cohort_filtered[user_id]
            if filtered_df.empty:
                continue

            filtered_sorted = filtered_df.sort_values("timestamp")
            days = (
                filtered_sorted["timestamp"] - filtered_sorted["timestamp"].min()
            ).dt.total_seconds() / 86400

            if len(days) > 0:
                idx = np.argmin(np.abs(days - checkpoint_day))
                if abs(days.iloc[idx] - checkpoint_day) < 15:
                    baseline = filtered_sorted["weight"].iloc[0]
                    if baseline > 0:
                        pct_change = (
                            (filtered_sorted["weight"].iloc[idx] - baseline) / baseline
                        ) * 100
                        filtered_values.append(pct_change)

        # Create violin plot
        parts = ax.violinplot(
            [raw_values, filtered_values],
            positions=[1, 2],
            widths=0.6,
            showmeans=True,
            showmedians=True,
        )

        # Customize colors
        colors = [self.colors["raw"], self.colors["filtered"]]
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Raw", "Filtered"])
        ax.set_ylabel("Weight Change (%)", fontsize=12)
        ax.set_title(f"Weight Change Distribution at Day {checkpoint_day}", fontsize=12)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax.axhline(y=-5, color="green", linestyle="--", alpha=0.5)
        ax.axhline(y=-10, color="blue", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        # Add statistics
        if raw_values and filtered_values:
            stats_text = (
                f"Raw: μ={np.mean(raw_values):.1f}%, σ={np.std(raw_values):.1f}%\n"
            )
            stats_text += f"Filtered: μ={np.mean(filtered_values):.1f}%, σ={np.std(filtered_values):.1f}%"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    def _create_impact_dashboard(self, cohort_metrics: Dict[str, Any]) -> Optional[str]:
        """Create dashboard showing medical and reporting impact."""
        try:
            fig = plt.figure(figsize=(16, 10))
            gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

            # 1. Success rate comparison (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            if "reporting" in cohort_metrics:
                reporting = cohort_metrics["reporting"]
                # Convert to dict if it's an object
                reporting_dict = (
                    reporting.to_dict() if hasattr(reporting, "to_dict") else reporting
                )

                categories = ["5% Loss", "10% Loss"]
                raw_rates = [
                    reporting_dict.get("success_rates", {})
                    .get("5pct_loss", {})
                    .get("raw", 0),
                    reporting_dict.get("success_rates", {})
                    .get("10pct_loss", {})
                    .get("raw", 0),
                ]
                filtered_rates = [
                    reporting_dict.get("success_rates", {})
                    .get("5pct_loss", {})
                    .get("filtered", 0),
                    reporting_dict.get("success_rates", {})
                    .get("10pct_loss", {})
                    .get("filtered", 0),
                ]

                x = np.arange(len(categories))
                width = 0.35

                bars1 = ax1.bar(
                    x - width / 2,
                    raw_rates,
                    width,
                    label="Raw",
                    color=self.colors["raw"],
                )
                bars2 = ax1.bar(
                    x + width / 2,
                    filtered_rates,
                    width,
                    label="Filtered",
                    color=self.colors["filtered"],
                )

                ax1.set_xlabel("Weight Loss Threshold")
                ax1.set_ylabel("Success Rate (%)")
                ax1.set_title("Clinical Success Rates")
                ax1.set_xticks(x)
                ax1.set_xticklabels(categories)
                ax1.legend()

                # Add value labels on bars
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax1.annotate(
                            f"{height:.1f}%",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                        )

            # 2. User inclusion funnel (top center)
            ax2 = fig.add_subplot(gs[0, 1])
            if "reporting" in cohort_metrics:
                reporting = cohort_metrics["reporting"]
                reporting_dict = (
                    reporting.to_dict() if hasattr(reporting, "to_dict") else reporting
                )
                inclusion = reporting_dict.get("inclusion", {})

                stages = [
                    "Baseline\n(Raw)",
                    "Baseline\n(Filtered)",
                    "Endpoint\n(Raw)",
                    "Endpoint\n(Filtered)",
                ]
                counts = [
                    inclusion.get("baseline", {}).get("raw", 0),
                    inclusion.get("baseline", {}).get("filtered", 0),
                    inclusion.get("endpoint", {}).get("raw", 0),
                    inclusion.get("endpoint", {}).get("filtered", 0),
                ]

                colors_list = [
                    self.colors["raw"],
                    self.colors["filtered"],
                    self.colors["raw"],
                    self.colors["filtered"],
                ]

                bars = ax2.bar(stages, counts, color=colors_list, alpha=0.7)
                ax2.set_ylabel("Valid Users")
                ax2.set_title("User Inclusion Funnel")

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax2.annotate(
                        f"{int(height)}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                    )

            # 3. Variance reduction (top right)
            ax3 = fig.add_subplot(gs[0, 2])
            if "aggregate" in cohort_metrics:
                aggregate = cohort_metrics["aggregate"]

                metrics_names = ["Removal\nRate", "Outlier\nRate", "CI\nImprovement"]
                values = [
                    aggregate.get("avg_removal_rate", 0) * 100,
                    aggregate.get("outlier_summary", {}).get("avg_outlier_rate", 0)
                    * 100,
                    aggregate.get("medical_summary", {}).get(
                        "avg_confidence_improvement", 0
                    )
                    * 100,
                ]

                bars = ax3.bar(metrics_names, values)
                ax3.set_ylabel("Percentage (%)")
                ax3.set_title("Key Improvements")

                # Color code bars
                for i, bar in enumerate(bars):
                    if i == 2:  # CI improvement is good
                        bar.set_color(self.colors["good"])
                    else:  # Others are neutral
                        bar.set_color(self.colors["filtered"])

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax3.annotate(
                        f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                    )

            # 4. Summary statistics table (bottom left)
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.axis("off")

            if "reporting" in cohort_metrics:
                reporting = cohort_metrics["reporting"]
                reporting_dict = (
                    reporting.to_dict() if hasattr(reporting, "to_dict") else reporting
                )
                cohort_stats = reporting_dict.get("cohort", {})

                text = (
                    f"Cohort Statistics\n"
                    f"{'=' * 25}\n"
                    f"Raw Mean Change: {cohort_stats.get('raw_mean', 0):.2f}%\n"
                    f"Filtered Mean Change: {cohort_stats.get('filtered_mean', 0):.2f}%\n"
                    f"Difference: {cohort_stats.get('difference', 0):.2f}%\n"
                    f"Percent Change: {cohort_stats.get('percent_change', 0):.1f}%\n\n"
                    f"Statistical Power\n"
                    f"{'=' * 25}\n"
                    f"Variance Reduction: {reporting_dict.get('power', {}).get('variance_reduction', 0):.1%}\n"
                    f"Effect Size Improvement: {reporting_dict.get('power', {}).get('effect_size_improvement', 0):.2f}"
                )

                ax4.text(
                    0.1,
                    0.9,
                    text,
                    transform=ax4.transAxes,
                    fontsize=11,
                    verticalalignment="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
                )

            # 5. Medical impact summary (bottom center)
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.axis("off")

            if "aggregate" in cohort_metrics:
                medical = cohort_metrics["aggregate"].get("medical_summary", {})
                temporal = cohort_metrics["aggregate"].get("temporal_summary", {})

                text = (
                    f"Clinical Impact\n"
                    f"{'=' * 25}\n"
                    f"Direction Errors: {medical.get('total_direction_errors', 0)}\n"
                    f"Avg CI Improvement: {medical.get('avg_confidence_improvement', 0):.1%}\n\n"
                    f"Temporal Consistency\n"
                    f"{'=' * 25}\n"
                    f"Avg Daily Volatility: {temporal.get('avg_daily_volatility', 0):.2f}kg\n"
                    f"Impossible Changes: {temporal.get('total_impossible_changes', 0)}"
                )

                ax5.text(
                    0.1,
                    0.9,
                    text,
                    transform=ax5.transAxes,
                    fontsize=11,
                    verticalalignment="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
                )

            # 6. Key findings (bottom right)
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.axis("off")

            if "aggregate" in cohort_metrics:
                outlier = cohort_metrics["aggregate"].get("outlier_summary", {})

                text = (
                    f"Key Findings\n"
                    f"{'=' * 25}\n"
                    f"Total Users: {cohort_metrics['aggregate'].get('total_users', 0)}\n"
                    f"Avg Removal Rate: {cohort_metrics['aggregate'].get('avg_removal_rate', 0):.1%}\n"
                    f"Total Outliers: {outlier.get('total_outliers', 0)}\n"
                    f"Avg Outlier Rate: {outlier.get('avg_outlier_rate', 0):.1%}"
                )

                ax6.text(
                    0.1,
                    0.9,
                    text,
                    transform=ax6.transAxes,
                    fontsize=11,
                    verticalalignment="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.3),
                )

            fig.suptitle("Filtering Impact Dashboard", fontsize=16, fontweight="bold")
            plt.tight_layout()

            # Save figure
            file_path = self.output_dir / "impact_dashboard.png"
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close()

            return str(file_path)

        except Exception as e:
            logger.error(f"Error creating impact dashboard: {e}")
            plt.close()
            return None

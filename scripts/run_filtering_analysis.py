#!/usr/bin/env python
"""
Run comprehensive filtering effectiveness analysis.
Processes sample data through both raw and filtered pipelines and generates reports.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import toml
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.filtering_effectiveness import FilteringAnalyzer
from src.analysis.visualization_generator import FilteringVisualizationGenerator
from src.analysis.quarterly_reporting import QuarterlyReportingAnalyzer
from src.analysis.quarterly_visualizations import QuarterlyVisualizationGenerator
from src.analysis.inline_charts import InlineChartGenerator
from src.database.database import get_state_db

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FilteringAnalysisRunner:
    """
    Orchestrates the complete filtering analysis workflow.
    """

    def __init__(self, config_path: str = "config.toml"):
        """
        Initialize the analysis runner.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.analyzer = FilteringAnalyzer(self.config)
        self.visualizer = FilteringVisualizationGenerator(
            output_dir=self.config.get("analysis", {}).get(
                "output_dir", "reports/visualizations"
            )
        )
        self.quarterly_analyzer = QuarterlyReportingAnalyzer(today_date="2025-09-05")
        base_dir = self.config.get("analysis", {}).get(
            "output_dir", "reports/visualizations"
        )
        self.quarterly_viz = QuarterlyVisualizationGenerator(
            output_dir=f"{base_dir}/quarterly"
        )
        self.db = get_state_db()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        try:
            with open(config_path, "r") as f:
                config = toml.load(f)

            # Add analysis-specific defaults if not present
            if "analysis" not in config:
                config["analysis"] = {}

            config["analysis"].setdefault("output_dir", "reports/visualizations")
            config["analysis"].setdefault("max_users", 10)
            config["analysis"].setdefault("min_measurements", 20)
            config["analysis"].setdefault("parallel_processing", True)

            return config

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def load_data_from_csv(
        self, csv_path: str, data_type: str = "raw", skip_user_limit: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Load weight data from CSV (raw or filtered).

        Args:
            csv_path: Path to CSV file
            data_type: Type of data ("raw" or "filtered")
            skip_user_limit: If True, don't apply max_users limit (for employer filtering)

        Returns:
            Dictionary of user_id -> DataFrame
        """
        try:
            logger.info(f"Loading {data_type} data from {csv_path}")

            # Read CSV
            df = pd.read_csv(csv_path)

            # Ensure required columns
            required_cols = ["user_id", "effectiveDateTime", "weight"]
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
                return {}

            # Parse timestamps
            df["timestamp"] = pd.to_datetime(df["effectiveDateTime"])

            # Clean data
            df = df.dropna(subset=["weight", "timestamp"])

            # Add a unique identifier for each measurement if not present
            # This helps track which measurements were filtered
            if "measurement_id" not in df.columns:
                df["measurement_id"] = (
                    df["user_id"].astype(str) + "_" + df["timestamp"].astype(str)
                )

            # Filter by date range if configured
            if "data" in self.config:
                if "min_date" in self.config["data"]:
                    min_date = pd.to_datetime(self.config["data"]["min_date"])
                    df = df[df["timestamp"] >= min_date]

                if "max_date" in self.config["data"]:
                    max_date = pd.to_datetime(self.config["data"]["max_date"])
                    df = df[df["timestamp"] <= max_date]

            # Group by user
            user_data = {}
            for user_id, user_df in df.groupby("user_id"):
                # Apply minimum measurements filter
                min_measurements = self.config.get("analysis", {}).get(
                    "min_measurements", 20
                )
                if len(user_df) >= min_measurements:
                    user_data[user_id] = user_df.sort_values("timestamp").reset_index(
                        drop=True
                    )

            # Limit number of users if configured (unless skipping for employer filter)
            if not skip_user_limit:
                max_users = self.config.get("analysis", {}).get("max_users", 10)
                if len(user_data) > max_users:
                    user_ids = list(user_data.keys())[:max_users]
                    user_data = {uid: user_data[uid] for uid in user_ids}
                    logger.info(
                        f"Loaded data for {len(user_data)} users (limited to {max_users})"
                    )
                else:
                    logger.info(f"Loaded data for {len(user_data)} users")
            else:
                logger.info(
                    f"Loaded data for {len(user_data)} users (no limit applied)"
                )
            return user_data

        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            return {}

    def load_employer_data(self, employer_path: str) -> Dict[str, str]:
        """
        Load employer data mapping.

        Args:
            employer_path: Path to employer CSV file

        Returns:
            Dictionary of user_id -> employer_id
        """
        try:
            if Path(employer_path).exists():
                logger.info(f"Loading employer data from {employer_path}")
                df = pd.read_csv(employer_path)

                # Check for different possible column names
                if "user_id" in df.columns and "employer_id" in df.columns:
                    return dict(zip(df["user_id"], df["employer_id"]))
                elif "user_id" in df.columns and "employer" in df.columns:
                    return dict(zip(df["user_id"], df["employer"]))
                else:
                    logger.warning(
                        f"Expected columns not found. Found: {df.columns.tolist()}"
                    )
            else:
                logger.warning(f"Employer file not found: {employer_path}")
        except Exception as e:
            logger.warning(f"Error loading employer data: {e}")
        return {}

    def load_partners_data(self, partners_path: str) -> List[str]:
        """
        Load partners data.

        Args:
            partners_path: Path to partners CSV file

        Returns:
            List of partner names
        """
        try:
            if Path(partners_path).exists():
                logger.info(f"Loading partners data from {partners_path}")
                df = pd.read_csv(partners_path)
                # Check for both 'partner' and 'name' columns
                if "partner" in df.columns:
                    return df["partner"].tolist()
                elif "name" in df.columns:
                    return df["name"].tolist()
                else:
                    logger.warning(
                        f"Expected 'partner' or 'name' column not found. Found columns: {df.columns.tolist()}"
                    )
                    return []
            logger.warning(f"Partners file not found: {partners_path}")
        except Exception as e:
            logger.warning(f"Error loading partners data: {e}")
        return []

    def run_analysis(
        self, raw_data: Dict[str, pd.DataFrame], filtered_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Run comprehensive analysis on raw and filtered data.

        Args:
            raw_data: Raw measurement data by user
            filtered_data: Filtered measurement data by user

        Returns:
            Analysis results
        """
        logger.info("Running filtering effectiveness analysis...")

        # Analyze cohort
        cohort_metrics = self.analyzer.analyze_cohort_data(raw_data, filtered_data)

        # Generate visualizations
        logger.info("Generating visualizations...")

        # Skip individual user visualizations - focus on cohort-level insights
        visualization_files = []
        logger.info(
            "Generating cohort-level visualizations only (per-user graphs disabled)"
        )

        # Cohort visualizations
        cohort_files = self.visualizer.generate_cohort_visualization_suite(
            raw_data, filtered_data, cohort_metrics
        )
        visualization_files.extend(cohort_files)

        # Add visualization paths to metrics
        cohort_metrics["visualizations"] = visualization_files

        return cohort_metrics

    def run_quarterly_analysis(
        self,
        raw_data: Dict[str, pd.DataFrame],
        filtered_data: Dict[str, pd.DataFrame],
        employer_csv_path: str = "data/2025-09-17-user-employers.csv",
        filter_to_users: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run quarterly reporting analysis for 90+ day users.

        Args:
            raw_data: Raw measurement data by user
            filtered_data: Filtered measurement data by user
            employer_csv_path: Path to employer CSV with start dates
            filter_to_users: Optional list of user IDs to limit analysis to

        Returns:
            Quarterly analysis results
        """
        logger.info("Running quarterly reporting analysis...")

        # Load program start dates
        start_dates_df = self.quarterly_analyzer.load_program_start_dates(
            employer_csv_path
        )

        if start_dates_df.empty:
            logger.warning("No start date data available for quarterly analysis")
            return {}

        # Filter to users we have weight data for
        users_with_data = set(raw_data.keys())
        start_dates_df = start_dates_df[start_dates_df["user_id"].isin(users_with_data)]

        # Further filter if specific users provided (e.g., for employer filtering)
        if filter_to_users:
            start_dates_df = start_dates_df[
                start_dates_df["user_id"].isin(filter_to_users)
            ]
            logger.info(
                f"Quarterly analysis limited to {len(filter_to_users)} specific users"
            )

        logger.info(
            f"Analyzing {len(start_dates_df)} users with start dates and weight data"
        )

        # 1. Analyze cohort progression (90-210 days)
        logger.info("Analyzing cohort progression at different time checkpoints...")
        cohort_results = self.quarterly_analyzer.analyze_cohort_by_duration(
            raw_data, filtered_data, start_dates_df
        )

        # 2. Analyze all 90+ day users
        logger.info("Analyzing all 90+ day users...")
        raw_metrics, filtered_metrics, results_df = (
            self.quarterly_analyzer.analyze_all_90plus_users(
                raw_data, filtered_data, start_dates_df
            )
        )

        # 3. Generate quarterly visualizations
        logger.info("Generating quarterly reporting visualizations...")
        viz_files = []

        # Weight loss distribution comparison
        viz_file = self.quarterly_viz.create_weight_loss_distribution_comparison(
            results_df, raw_metrics, filtered_metrics
        )
        if viz_file:
            viz_files.append(viz_file)

        # Clean weight loss progression chart (for embedding in report)
        viz_file = self.quarterly_viz.create_weight_loss_progression_chart(
            cohort_results
        )
        if viz_file:
            viz_files.append(viz_file)

        # Cohort progression analysis (detailed multi-panel)
        viz_file = self.quarterly_viz.create_cohort_progression_analysis(cohort_results)
        if viz_file:
            viz_files.append(viz_file)

        # Detailed metrics comparison
        viz_file = self.quarterly_viz.create_detailed_metrics_comparison(
            raw_metrics, filtered_metrics
        )
        if viz_file:
            viz_files.append(viz_file)

        # Impact summary dashboard
        viz_file = self.quarterly_viz.create_impact_summary_dashboard(
            raw_metrics, filtered_metrics, cohort_results
        )
        if viz_file:
            viz_files.append(viz_file)

        logger.info(f"Generated {len(viz_files)} quarterly visualizations")

        return {
            "cohort_results": cohort_results,
            "raw_metrics": raw_metrics,
            "filtered_metrics": filtered_metrics,
            "results_df": results_df,
            "visualizations": viz_files,
        }

    def generate_all_inline_charts(
        self,
        metrics: Dict[str, Any],
        raw_data: Dict[str, pd.DataFrame],
        filtered_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, str]:
        """Generate all inline charts for the report."""
        report_dir = Path(
            self.config.get("analysis", {}).get("output_dir", "reports/visualizations")
        )
        report_dir.mkdir(parents=True, exist_ok=True)

        charts = {}

        # Generate each chart
        logger.info("Generating inline visualizations...")

        # Executive Summary
        chart_path = self.generate_executive_summary_chart(metrics, report_dir)
        if chart_path:
            charts["executive_summary"] = chart_path

        # Weight Change Statistics
        chart_path = self.generate_weight_change_chart(metrics, report_dir)
        if chart_path:
            charts["weight_change"] = chart_path

        # Clinical Success Rates
        chart_path = self.generate_clinical_success_chart(metrics, report_dir)
        if chart_path:
            charts["clinical_success"] = chart_path

        # User Inclusion Funnel
        chart_path = self.generate_user_inclusion_funnel(metrics, report_dir)
        if chart_path:
            charts["user_inclusion"] = chart_path

        # Statistical Power
        chart_path = self.generate_statistical_power_chart(metrics, report_dir)
        if chart_path:
            charts["statistical_power"] = chart_path

        # Quarterly Data Quality
        if "quarterly" in metrics and metrics["quarterly"]:
            chart_path = self.generate_quarterly_data_quality_chart(
                metrics["quarterly"], report_dir
            )
            if chart_path:
                charts["quarterly_data_quality"] = chart_path

            # Quarterly Success Rates
            chart_path = self.generate_quarterly_success_rates_chart(
                metrics["quarterly"], report_dir
            )
            if chart_path:
                charts["quarterly_success_rates"] = chart_path

        # User Analysis Histograms
        chart_path = self.generate_user_analysis_histograms(
            metrics, raw_data, filtered_data, report_dir
        )
        if chart_path:
            charts["user_histograms"] = chart_path

        # Data Quality Summary
        chart_path = self.generate_data_quality_summary_chart(metrics, report_dir)
        if chart_path:
            charts["data_quality_summary"] = chart_path

        # Clinical Impact
        chart_path = self.generate_clinical_impact_chart(metrics, report_dir)
        if chart_path:
            charts["clinical_impact"] = chart_path

        logger.info(f"Generated {len(charts)} inline visualizations")
        return charts

    def generate_report(
        self,
        metrics: Dict[str, Any],
        raw_data: Dict[str, pd.DataFrame] = None,
        filtered_data: Dict[str, pd.DataFrame] = None,
    ) -> str:
        """
        Generate comprehensive markdown report.

        Args:
            metrics: Analysis metrics
            raw_data: Raw measurement data (optional, for histograms)
            filtered_data: Filtered measurement data (optional, for histograms)

        Returns:
            Path to generated report
        """
        logger.info("Generating analysis report...")

        # Use configured output directory
        report_dir = Path(
            self.config.get("analysis", {}).get("output_dir", "reports/visualizations")
        )
        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate all inline charts if data is provided
        inline_charts = {}
        if raw_data and filtered_data:
            inline_charts = self.generate_all_inline_charts(
                metrics, raw_data, filtered_data
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"filtering_analysis_{timestamp}.md"

        try:
            with open(report_path, "w") as f:
                f.write("# Comprehensive Filtering Effectiveness Analysis\n\n")
                f.write(
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                )
                f.write("---\n\n")

                # Analysis Overview
                f.write("## Analysis Overview\n\n")
                f.write(
                    "This report analyzes the effectiveness of our weight measurement filtering system, which uses "
                    "Kalman filtering and intelligent outlier detection to improve data quality while preserving "
                    "clinical validity. The analysis compares raw (unfiltered) weight measurements against filtered "
                    "data to quantify improvements in data reliability and reporting accuracy.\n\n"
                )

                f.write("### Methodology\n\n")
                f.write(
                    "- **Raw Data**: Original weight measurements from all sources without any filtering\n"
                )
                f.write(
                    "- **Filtered Data**: Measurements processed through our quality pipeline including:\n"
                )
                f.write("  - Adaptive Kalman filtering for noise reduction\n")
                f.write(
                    "  - Statistical outlier detection (IQR, MAD, temporal consistency)\n"
                )
                f.write("  - Source-specific reliability weighting\n")
                f.write("  - Quality score-based acceptance thresholds\n\n")

                # Executive Summary
                f.write("## Executive Summary\n\n")

                if "aggregate" in metrics:
                    agg = metrics["aggregate"]
                    f.write(
                        f"- **Total Users Analyzed**: {agg.get('total_users', 0)}\n"
                    )
                    f.write(
                        f"- **Average Removal Rate**: {agg.get('avg_removal_rate', 0):.1%}\n"
                    )
                    f.write(
                        f"- **Average Outlier Rate**: {agg.get('outlier_summary', {}).get('avg_outlier_rate', 0):.1%}\n"
                    )
                    f.write(
                        f"- **Average CI Improvement**: {agg.get('medical_summary', {}).get('avg_confidence_improvement', 0):.1%}\n\n"
                    )

                    # Add PNG visualization instead of ASCII art
                    if "executive_summary" in inline_charts:
                        f.write(
                            f"![Executive Summary Metrics]({inline_charts['executive_summary']})\n\n"
                        )
                    else:
                        # Fallback to ASCII if chart generation failed
                        removal_rate = agg.get("avg_removal_rate", 0) * 100
                        bar_width = int(removal_rate / 2)  # Scale to max 50 chars
                        f.write(f"**Data Quality Impact:**\n")
                        f.write(
                            f"```\nRemoval Rate: {'█' * bar_width} {removal_rate:.1f}%\n```\n\n"
                        )

                # Cohort Statistics
                if "reporting" in metrics:
                    f.write("## Cohort-Level Impact\n\n")

                    f.write(
                        "This section shows how filtering affects cohort-level reporting metrics that are critical "
                        "for clinical trials and population health studies.\n\n"
                    )

                    f.write("### Weight Change Statistics\n\n")
                    f.write(
                        "*These metrics show the average weight change across all users in the cohort.*\n\n"
                    )

                    reporting = metrics["reporting"]
                    cohort_dict = (
                        reporting.to_dict() if hasattr(reporting, "to_dict") else {}
                    )

                    f.write("| Metric | Raw | Filtered | Improvement |\n")
                    f.write("|--------|-----|----------|-------------|\n")

                    raw_mean = cohort_dict.get("cohort", {}).get("raw_mean", 0)
                    filtered_mean = cohort_dict.get("cohort", {}).get(
                        "filtered_mean", 0
                    )
                    difference = cohort_dict.get("cohort", {}).get("difference", 0)
                    f.write(
                        f"| Mean Weight Change | {raw_mean:.2f}% | {filtered_mean:.2f}% | "
                        f"{difference:.2f}% |\n\n"
                    )

                    # Add PNG visualization instead of ASCII art
                    if "weight_change" in inline_charts:
                        f.write(
                            f"![Weight Change Comparison]({inline_charts['weight_change']})\n\n"
                        )
                    else:
                        # Fallback to ASCII if chart generation failed
                        f.write("```\n")
                        raw_bar = "█" * int(abs(raw_mean) * 2)
                        filt_bar = "█" * int(abs(filtered_mean) * 2)
                        f.write(f"Raw:      {raw_bar} {raw_mean:.2f}%\n")
                        f.write(f"Filtered: {filt_bar} {filtered_mean:.2f}%\n")
                        if difference != 0:
                            improvement = (
                                "↑ Better"
                                if filtered_mean < raw_mean
                                else "↓ Worse"
                                if filtered_mean > raw_mean
                                else "→ Same"
                            )
                            f.write(
                                f"Impact:   {improvement} by {abs(difference):.2f}%\n"
                            )
                        f.write("```\n\n")

                    # Success rates
                    f.write("\n### Clinical Success Rates\n\n")
                    f.write(
                        "*Percentage of users achieving clinically significant weight loss thresholds.*\n\n"
                    )
                    f.write("| Threshold | Raw | Filtered | Delta |\n")
                    f.write("|-----------|-----|----------|-------|\n")

                    success = cohort_dict.get("success_rates", {})
                    pct_5_raw = success.get("5pct_loss", {}).get("raw", 0)
                    pct_5_filt = success.get("5pct_loss", {}).get("filtered", 0)
                    f.write(
                        f"| 5% Weight Loss | {pct_5_raw:.1f}% | {pct_5_filt:.1f}% | "
                        f"{pct_5_filt - pct_5_raw:+.1f}% |\n"
                    )

                    pct_10_raw = success.get("10pct_loss", {}).get("raw", 0)
                    pct_10_filt = success.get("10pct_loss", {}).get("filtered", 0)
                    f.write(
                        f"| 10% Weight Loss | {pct_10_raw:.1f}% | {pct_10_filt:.1f}% | "
                        f"{pct_10_filt - pct_10_raw:+.1f}% |\n\n"
                    )

                    # Add PNG visualization instead of ASCII art
                    if "clinical_success" in inline_charts:
                        f.write(
                            f"![Clinical Success Rates]({inline_charts['clinical_success']})\n\n"
                        )
                    else:
                        # Fallback to ASCII if chart generation failed
                        f.write("**Visual Comparison:**\n```\n")
                        f.write(
                            f"5% Success:  Raw    {'█' * int(pct_5_raw / 2)} {pct_5_raw:.1f}%\n"
                        )
                        f.write(
                            f"             Filtered {'█' * int(pct_5_filt / 2)} {pct_5_filt:.1f}%\n"
                        )
                        f.write(
                            f"10% Success: Raw    {'█' * int(pct_10_raw / 2)} {pct_10_raw:.1f}%\n"
                        )
                        f.write(
                            f"             Filtered {'█' * int(pct_10_filt / 2)} {pct_10_filt:.1f}%\n"
                        )
                        f.write("```\n\n")

                    # User inclusion
                    f.write("\n### User Inclusion Impact\n\n")
                    f.write(
                        "*How filtering affects the number of users with valid data for analysis.*\n\n"
                    )
                    f.write("| Stage | Raw | Filtered | Change |\n")
                    f.write("|-------|-----|----------|--------|\n")

                    inclusion = cohort_dict.get("inclusion", {})
                    baseline_raw = inclusion.get("baseline", {}).get("raw", 0)
                    baseline_filt = inclusion.get("baseline", {}).get("filtered", 0)
                    f.write(
                        f"| Valid Baseline | {baseline_raw} | {baseline_filt} | "
                        f"{baseline_filt - baseline_raw:+d} |\n"
                    )

                    endpoint_raw = inclusion.get("endpoint", {}).get("raw", 0)
                    endpoint_filt = inclusion.get("endpoint", {}).get("filtered", 0)
                    f.write(
                        f"| Valid Endpoint | {endpoint_raw} | {endpoint_filt} | "
                        f"{endpoint_filt - endpoint_raw:+d} |\n\n"
                    )

                    # Add PNG visualization instead of ASCII art
                    if "user_inclusion" in inline_charts:
                        f.write(
                            f"![Data Retention Funnel]({inline_charts['user_inclusion']})\n\n"
                        )
                    else:
                        # Fallback to ASCII if chart generation failed
                        if baseline_raw > 0:
                            f.write("**Data Retention Funnel:**\n```\n")
                            f.write(
                                f"Raw:      [{'█' * int((baseline_raw / baseline_raw) * 30)}] {baseline_raw} → "
                            )
                            f.write(
                                f"[{'█' * int((endpoint_raw / baseline_raw) * 30)}] {endpoint_raw}\n"
                            )
                            f.write(
                                f"Filtered: [{'█' * int((baseline_filt / baseline_raw) * 30)}] {baseline_filt} → "
                            )
                            f.write(
                                f"[{'█' * int((endpoint_filt / baseline_raw) * 30)}] {endpoint_filt}\n"
                            )
                            f.write("```\n\n")

                    # Statistical power
                    f.write("\n### Statistical Power Improvements\n\n")
                    f.write(
                        "*How filtering improves the statistical reliability of analyses.*\n\n"
                    )
                    power = cohort_dict.get("power", {})
                    var_reduction = power.get("variance_reduction", 0)
                    effect_improvement = power.get("effect_size_improvement", 0)
                    f.write(
                        f"- **Variance Reduction**: {var_reduction:.1%} - Lower variance means more consistent measurements\n"
                    )
                    f.write(
                        f"- **Effect Size Improvement**: {effect_improvement:.3f} - Larger effect sizes are easier to detect statistically\n\n"
                    )

                    # Add PNG visualization instead of ASCII art
                    if "statistical_power" in inline_charts:
                        f.write(
                            f"![Statistical Power Improvements]({inline_charts['statistical_power']})\n\n"
                        )
                    else:
                        # Fallback to ASCII if chart generation failed
                        f.write("**Statistical Power Boost:**\n```\n")
                        var_bar = int(var_reduction * 50)  # Scale to max 50 chars
                        effect_bar = int(
                            min(effect_improvement, 1.0) * 50
                        )  # Cap at 1.0 for display
                        f.write(
                            f"Variance Reduction: {'█' * var_bar} {var_reduction:.1%}\n"
                        )
                        f.write(
                            f"Effect Size:        {'█' * effect_bar} {effect_improvement:.2f}\n"
                        )
                        f.write("```\n\n")

                # Quarterly Reporting Analysis (90+ Day Users)
                if "quarterly" in metrics and metrics["quarterly"]:
                    quarterly = metrics["quarterly"]
                    f.write("## 📊 QUARTERLY REPORTING ANALYSIS\n\n")

                    f.write(
                        "This section analyzes users who have been in the program for 90+ days, which is the standard "
                        "timeframe for quarterly business reporting and clinical outcome assessment.\n\n"
                    )

                    f.write("### Key Business Question Answered\n\n")

                    if (
                        "filtered_metrics" in quarterly
                        and quarterly["filtered_metrics"]
                    ):
                        fm = quarterly["filtered_metrics"]
                        rm = quarterly["raw_metrics"]

                        f.write(
                            f'**"What is the average weight loss for users in the program for 90+ days?"**\n\n'
                        )

                        # Main answer box
                        f.write("| Metric | Raw Data | Filtered Data | Improvement |\n")
                        f.write("|--------|----------|---------------|-------------|\n")
                        f.write(
                            f"| **Average Weight Loss** | {rm.mean_weight_loss_pct:.2f}% | "
                            f"{fm.mean_weight_loss_pct:.2f}% | "
                            f"{fm.mean_weight_loss_pct - rm.mean_weight_loss_pct:+.2f}% |\n"
                        )
                        f.write(
                            f"| Median Weight Loss | {rm.median_weight_loss_pct:.2f}% | "
                            f"{fm.median_weight_loss_pct:.2f}% | "
                            f"{fm.median_weight_loss_pct - rm.median_weight_loss_pct:+.2f}% |\n"
                        )
                        f.write(
                            f"| Standard Deviation | {rm.std_weight_loss_pct:.2f}% | "
                            f"{fm.std_weight_loss_pct:.2f}% | "
                            f"{abs(rm.std_weight_loss_pct - fm.std_weight_loss_pct):.2f}% reduction |\n"
                        )
                        f.write("\n")

                        # Data quality impact
                        f.write("### Data Quality Impact\n\n")
                        f.write(
                            "*How many users have usable data for quarterly reporting.*\n\n"
                        )
                        f.write(
                            f"- **Eligible Users**: {rm.eligible_users} users with 90+ days in program\n"
                        )
                        f.write(
                            f"- **Valid Data (Raw)**: {rm.users_with_valid_endpoint} users ({rm.users_with_valid_endpoint / rm.eligible_users * 100:.1f}%)\n"
                        )
                        f.write(
                            f"- **Valid Data (Filtered)**: {fm.users_with_valid_endpoint} users ({fm.users_with_valid_endpoint / fm.eligible_users * 100:.1f}%)\n\n"
                        )

                        # Add PNG visualization
                        if "quarterly_data_quality" in inline_charts:
                            f.write(
                                f"![Quarterly Data Quality]({inline_charts['quarterly_data_quality']})\n\n"
                            )

                        # Success rates
                        f.write("### Clinical Success Rates (90+ Day Users)\n\n")
                        f.write(
                            "| Threshold | Raw Success Rate | Filtered Success Rate | Difference |\n"
                        )
                        f.write(
                            "|-----------|-----------------|----------------------|------------|\n"
                        )

                        raw_5pct = (
                            rm.users_losing_5pct / rm.users_with_valid_endpoint * 100
                            if rm.users_with_valid_endpoint > 0
                            else 0
                        )
                        filt_5pct = (
                            fm.users_losing_5pct / fm.users_with_valid_endpoint * 100
                            if fm.users_with_valid_endpoint > 0
                            else 0
                        )
                        f.write(
                            f"| 5% Loss | {raw_5pct:.1f}% ({rm.users_losing_5pct} users) | "
                            f"{filt_5pct:.1f}% ({fm.users_losing_5pct} users) | "
                            f"{filt_5pct - raw_5pct:+.1f}% |\n"
                        )

                        raw_10pct = (
                            rm.users_losing_10pct / rm.users_with_valid_endpoint * 100
                            if rm.users_with_valid_endpoint > 0
                            else 0
                        )
                        filt_10pct = (
                            fm.users_losing_10pct / fm.users_with_valid_endpoint * 100
                            if fm.users_with_valid_endpoint > 0
                            else 0
                        )
                        f.write(
                            f"| 10% Loss | {raw_10pct:.1f}% ({rm.users_losing_10pct} users) | "
                            f"{filt_10pct:.1f}% ({fm.users_losing_10pct} users) | "
                            f"{filt_10pct - raw_10pct:+.1f}% |\n"
                        )

                        raw_15pct = (
                            rm.users_losing_15pct / rm.users_with_valid_endpoint * 100
                            if rm.users_with_valid_endpoint > 0
                            else 0
                        )
                        filt_15pct = (
                            fm.users_losing_15pct / fm.users_with_valid_endpoint * 100
                            if fm.users_with_valid_endpoint > 0
                            else 0
                        )
                        f.write(
                            f"| 15% Loss | {raw_15pct:.1f}% ({rm.users_losing_15pct} users) | "
                            f"{filt_15pct:.1f}% ({fm.users_losing_15pct} users) | "
                            f"{filt_15pct - raw_15pct:+.1f}% |\n\n"
                        )

                        # Add PNG visualization
                        if "quarterly_success_rates" in inline_charts:
                            f.write(
                                f"![Quarterly Success Rates]({inline_charts['quarterly_success_rates']})\n\n"
                            )

                        # Cohort progression
                        if (
                            "cohort_results" in quarterly
                            and quarterly["cohort_results"]
                        ):
                            f.write(
                                "### Weight Loss Progression by Program Duration\n\n"
                            )

                            # Check if the progression chart exists and reference it
                            progression_chart_path = (
                                Path(
                                    self.config.get("analysis", {}).get(
                                        "output_dir", "reports/visualizations"
                                    )
                                )
                                / "quarterly"
                                / "weight_loss_progression_chart.png"
                            )
                            if progression_chart_path.exists():
                                f.write(
                                    f"![Weight Loss Progression Chart]({progression_chart_path})\n\n"
                                )

                            f.write(
                                "Average weight loss at different time checkpoints:\n\n"
                            )
                            f.write(
                                "| Days in Program | Raw Avg Loss | Filtered Avg Loss | Improvement |\n"
                            )
                            f.write(
                                "|-----------------|--------------|-------------------|-------------|\n"
                            )

                            for cohort in quarterly["cohort_results"]:
                                # Add visual indicator for improvement
                                improvement = cohort.mean_loss_difference
                                indicator = (
                                    "📈"
                                    if improvement > 0.1
                                    else "➡️"
                                    if improvement > 0
                                    else "📉"
                                )
                                f.write(
                                    f"| {cohort.day_checkpoint} days | "
                                    f"{cohort.raw_mean_loss_pct:.2f}% | "
                                    f"{cohort.filtered_mean_loss_pct:.2f}% | "
                                    f"{cohort.mean_loss_difference:+.2f}% {indicator} |\n"
                                )

                            # Add summary statistics
                            import numpy as np

                            avg_improvement = np.mean(
                                [
                                    c.mean_loss_difference
                                    for c in quarterly["cohort_results"]
                                ]
                            )
                            max_improvement = max(
                                c.mean_loss_difference
                                for c in quarterly["cohort_results"]
                            )
                            f.write("\n")
                            f.write(
                                f"**Average Improvement Across All Checkpoints:** {avg_improvement:+.2f}%\n"
                            )
                            f.write(
                                f"**Maximum Improvement:** {max_improvement:+.2f}% at "
                            )
                            max_cohort = max(
                                quarterly["cohort_results"],
                                key=lambda c: c.mean_loss_difference,
                            )
                            f.write(f"{max_cohort.day_checkpoint} days\n")
                            f.write("\n")

                        # Visualizations
                        if (
                            "visualizations" in quarterly
                            and quarterly["visualizations"]
                        ):
                            f.write("### Quarterly Reporting Visualizations\n\n")
                            f.write(
                                "The following visualizations have been generated:\n\n"
                            )
                            for viz_path in quarterly["visualizations"]:
                                viz_name = Path(viz_path).name
                                f.write(f"- `{viz_path}` - {viz_name}\n")
                            f.write("\n")

                    f.write("\n")

                # Individual User Analysis Summary
                if "users" in metrics and metrics["users"]:
                    f.write("## Individual User Analysis\n\n")
                    total_users = metrics.get("aggregate", {}).get(
                        "total_users", len(metrics["users"])
                    )
                    f.write(f"Analyzed {total_users} users in total.\n")
                    f.write(
                        f"Detailed metrics calculated for all {len(metrics['users'])} users.\n\n"
                    )

                    # Create summary table
                    f.write(
                        "| User ID | Measurements | Filtered | Removal Rate | Outlier Rate |\n"
                    )
                    f.write(
                        "|---------|--------------|----------|--------------|-------------|\n"
                    )

                    for user_metrics in metrics["users"][:10]:  # Show first 10
                        user_id = user_metrics["user_id"][:8]
                        raw_count = user_metrics["data_summary"]["raw"].get("count", 0)
                        filt_count = user_metrics["data_summary"]["filtered"].get(
                            "count", 0
                        )
                        removal_rate = user_metrics["data_summary"].get(
                            "removal_rate", 0
                        )
                        outlier_rate = user_metrics.get("outliers", {}).get(
                            "outlier_rate", 0
                        )

                        f.write(
                            f"| {user_id} | {raw_count} | {filt_count} | "
                            f"{removal_rate:.1%} | {outlier_rate:.1%} |\n"
                        )

                    f.write("\n")

                    # Add PNG visualization for user analysis
                    if "user_histograms" in inline_charts:
                        f.write(
                            f"![User Analysis Distributions]({inline_charts['user_histograms']})\n\n"
                        )

                # Key Findings
                f.write("## Key Findings & Interpretation\n\n")

                if "aggregate" in metrics:
                    agg = metrics["aggregate"]

                    f.write("### Data Quality Improvements\n\n")
                    f.write(
                        "*These metrics show how filtering improves the reliability of weight measurements.*\n\n"
                    )
                    f.write(
                        f"1. **Outlier Detection**: Successfully identified and removed "
                        f"{agg.get('outlier_summary', {}).get('total_outliers', 0)} outliers "
                        f"across all users\n"
                    )
                    f.write(
                        f"2. **Temporal Consistency**: Reduced daily weight volatility by an average of "
                        f"{agg.get('temporal_summary', {}).get('avg_daily_volatility', 0):.2f}kg\n"
                    )
                    f.write(
                        f"3. **Impossible Changes**: Eliminated "
                        f"{agg.get('temporal_summary', {}).get('total_impossible_changes', 0)} "
                        f"physiologically impossible weight changes\n\n"
                    )

                    # Add PNG visualization for data quality summary
                    if "data_quality_summary" in inline_charts:
                        f.write(
                            f"![Data Quality Summary]({inline_charts['data_quality_summary']})\n\n"
                        )

                    f.write("### Clinical Impact\n\n")
                    f.write(
                        "*How filtering prevents medical misinterpretations and improves clinical decision-making.*\n\n"
                    )
                    f.write(
                        f"1. **Direction Errors**: Prevented "
                        f"{agg.get('medical_summary', {}).get('total_direction_errors', 0)} "
                        f"cases where weight change direction would be misclassified (e.g., showing gain instead of loss)\n"
                    )
                    f.write(
                        f"2. **Confidence Intervals**: Improved measurement confidence by "
                        f"{agg.get('medical_summary', {}).get('avg_confidence_improvement', 0):.1%} "
                        f"on average (tighter confidence bands mean more reliable measurements)\n\n"
                    )

                    # Add PNG visualization for clinical impact
                    if "clinical_impact" in inline_charts:
                        f.write(
                            f"![Clinical Impact]({inline_charts['clinical_impact']})\n\n"
                        )

                # Visualizations
                if "visualizations" in metrics and metrics["visualizations"]:
                    f.write("## Generated Visualizations\n\n")
                    f.write(
                        "The following visualization files have been generated:\n\n"
                    )

                    for viz_path in metrics["visualizations"]:
                        viz_name = Path(viz_path).name
                        f.write(f"- `{viz_path}` - {viz_name}\n")

                    f.write("\n")

                # Recommendations
                f.write("## Recommendations\n\n")
                f.write("Based on the analysis results, we recommend:\n\n")

                f.write(
                    "1. **Continue Filtering**: The filtering process significantly improves data quality "
                    "without compromising clinical validity\n"
                )
                f.write(
                    "2. **Source Monitoring**: Pay special attention to data sources with high outlier rates\n"
                )
                f.write(
                    "3. **Threshold Tuning**: Consider adjusting quality thresholds based on source reliability\n"
                )
                f.write(
                    "4. **Regular Validation**: Implement periodic manual review of filtered data\n\n"
                )

                f.write("### How to Interpret These Results\n\n")
                f.write(
                    "- **Higher filtered success rates**: More accurate assessment of true program effectiveness\n"
                )
                f.write(
                    "- **Reduced variance**: More reliable individual measurements and trend detection\n"
                )
                f.write(
                    "- **Improved mean weight loss**: Removal of erroneous measurements reveals true outcomes\n"
                )
                f.write(
                    "- **Better statistical power**: Easier to detect real changes and treatment effects\n\n"
                )

                # Technical Details
                f.write("## Technical Details\n\n")
                f.write("### Configuration Used\n\n")
                f.write("```toml\n")
                if "processing" in self.config:
                    f.write(
                        f"quality_threshold = {self.config['processing'].get('quality_threshold', 0.45)}\n"
                    )
                if "kalman" in self.config:
                    f.write(
                        f"initial_variance = {self.config['kalman'].get('initial_variance', 1.0)}\n"
                    )
                f.write("```\n\n")

                f.write("### Analysis Parameters\n\n")
                f.write(f"- Analysis timestamp: {metrics.get('timestamp', 'N/A')}\n")
                f.write(f"- Cohort size: {metrics.get('cohort_size', 0)} users\n")
                f.write(
                    f"- Minimum measurements per user: "
                    f"{self.config.get('analysis', {}).get('min_measurements', 20)}\n"
                )

                f.write("\n---\n\n")
                f.write("*End of Report*\n")

            logger.info(f"Report saved to {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""

    def save_metrics_json(self, metrics: Dict[str, Any]) -> str:
        """
        Save detailed metrics as JSON for further analysis.

        Args:
            metrics: Analysis metrics

        Returns:
            Path to saved JSON file
        """
        try:
            # Use configured output directory
            report_dir = Path(
                self.config.get("analysis", {}).get(
                    "output_dir", "reports/visualizations"
                )
            )
            report_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = report_dir / f"filtering_metrics_{timestamp}.json"

            # Convert any non-serializable objects
            def make_serializable(obj):
                if hasattr(obj, "to_dict"):
                    return obj.to_dict()
                elif hasattr(obj, "__dict__"):
                    return obj.__dict__
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                else:
                    return str(obj)

            # Clean metrics for JSON serialization
            clean_metrics = json.loads(json.dumps(metrics, default=make_serializable))

            with open(json_path, "w") as f:
                json.dump(clean_metrics, f, indent=2)

            logger.info(f"Metrics saved to {json_path}")
            return str(json_path)

        except Exception as e:
            logger.error(f"Error saving metrics JSON: {e}")
            return ""

    def save_user_metrics_csv(
        self,
        metrics: Dict[str, Any],
        raw_data: Dict[str, pd.DataFrame],
        filtered_data: Dict[str, pd.DataFrame],
    ) -> str:
        """
        Save per-user analysis results as CSV for further analysis.

        Args:
            metrics: Analysis metrics
            raw_data: Raw measurement data by user
            filtered_data: Filtered measurement data by user

        Returns:
            Path to saved CSV file
        """
        try:
            # Use configured output directory
            report_dir = Path(
                self.config.get("analysis", {}).get(
                    "output_dir", "reports/visualizations"
                )
            )
            report_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = report_dir / f"user_analysis_results_{timestamp}.csv"

            # Build rows for CSV
            rows = []

            for user_id in raw_data.keys():
                row = {
                    "user_id": user_id,
                    "raw_measurement_count": len(raw_data[user_id]),
                    "filtered_measurement_count": len(filtered_data.get(user_id, [])),
                    "removal_rate": 0,
                    "impact_score": 0,
                    "raw_mean_weight": 0,
                    "filtered_mean_weight": 0,
                    "raw_std_weight": 0,
                    "filtered_std_weight": 0,
                    "raw_min_weight": 0,
                    "raw_max_weight": 0,
                    "filtered_min_weight": 0,
                    "filtered_max_weight": 0,
                    "raw_weight_change_pct": 0,
                    "filtered_weight_change_pct": 0,
                    "outlier_count": 0,
                    "outlier_rate": 0,
                    "data_duration_days": 0,
                    "raw_daily_volatility": 0,
                    "filtered_daily_volatility": 0,
                }

                # Calculate metrics for raw data
                raw_df = raw_data[user_id]
                if not raw_df.empty:
                    row["raw_mean_weight"] = raw_df["weight"].mean()
                    row["raw_std_weight"] = raw_df["weight"].std()
                    row["raw_min_weight"] = raw_df["weight"].min()
                    row["raw_max_weight"] = raw_df["weight"].max()

                    # Calculate weight change percentage
                    if len(raw_df) > 1:
                        sorted_df = raw_df.sort_values("timestamp")
                        baseline = sorted_df["weight"].iloc[0]
                        endpoint = sorted_df["weight"].iloc[-1]
                        if baseline > 0:
                            row["raw_weight_change_pct"] = (
                                (endpoint - baseline) / baseline
                            ) * 100

                        # Calculate daily volatility
                        daily_changes = np.diff(sorted_df["weight"].values)
                        row["raw_daily_volatility"] = (
                            np.std(daily_changes) if len(daily_changes) > 0 else 0
                        )

                    # Data duration
                    row["data_duration_days"] = (
                        raw_df["timestamp"].max() - raw_df["timestamp"].min()
                    ).days

                # Calculate metrics for filtered data
                if user_id in filtered_data:
                    filtered_df = filtered_data[user_id]
                    if not filtered_df.empty:
                        row["filtered_mean_weight"] = filtered_df["weight"].mean()
                        row["filtered_std_weight"] = filtered_df["weight"].std()
                        row["filtered_min_weight"] = filtered_df["weight"].min()
                        row["filtered_max_weight"] = filtered_df["weight"].max()

                        # Calculate weight change percentage
                        if len(filtered_df) > 1:
                            sorted_df = filtered_df.sort_values("timestamp")
                            baseline = sorted_df["weight"].iloc[0]
                            endpoint = sorted_df["weight"].iloc[-1]
                            if baseline > 0:
                                row["filtered_weight_change_pct"] = (
                                    (endpoint - baseline) / baseline
                                ) * 100

                            # Calculate daily volatility
                            daily_changes = np.diff(sorted_df["weight"].values)
                            row["filtered_daily_volatility"] = (
                                np.std(daily_changes) if len(daily_changes) > 0 else 0
                            )

                # Calculate removal metrics
                row["removal_rate"] = (
                    (row["raw_measurement_count"] - row["filtered_measurement_count"])
                    / row["raw_measurement_count"]
                    if row["raw_measurement_count"] > 0
                    else 0
                )
                row["outlier_count"] = (
                    row["raw_measurement_count"] - row["filtered_measurement_count"]
                )
                row["outlier_rate"] = (
                    row["outlier_count"] / row["raw_measurement_count"]
                    if row["raw_measurement_count"] > 0
                    else 0
                )

                # Calculate impact score (removal rate + variance reduction)
                if user_id in filtered_data and not filtered_data[user_id].empty:
                    variance_reduction = (
                        (row["raw_std_weight"] - row["filtered_std_weight"])
                        / row["raw_std_weight"]
                        if row["raw_std_weight"] > 0
                        else 0
                    )
                    row["impact_score"] = row["removal_rate"] + abs(variance_reduction)
                else:
                    row["impact_score"] = row["removal_rate"]

                # Add individual user metrics if available
                if "users" in metrics:
                    user_metrics = next(
                        (m for m in metrics["users"] if m["user_id"] == user_id), None
                    )
                    if user_metrics:
                        # Add additional metrics from the analysis
                        if "medical_impact" in user_metrics:
                            row["direction_errors"] = (
                                user_metrics["medical_impact"]
                                .get("clinical", {})
                                .get("direction_errors", 0)
                            )
                            row["confidence_improvement"] = (
                                user_metrics["medical_impact"]
                                .get("confidence", {})
                                .get("ci_reduction", 0)
                            )

                        if "temporal" in user_metrics:
                            row["impossible_changes"] = (
                                user_metrics["temporal"]
                                .get("daily_change", {})
                                .get("impossible_count", 0)
                            )
                            row["max_daily_change"] = (
                                user_metrics["temporal"]
                                .get("daily_change", {})
                                .get("max", 0)
                            )

                rows.append(row)

            # Create DataFrame and save to CSV
            results_df = pd.DataFrame(rows)

            # Sort by impact score for easier analysis
            results_df = results_df.sort_values("impact_score", ascending=False)

            # Save to CSV
            results_df.to_csv(csv_path, index=False)

            logger.info(f"User metrics CSV saved to {csv_path}")
            return str(csv_path)

        except Exception as e:
            logger.error(f"Error saving user metrics CSV: {e}")
            return ""

    def generate_executive_summary_chart(
        self, metrics: Dict[str, Any], output_dir: Path
    ) -> str:
        """Generate executive summary metrics visualization."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            if "aggregate" in metrics:
                agg = metrics["aggregate"]

                # Prepare data
                labels = [
                    "Total Users\n(count)",
                    "Avg Removal\nRate (%)",
                    "Avg Outlier\nRate (%)",
                    "Avg CI\nImprovement (%)",
                ]
                values = [
                    agg.get("total_users", 0) / 100,  # Scale for visibility
                    agg.get("avg_removal_rate", 0) * 100,
                    agg.get("outlier_summary", {}).get("avg_outlier_rate", 0) * 100,
                    agg.get("medical_summary", {}).get("avg_confidence_improvement", 0),
                ]
                colors = ["#2E86AB", "#A23B72", "#F18F01", "#73AB84"]

                # Create bars
                bars = ax.bar(
                    labels,
                    values,
                    color=colors,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                )

                # Add value labels on bars
                for bar, value, label in zip(bars, values, labels):
                    if "Total Users" in label:
                        display_val = f"{int(value * 100)}"
                    else:
                        display_val = f"{value:.1f}%"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        display_val,
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

                ax.set_title(
                    "Executive Summary - Key Metrics", fontsize=14, fontweight="bold"
                )
                ax.set_ylabel("Value", fontsize=12)
                ax.grid(axis="y", alpha=0.3)
                ax.set_ylim(0, max(values) * 1.15)

            plt.tight_layout()
            chart_path = output_dir / "executive_summary_metrics.png"
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close()

            return str(chart_path)
        except Exception as e:
            logger.error(f"Error generating executive summary chart: {e}")
            return ""

    def generate_weight_change_chart(
        self, metrics: Dict[str, Any], output_dir: Path
    ) -> str:
        """Generate weight change statistics comparison chart."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))

            if "reporting" in metrics:
                reporting = metrics["reporting"]
                cohort_dict = (
                    reporting.to_dict() if hasattr(reporting, "to_dict") else {}
                )

                raw_mean = cohort_dict.get("cohort", {}).get("raw_mean", 0)
                filtered_mean = cohort_dict.get("cohort", {}).get("filtered_mean", 0)

                # Data
                categories = ["Raw Data", "Filtered Data"]
                values = [raw_mean, filtered_mean]
                colors = ["#E74C3C", "#2ECC71"]

                # Create bars
                bars = ax.bar(
                    categories,
                    values,
                    color=colors,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                )

                # Add value labels
                for bar, val in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        f"{val:.2f}%",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        fontweight="bold",
                    )

                # Add improvement indicator
                difference = filtered_mean - raw_mean
                ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

                # Improvement annotation
                if difference != 0:
                    improvement_text = f"Improvement: {abs(difference):.2f}%"
                    ax.text(
                        0.5,
                        min(values) - 0.5,
                        improvement_text,
                        ha="center",
                        transform=ax.transData,
                        fontsize=11,
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3
                        ),
                    )

                ax.set_title(
                    "Mean Weight Change Comparison", fontsize=14, fontweight="bold"
                )
                ax.set_ylabel("Weight Change (%)", fontsize=12)
                ax.grid(axis="y", alpha=0.3)
                ax.set_ylim(min(values) - 1, max(0, max(values)) + 0.5)

            plt.tight_layout()
            chart_path = output_dir / "weight_change_comparison.png"
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close()

            return str(chart_path)
        except Exception as e:
            logger.error(f"Error generating weight change chart: {e}")
            return ""

    def generate_clinical_success_chart(
        self, metrics: Dict[str, Any], output_dir: Path
    ) -> str:
        """Generate clinical success rates comparison chart."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            if "reporting" in metrics:
                reporting = metrics["reporting"]
                cohort_dict = (
                    reporting.to_dict() if hasattr(reporting, "to_dict") else {}
                )
                success = cohort_dict.get("success_rates", {})

                # Data preparation
                thresholds = ["5% Weight Loss", "10% Weight Loss"]
                raw_rates = [
                    success.get("5pct_loss", {}).get("raw", 0),
                    success.get("10pct_loss", {}).get("raw", 0),
                ]
                filtered_rates = [
                    success.get("5pct_loss", {}).get("filtered", 0),
                    success.get("10pct_loss", {}).get("filtered", 0),
                ]

                x = np.arange(len(thresholds))
                width = 0.35

                # Create bars
                rects1 = ax.bar(
                    x - width / 2,
                    raw_rates,
                    width,
                    label="Raw Data",
                    color="#E74C3C",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                )
                rects2 = ax.bar(
                    x + width / 2,
                    filtered_rates,
                    width,
                    label="Filtered Data",
                    color="#2ECC71",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                )

                # Add value labels
                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        ax.annotate(
                            f"{height:.1f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontweight="bold",
                        )

                autolabel(rects1)
                autolabel(rects2)

                ax.set_xlabel("Success Threshold", fontsize=12)
                ax.set_ylabel("Success Rate (%)", fontsize=12)
                ax.set_title(
                    "Clinical Success Rates Comparison", fontsize=14, fontweight="bold"
                )
                ax.set_xticks(x)
                ax.set_xticklabels(thresholds)
                ax.legend(loc="upper right")
                ax.grid(axis="y", alpha=0.3)
                ax.set_ylim(0, max(max(raw_rates), max(filtered_rates)) * 1.15)

            plt.tight_layout()
            chart_path = output_dir / "clinical_success_rates.png"
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close()

            return str(chart_path)
        except Exception as e:
            logger.error(f"Error generating clinical success chart: {e}")
            return ""

    def generate_user_inclusion_funnel(
        self, metrics: Dict[str, Any], output_dir: Path
    ) -> str:
        """Generate user inclusion funnel visualization."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            if "reporting" in metrics:
                reporting = metrics["reporting"]
                cohort_dict = (
                    reporting.to_dict() if hasattr(reporting, "to_dict") else {}
                )
                inclusion = cohort_dict.get("inclusion", {})

                # Data
                baseline_raw = inclusion.get("baseline", {}).get("raw", 0)
                baseline_filt = inclusion.get("baseline", {}).get("filtered", 0)
                endpoint_raw = inclusion.get("endpoint", {}).get("raw", 0)
                endpoint_filt = inclusion.get("endpoint", {}).get("filtered", 0)

                # Create funnel chart
                stages = ["Baseline", "Endpoint"]
                raw_values = [baseline_raw, endpoint_raw]
                filtered_values = [baseline_filt, endpoint_filt]

                x = np.arange(len(stages))
                width = 0.35

                rects1 = ax.bar(
                    x - width / 2,
                    raw_values,
                    width,
                    label="Raw Data",
                    color="#3498DB",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                )
                rects2 = ax.bar(
                    x + width / 2,
                    filtered_values,
                    width,
                    label="Filtered Data",
                    color="#9B59B6",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                )

                # Add value labels
                for rect in rects1 + rects2:
                    height = rect.get_height()
                    ax.annotate(
                        f"{int(height)}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

                # Add retention percentage
                if baseline_raw > 0:
                    raw_retention = (endpoint_raw / baseline_raw) * 100
                    filtered_retention = (
                        (endpoint_filt / baseline_filt) * 100
                        if baseline_filt > 0
                        else 0
                    )

                    ax.text(
                        0.5,
                        -0.15,
                        f"Raw Retention: {raw_retention:.1f}%",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=10,
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5
                        ),
                    )
                    ax.text(
                        0.5,
                        -0.22,
                        f"Filtered Retention: {filtered_retention:.1f}%",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=10,
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="lavender", alpha=0.5
                        ),
                    )

                ax.set_xlabel("Stage", fontsize=12)
                ax.set_ylabel("Number of Users", fontsize=12)
                ax.set_title("Data Retention Funnel", fontsize=14, fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels(stages)
                ax.legend(loc="upper right")
                ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            chart_path = output_dir / "user_inclusion_funnel.png"
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close()

            return str(chart_path)
        except Exception as e:
            logger.error(f"Error generating user inclusion funnel: {e}")
            return ""

    def generate_statistical_power_chart(
        self, metrics: Dict[str, Any], output_dir: Path
    ) -> str:
        """Generate statistical power improvements visualization."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            if "reporting" in metrics:
                reporting = metrics["reporting"]
                cohort_dict = (
                    reporting.to_dict() if hasattr(reporting, "to_dict") else {}
                )
                power = cohort_dict.get("power", {})

                var_reduction = power.get("variance_reduction", 0) * 100
                effect_improvement = power.get("effect_size_improvement", 0)

                # Variance Reduction Chart
                ax1.barh(
                    ["Variance\nReduction"],
                    [var_reduction],
                    color="#16A085",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                )
                ax1.set_xlim(0, 100)
                ax1.set_xlabel("Reduction (%)", fontsize=11)
                ax1.set_title("Variance Reduction", fontsize=12, fontweight="bold")
                ax1.text(
                    var_reduction + 2,
                    0,
                    f"{var_reduction:.1f}%",
                    va="center",
                    fontweight="bold",
                )
                ax1.grid(axis="x", alpha=0.3)

                # Effect Size Improvement Chart
                ax2.barh(
                    ["Effect Size\nImprovement"],
                    [effect_improvement],
                    color="#E67E22",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                )
                ax2.set_xlim(0, max(1.0, effect_improvement * 1.2))
                ax2.set_xlabel("Effect Size (Cohen's d)", fontsize=11)
                ax2.set_title("Effect Size Improvement", fontsize=12, fontweight="bold")
                ax2.text(
                    effect_improvement + 0.02,
                    0,
                    f"{effect_improvement:.3f}",
                    va="center",
                    fontweight="bold",
                )
                ax2.grid(axis="x", alpha=0.3)

            plt.suptitle(
                "Statistical Power Improvements", fontsize=14, fontweight="bold"
            )
            plt.tight_layout()
            chart_path = output_dir / "statistical_power_improvements.png"
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close()

            return str(chart_path)
        except Exception as e:
            logger.error(f"Error generating statistical power chart: {e}")
            return ""

    def generate_quarterly_data_quality_chart(
        self, quarterly: Dict[str, Any], output_dir: Path
    ) -> str:
        """Generate quarterly data quality comparison chart."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            if "filtered_metrics" in quarterly and quarterly["filtered_metrics"]:
                fm = quarterly["filtered_metrics"]
                rm = quarterly["raw_metrics"]

                # Data preparation
                categories = [
                    "Eligible Users",
                    "Valid Data (Raw)",
                    "Valid Data (Filtered)",
                ]
                values = [
                    rm.eligible_users,
                    rm.users_with_valid_endpoint,
                    fm.users_with_valid_endpoint,
                ]
                colors = ["#3498DB", "#E74C3C", "#2ECC71"]

                # Create bars
                bars = ax.bar(
                    categories,
                    values,
                    color=colors,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                )

                # Add value labels
                for bar, val in zip(bars, values):
                    percentage = (
                        (val / rm.eligible_users * 100) if rm.eligible_users > 0 else 0
                    )
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 10,
                        f"{int(val)}\n({percentage:.1f}%)",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

                ax.set_title(
                    "Quarterly Reporting - Data Quality", fontsize=14, fontweight="bold"
                )
                ax.set_ylabel("Number of Users", fontsize=12)
                ax.grid(axis="y", alpha=0.3)
                ax.set_ylim(0, max(values) * 1.15)

            plt.tight_layout()
            chart_path = output_dir / "quarterly_data_quality.png"
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close()

            return str(chart_path)
        except Exception as e:
            logger.error(f"Error generating quarterly data quality chart: {e}")
            return ""

    def generate_quarterly_success_rates_chart(
        self, quarterly: Dict[str, Any], output_dir: Path
    ) -> str:
        """Generate 90+ day clinical success rates visualization."""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            if "filtered_metrics" in quarterly and quarterly["filtered_metrics"]:
                fm = quarterly["filtered_metrics"]
                rm = quarterly["raw_metrics"]

                # Calculate success rates
                raw_5pct = (
                    rm.users_losing_5pct / rm.users_with_valid_endpoint * 100
                    if rm.users_with_valid_endpoint > 0
                    else 0
                )
                filt_5pct = (
                    fm.users_losing_5pct / fm.users_with_valid_endpoint * 100
                    if fm.users_with_valid_endpoint > 0
                    else 0
                )
                raw_10pct = (
                    rm.users_losing_10pct / rm.users_with_valid_endpoint * 100
                    if rm.users_with_valid_endpoint > 0
                    else 0
                )
                filt_10pct = (
                    fm.users_losing_10pct / fm.users_with_valid_endpoint * 100
                    if fm.users_with_valid_endpoint > 0
                    else 0
                )
                raw_15pct = (
                    rm.users_losing_15pct / rm.users_with_valid_endpoint * 100
                    if rm.users_with_valid_endpoint > 0
                    else 0
                )
                filt_15pct = (
                    fm.users_losing_15pct / fm.users_with_valid_endpoint * 100
                    if fm.users_with_valid_endpoint > 0
                    else 0
                )

                # Data preparation
                thresholds = ["5% Loss", "10% Loss", "15% Loss"]
                raw_rates = [raw_5pct, raw_10pct, raw_15pct]
                filtered_rates = [filt_5pct, filt_10pct, filt_15pct]
                raw_counts = [
                    rm.users_losing_5pct,
                    rm.users_losing_10pct,
                    rm.users_losing_15pct,
                ]
                filt_counts = [
                    fm.users_losing_5pct,
                    fm.users_losing_10pct,
                    fm.users_losing_15pct,
                ]

                x = np.arange(len(thresholds))
                width = 0.35

                # Create bars
                rects1 = ax.bar(
                    x - width / 2,
                    raw_rates,
                    width,
                    label="Raw Data",
                    color="#E74C3C",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                )
                rects2 = ax.bar(
                    x + width / 2,
                    filtered_rates,
                    width,
                    label="Filtered Data",
                    color="#2ECC71",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                )

                # Add value labels with counts
                for i, (rect1, rect2) in enumerate(zip(rects1, rects2)):
                    # Raw data labels
                    height1 = rect1.get_height()
                    ax.annotate(
                        f"{height1:.1f}%\n({raw_counts[i]})",
                        xy=(rect1.get_x() + rect1.get_width() / 2, height1),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=10,
                    )
                    # Filtered data labels
                    height2 = rect2.get_height()
                    ax.annotate(
                        f"{height2:.1f}%\n({filt_counts[i]})",
                        xy=(rect2.get_x() + rect2.get_width() / 2, height2),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=10,
                    )

                ax.set_xlabel("Weight Loss Threshold", fontsize=12)
                ax.set_ylabel("Success Rate (%)", fontsize=12)
                ax.set_title(
                    "Clinical Success Rates - 90+ Day Users",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.set_xticks(x)
                ax.set_xticklabels(thresholds)
                ax.legend(loc="upper right")
                ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            chart_path = output_dir / "quarterly_success_rates.png"
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close()

            return str(chart_path)
        except Exception as e:
            logger.error(f"Error generating quarterly success rates chart: {e}")
            return ""

    def generate_user_analysis_histograms(
        self,
        metrics: Dict[str, Any],
        raw_data: Dict[str, pd.DataFrame],
        filtered_data: Dict[str, pd.DataFrame],
        output_dir: Path,
    ) -> str:
        """Generate histograms for user removal and outlier rates."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Calculate removal and outlier rates for all users
            removal_rates = []
            outlier_rates = []

            for user_id in raw_data.keys():
                raw_count = len(raw_data[user_id])
                filt_count = len(filtered_data.get(user_id, []))
                removal_rate = (
                    ((raw_count - filt_count) / raw_count * 100) if raw_count > 0 else 0
                )
                removal_rates.append(removal_rate)
                outlier_rates.append(
                    removal_rate
                )  # Same as removal rate in this context

            # Removal Rate Histogram
            ax1.hist(
                removal_rates,
                bins=20,
                color="#E74C3C",
                alpha=0.7,
                edgecolor="black",
                linewidth=1.2,
            )
            ax1.axvline(
                np.mean(removal_rates),
                color="red",
                linestyle="dashed",
                linewidth=2,
                label=f"Mean: {np.mean(removal_rates):.1f}%",
            )
            ax1.set_xlabel("Removal Rate (%)", fontsize=11)
            ax1.set_ylabel("Number of Users", fontsize=11)
            ax1.set_title(
                "Distribution of Removal Rates", fontsize=12, fontweight="bold"
            )
            ax1.legend()
            ax1.grid(axis="y", alpha=0.3)

            # Outlier Rate Histogram
            ax2.hist(
                outlier_rates,
                bins=20,
                color="#F39C12",
                alpha=0.7,
                edgecolor="black",
                linewidth=1.2,
            )
            ax2.axvline(
                np.mean(outlier_rates),
                color="orange",
                linestyle="dashed",
                linewidth=2,
                label=f"Mean: {np.mean(outlier_rates):.1f}%",
            )
            ax2.set_xlabel("Outlier Rate (%)", fontsize=11)
            ax2.set_ylabel("Number of Users", fontsize=11)
            ax2.set_title(
                "Distribution of Outlier Rates", fontsize=12, fontweight="bold"
            )
            ax2.legend()
            ax2.grid(axis="y", alpha=0.3)

            plt.suptitle(
                "User-Level Data Quality Analysis", fontsize=14, fontweight="bold"
            )
            plt.tight_layout()
            chart_path = output_dir / "user_analysis_histograms.png"
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close()

            return str(chart_path)
        except Exception as e:
            logger.error(f"Error generating user analysis histograms: {e}")
            return ""

    def generate_data_quality_summary_chart(
        self, metrics: Dict[str, Any], output_dir: Path
    ) -> str:
        """Generate data quality improvements summary visualization."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            if "aggregate" in metrics:
                agg = metrics["aggregate"]

                # Prepare data
                categories = [
                    "Outliers\nDetected",
                    "Daily Volatility\nReduction (kg)",
                    "Impossible\nChanges",
                ]
                values = [
                    agg.get("outlier_summary", {}).get("total_outliers", 0)
                    / 100,  # Scale for visibility
                    agg.get("temporal_summary", {}).get("avg_daily_volatility", 0)
                    * 10,  # Scale up
                    agg.get("temporal_summary", {}).get("total_impossible_changes", 0)
                    / 100,  # Scale
                ]
                actual_values = [
                    agg.get("outlier_summary", {}).get("total_outliers", 0),
                    agg.get("temporal_summary", {}).get("avg_daily_volatility", 0),
                    agg.get("temporal_summary", {}).get("total_impossible_changes", 0),
                ]
                colors = ["#E74C3C", "#3498DB", "#F39C12"]

                bars = ax.bar(
                    categories,
                    values,
                    color=colors,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                )

                # Add actual value labels
                for bar, actual_val, cat in zip(bars, actual_values, categories):
                    if "Volatility" in cat:
                        label = f"{actual_val:.2f} kg"
                    else:
                        label = f"{int(actual_val)}"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        label,
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

                ax.set_title(
                    "Data Quality Improvements Summary", fontsize=14, fontweight="bold"
                )
                ax.set_ylabel("Scaled Value", fontsize=12)
                ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            chart_path = output_dir / "data_quality_summary.png"
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close()

            return str(chart_path)
        except Exception as e:
            logger.error(f"Error generating data quality summary chart: {e}")
            return ""

    def generate_clinical_impact_chart(
        self, metrics: Dict[str, Any], output_dir: Path
    ) -> str:
        """Generate clinical impact visualization."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            if "aggregate" in metrics:
                agg = metrics["aggregate"]

                # Direction Errors Chart
                direction_errors = agg.get("medical_summary", {}).get(
                    "total_direction_errors", 0
                )
                ax1.bar(
                    ["Direction\nErrors\nPrevented"],
                    [direction_errors],
                    color="#E74C3C",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                )
                ax1.set_ylabel("Count", fontsize=11)
                ax1.set_title(
                    "Weight Change Direction Errors Prevented",
                    fontsize=12,
                    fontweight="bold",
                )
                ax1.text(
                    0,
                    direction_errors + 0.5,
                    f"{int(direction_errors)}",
                    ha="center",
                    fontweight="bold",
                    fontsize=14,
                )
                ax1.grid(axis="y", alpha=0.3)

                # Confidence Improvement Chart
                ci_improvement = agg.get("medical_summary", {}).get(
                    "avg_confidence_improvement", 0
                )
                ax2.bar(
                    ["Confidence\nInterval\nImprovement"],
                    [ci_improvement],
                    color="#27AE60",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                )
                ax2.set_ylabel("Improvement (%)", fontsize=11)
                ax2.set_title(
                    "Measurement Confidence Improvement", fontsize=12, fontweight="bold"
                )
                ax2.text(
                    0,
                    ci_improvement + 0.1,
                    f"{ci_improvement:.1f}%",
                    ha="center",
                    fontweight="bold",
                    fontsize=14,
                )
                ax2.grid(axis="y", alpha=0.3)

            plt.suptitle("Clinical Impact of Filtering", fontsize=14, fontweight="bold")
            plt.tight_layout()
            chart_path = output_dir / "clinical_impact.png"
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close()

            return str(chart_path)
        except Exception as e:
            logger.error(f"Error generating clinical impact chart: {e}")
            return ""


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive filtering effectiveness analysis"
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="data/2025-09-05_all.csv",
        help="Path to CSV file with weight measurements (default: data/2025-09-05_all.csv)",
    )
    parser.add_argument(
        "--filtered",
        default="data/2025-09-05_all_filtered.csv",
        help="Path to filtered CSV file (default: data/2025-09-05_all_filtered.csv)",
    )
    parser.add_argument(
        "--employer",
        default="data/2025-09-17-user-employers.csv",
        help="Path to employer data file (default: data/2025-09-17-user-employers.csv)",
    )
    parser.add_argument(
        "--partners",
        default="data/partners.csv",
        help="Path to partners data file (default: data/partners.csv)",
    )
    parser.add_argument(
        "--filter-employer", help="Filter analysis to only users from this employer"
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to configuration file (default: config.toml)",
    )
    parser.add_argument(
        "--max-users", type=int, help="Maximum number of users to analyze"
    )
    parser.add_argument("--output-dir", help="Directory for output files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize runner
    runner = FilteringAnalysisRunner(args.config)

    # Override config with command line args
    if args.max_users:
        runner.config.setdefault("analysis", {})["max_users"] = args.max_users
    if args.output_dir:
        runner.config.setdefault("analysis", {})["output_dir"] = args.output_dir
        # Recreate visualizer with new output directory
        runner.visualizer = FilteringVisualizationGenerator(output_dir=args.output_dir)
        # Also update quarterly visualizer
        runner.quarterly_viz = QuarterlyVisualizationGenerator(
            output_dir=f"{args.output_dir}/quarterly"
        )

    # Determine if we should skip user limit (when filtering by employer)
    skip_limit = bool(args.filter_employer)

    # Load raw data
    raw_data = runner.load_data_from_csv(
        args.csv_file, "raw", skip_user_limit=skip_limit
    )
    if not raw_data:
        logger.error("No raw data loaded. Exiting.")
        return 1

    # Load pre-filtered data if available
    filtered_data = {}
    if Path(args.filtered).exists():
        logger.info("Loading pre-filtered data...")
        filtered_data_full = runner.load_data_from_csv(
            args.filtered, "filtered", skip_user_limit=skip_limit
        )

        # The filtered dataset is a subset of raw - align them properly
        filtered_data = {}
        for user_id in raw_data.keys():
            if user_id in filtered_data_full:
                # Get the filtered subset for this user
                filtered_df = filtered_data_full[user_id]
                raw_df = raw_data[user_id]

                # The filtered data should be a subset of raw data
                # Match by timestamp (since filtered removed some measurements)
                filtered_timestamps = set(filtered_df["timestamp"])

                # Keep only the filtered measurements
                filtered_data[user_id] = filtered_df

                # Log removal statistics
                removed_count = len(raw_df) - len(filtered_df)
                removal_rate = removed_count / len(raw_df) if len(raw_df) > 0 else 0
                logger.info(
                    f"User {user_id[:8]}: {len(raw_df)} raw -> {len(filtered_df)} filtered "
                    f"({removal_rate:.1%} removed)"
                )

        if skip_limit:
            logger.info(
                f"Loaded filtered data for {len(filtered_data)} users (all users, no limit applied)"
            )
        else:
            logger.info(f"Loaded filtered data for {len(filtered_data)} users")
    else:
        logger.warning(f"Filtered data file not found: {args.filtered}")
        logger.info("Processing raw data through filtering pipeline...")
        from src.processing.processor import process_measurement
        from src.database.database import get_state_db

        filtered_data = {}
        db = get_state_db()

        for user_id, raw_df in raw_data.items():
            logger.info(
                f"Processing user {user_id[:8]}... ({len(raw_df)} measurements)"
            )

            filtered_measurements = []

            # Process each measurement through the pipeline
            for _, row in raw_df.iterrows():
                try:
                    result = process_measurement(
                        user_id=user_id,
                        weight=row["weight"],
                        timestamp=row["timestamp"],
                        source=row.get("source", "unknown"),
                        config=runner.config,
                        unit=row.get("unit", "kg"),
                        db=db,
                    )

                    # Only keep accepted measurements
                    if result.get("accepted", False):
                        filtered_measurements.append(
                            {
                                "timestamp": row["timestamp"],
                                "weight": result.get("filtered_weight", row["weight"]),
                                "source": row.get("source", "unknown"),
                                "quality_score": result.get("quality_score", 0),
                                "unit": row.get("unit", "kg"),
                            }
                        )

                except Exception as e:
                    logger.warning(
                        f"Error processing measurement for user {user_id}: {e}"
                    )
                    continue

            # Create filtered DataFrame
            if filtered_measurements:
                filtered_df = pd.DataFrame(filtered_measurements)
                filtered_data[user_id] = filtered_df.sort_values(
                    "timestamp"
                ).reset_index(drop=True)
                logger.info(f"  Filtered: {len(filtered_df)} measurements retained")

    # Load auxiliary data if available
    employer_data = runner.load_employer_data(args.employer)
    partners_data = runner.load_partners_data(args.partners)

    if employer_data:
        logger.info(f"Loaded employer data for {len(employer_data)} users")
    if partners_data:
        logger.info(f"Loaded {len(partners_data)} partners")

    # Filter by employer if requested
    if args.filter_employer and employer_data:
        logger.info(f"Filtering analysis to employer: {args.filter_employer}")

        # Find users belonging to the specified employer
        target_users = {
            uid
            for uid, emp in employer_data.items()
            if emp and str(emp).lower() == args.filter_employer.lower()
        }

        if not target_users:
            logger.error(f"No users found for employer: {args.filter_employer}")
            logger.info("Available employers:")
            unique_employers = set(emp for emp in employer_data.values() if emp)
            for emp in sorted(unique_employers)[:20]:  # Show first 20
                logger.info(f"  - {emp}")
            return 1

        # Filter both raw and filtered data to only include these users
        raw_data = {uid: data for uid, data in raw_data.items() if uid in target_users}
        filtered_data = {
            uid: data for uid, data in filtered_data.items() if uid in target_users
        }

        logger.info(
            f"Filtering to employer {args.filter_employer}: {len(raw_data)} users found"
        )
        logger.info(f"Will analyze ALL {len(raw_data)} employer users")

        if not raw_data:
            logger.error(f"No data available for users from {args.filter_employer}")
            return 1

    # Run analysis
    metrics = runner.run_analysis(raw_data, filtered_data)

    # Run quarterly analysis
    # Pass the filtered user list if we're filtering by employer
    filter_users = list(raw_data.keys()) if args.filter_employer else None
    quarterly_metrics = runner.run_quarterly_analysis(
        raw_data, filtered_data, args.employer, filter_users
    )
    metrics["quarterly"] = quarterly_metrics

    # Generate outputs (pass raw_data and filtered_data for generating visualizations)
    report_path = runner.generate_report(metrics, raw_data, filtered_data)
    json_path = runner.save_metrics_json(metrics)
    csv_path = runner.save_user_metrics_csv(metrics, raw_data, filtered_data)

    logger.info("=" * 60)
    logger.info("Analysis Complete!")
    logger.info(f"Report: {report_path}")
    logger.info(f"Metrics (JSON): {json_path}")
    logger.info(f"User Results (CSV): {csv_path}")
    logger.info(
        f"Visualizations: {runner.config.get('analysis', {}).get('output_dir', 'reports/visualizations')}"
    )
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

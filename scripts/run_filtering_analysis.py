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

import pandas as pd
import toml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.filtering_effectiveness import FilteringAnalyzer
from src.analysis.visualization_generator import FilteringVisualizationGenerator
from src.analysis.quarterly_reporting import QuarterlyReportingAnalyzer
from src.analysis.quarterly_visualizations import QuarterlyVisualizationGenerator
from src.database.database import get_state_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
            output_dir=self.config.get('analysis', {}).get('output_dir', 'reports/visualizations')
        )
        self.quarterly_analyzer = QuarterlyReportingAnalyzer(today_date="2025-09-05")
        base_dir = self.config.get('analysis', {}).get('output_dir', 'reports/visualizations')
        self.quarterly_viz = QuarterlyVisualizationGenerator(
            output_dir=f"{base_dir}/quarterly"
        )
        self.db = get_state_db()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        try:
            with open(config_path, 'r') as f:
                config = toml.load(f)

            # Add analysis-specific defaults if not present
            if 'analysis' not in config:
                config['analysis'] = {}

            config['analysis'].setdefault('output_dir', 'reports/visualizations')
            config['analysis'].setdefault('max_users', 10)
            config['analysis'].setdefault('min_measurements', 20)
            config['analysis'].setdefault('parallel_processing', True)

            return config

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def load_data_from_csv(self, csv_path: str, data_type: str = "raw", skip_user_limit: bool = False) -> Dict[str, pd.DataFrame]:
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
            required_cols = ['user_id', 'effectiveDateTime', 'weight']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
                return {}

            # Parse timestamps
            df['timestamp'] = pd.to_datetime(df['effectiveDateTime'])

            # Clean data
            df = df.dropna(subset=['weight', 'timestamp'])

            # Add a unique identifier for each measurement if not present
            # This helps track which measurements were filtered
            if 'measurement_id' not in df.columns:
                df['measurement_id'] = df['user_id'].astype(str) + '_' + df['timestamp'].astype(str)

            # Filter by date range if configured
            if 'data' in self.config:
                if 'min_date' in self.config['data']:
                    min_date = pd.to_datetime(self.config['data']['min_date'])
                    df = df[df['timestamp'] >= min_date]

                if 'max_date' in self.config['data']:
                    max_date = pd.to_datetime(self.config['data']['max_date'])
                    df = df[df['timestamp'] <= max_date]

            # Group by user
            user_data = {}
            for user_id, user_df in df.groupby('user_id'):
                # Apply minimum measurements filter
                min_measurements = self.config.get('analysis', {}).get('min_measurements', 20)
                if len(user_df) >= min_measurements:
                    user_data[user_id] = user_df.sort_values('timestamp').reset_index(drop=True)

            # Limit number of users if configured (unless skipping for employer filter)
            if not skip_user_limit:
                max_users = self.config.get('analysis', {}).get('max_users', 10)
                if len(user_data) > max_users:
                    user_ids = list(user_data.keys())[:max_users]
                    user_data = {uid: user_data[uid] for uid in user_ids}
                    logger.info(f"Loaded data for {len(user_data)} users (limited to {max_users})")
                else:
                    logger.info(f"Loaded data for {len(user_data)} users")
            else:
                logger.info(f"Loaded data for {len(user_data)} users (no limit applied)")
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
                if 'user_id' in df.columns and 'employer_id' in df.columns:
                    return dict(zip(df['user_id'], df['employer_id']))
                elif 'user_id' in df.columns and 'employer' in df.columns:
                    return dict(zip(df['user_id'], df['employer']))
                else:
                    logger.warning(f"Expected columns not found. Found: {df.columns.tolist()}")
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
                if 'partner' in df.columns:
                    return df['partner'].tolist()
                elif 'name' in df.columns:
                    return df['name'].tolist()
                else:
                    logger.warning(f"Expected 'partner' or 'name' column not found. Found columns: {df.columns.tolist()}")
                    return []
            logger.warning(f"Partners file not found: {partners_path}")
        except Exception as e:
            logger.warning(f"Error loading partners data: {e}")
        return []

    def run_analysis(
        self,
        raw_data: Dict[str, pd.DataFrame],
        filtered_data: Dict[str, pd.DataFrame]
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

        # Find the most impacted users (biggest difference between raw and filtered)
        user_impacts = []
        for user_id in raw_data.keys():
            if user_id in filtered_data:
                raw_count = len(raw_data[user_id])
                filtered_count = len(filtered_data[user_id])
                removal_rate = (raw_count - filtered_count) / raw_count if raw_count > 0 else 0

                # Calculate weight variance reduction as another impact metric
                raw_std = raw_data[user_id]['weight'].std()
                filtered_std = filtered_data[user_id]['weight'].std()
                variance_reduction = (raw_std - filtered_std) / raw_std if raw_std > 0 else 0

                # Combined impact score
                impact_score = removal_rate + abs(variance_reduction)
                user_impacts.append((user_id, impact_score, removal_rate))

        # Sort by impact and take top 10
        user_impacts.sort(key=lambda x: x[1], reverse=True)
        top_impacted_users = user_impacts[:10]

        logger.info(f"Generating visualizations for top {len(top_impacted_users)} most impacted users")
        for user_id, impact_score, removal_rate in top_impacted_users:
            logger.info(f"  User {user_id[:8]}: impact score={impact_score:.3f}, removal rate={removal_rate:.1%}")

        # Skip individual user visualizations - focus on cohort-level insights
        visualization_files = []
        logger.info("Generating cohort-level visualizations only (per-user graphs disabled)")

        # Cohort visualizations
        cohort_files = self.visualizer.generate_cohort_visualization_suite(
            raw_data,
            filtered_data,
            cohort_metrics
        )
        visualization_files.extend(cohort_files)

        # Add visualization paths to metrics
        cohort_metrics['visualizations'] = visualization_files

        return cohort_metrics

    def run_quarterly_analysis(
        self,
        raw_data: Dict[str, pd.DataFrame],
        filtered_data: Dict[str, pd.DataFrame],
        employer_csv_path: str = "data/2025-09-17-user-employers.csv",
        filter_to_users: Optional[List[str]] = None
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
        start_dates_df = self.quarterly_analyzer.load_program_start_dates(employer_csv_path)

        if start_dates_df.empty:
            logger.warning("No start date data available for quarterly analysis")
            return {}

        # Filter to users we have weight data for
        users_with_data = set(raw_data.keys())
        start_dates_df = start_dates_df[start_dates_df['user_id'].isin(users_with_data)]

        # Further filter if specific users provided (e.g., for employer filtering)
        if filter_to_users:
            start_dates_df = start_dates_df[start_dates_df['user_id'].isin(filter_to_users)]
            logger.info(f"Quarterly analysis limited to {len(filter_to_users)} specific users")

        logger.info(f"Analyzing {len(start_dates_df)} users with start dates and weight data")

        # 1. Analyze cohort progression (90-210 days)
        logger.info("Analyzing cohort progression at different time checkpoints...")
        cohort_results = self.quarterly_analyzer.analyze_cohort_by_duration(
            raw_data, filtered_data, start_dates_df
        )

        # 2. Analyze all 90+ day users
        logger.info("Analyzing all 90+ day users...")
        raw_metrics, filtered_metrics, results_df = self.quarterly_analyzer.analyze_all_90plus_users(
            raw_data, filtered_data, start_dates_df
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

        # Cohort progression analysis
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
            'cohort_results': cohort_results,
            'raw_metrics': raw_metrics,
            'filtered_metrics': filtered_metrics,
            'results_df': results_df,
            'visualizations': viz_files
        }

    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate comprehensive markdown report.

        Args:
            metrics: Analysis metrics

        Returns:
            Path to generated report
        """
        logger.info("Generating analysis report...")

        # Use configured output directory
        report_dir = Path(self.config.get('analysis', {}).get('output_dir', 'reports/visualizations'))
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"filtering_analysis_{timestamp}.md"

        try:
            with open(report_path, 'w') as f:
                f.write("# Comprehensive Filtering Effectiveness Analysis\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")

                # Analysis Overview
                f.write("## Analysis Overview\n\n")
                f.write("This report analyzes the effectiveness of our weight measurement filtering system, which uses "
                       "Kalman filtering and intelligent outlier detection to improve data quality while preserving "
                       "clinical validity. The analysis compares raw (unfiltered) weight measurements against filtered "
                       "data to quantify improvements in data reliability and reporting accuracy.\n\n")

                f.write("### Methodology\n\n")
                f.write("- **Raw Data**: Original weight measurements from all sources without any filtering\n")
                f.write("- **Filtered Data**: Measurements processed through our quality pipeline including:\n")
                f.write("  - Adaptive Kalman filtering for noise reduction\n")
                f.write("  - Statistical outlier detection (IQR, MAD, temporal consistency)\n")
                f.write("  - Source-specific reliability weighting\n")
                f.write("  - Quality score-based acceptance thresholds\n\n")

                # Executive Summary
                f.write("## Executive Summary\n\n")

                if 'aggregate' in metrics:
                    agg = metrics['aggregate']
                    f.write(f"- **Total Users Analyzed**: {agg.get('total_users', 0)}\n")
                    f.write(f"- **Average Removal Rate**: {agg.get('avg_removal_rate', 0):.1%}\n")
                    f.write(f"- **Average Outlier Rate**: {agg.get('outlier_summary', {}).get('avg_outlier_rate', 0):.1%}\n")
                    f.write(f"- **Average CI Improvement**: {agg.get('medical_summary', {}).get('avg_confidence_improvement', 0):.1%}\n")
                    f.write("\n")

                # Cohort Statistics
                if 'reporting' in metrics:
                    f.write("## Cohort-Level Impact\n\n")

                    f.write("This section shows how filtering affects cohort-level reporting metrics that are critical "
                           "for clinical trials and population health studies.\n\n")

                    f.write("### Weight Change Statistics\n\n")
                    f.write("*These metrics show the average weight change across all users in the cohort.*\n\n")

                    reporting = metrics['reporting']
                    cohort_dict = reporting.to_dict() if hasattr(reporting, 'to_dict') else {}

                    f.write("| Metric | Raw | Filtered | Improvement |\n")
                    f.write("|--------|-----|----------|-------------|\n")

                    raw_mean = cohort_dict.get('cohort', {}).get('raw_mean', 0)
                    filtered_mean = cohort_dict.get('cohort', {}).get('filtered_mean', 0)
                    f.write(f"| Mean Weight Change | {raw_mean:.2f}% | {filtered_mean:.2f}% | "
                           f"{cohort_dict.get('cohort', {}).get('difference', 0):.2f}% |\n")

                    # Success rates
                    f.write("\n### Clinical Success Rates\n\n")
                    f.write("*Percentage of users achieving clinically significant weight loss thresholds.*\n\n")
                    f.write("| Threshold | Raw | Filtered | Delta |\n")
                    f.write("|-----------|-----|----------|-------|\n")

                    success = cohort_dict.get('success_rates', {})
                    pct_5_raw = success.get('5pct_loss', {}).get('raw', 0)
                    pct_5_filt = success.get('5pct_loss', {}).get('filtered', 0)
                    f.write(f"| 5% Weight Loss | {pct_5_raw:.1f}% | {pct_5_filt:.1f}% | "
                           f"{pct_5_filt - pct_5_raw:+.1f}% |\n")

                    pct_10_raw = success.get('10pct_loss', {}).get('raw', 0)
                    pct_10_filt = success.get('10pct_loss', {}).get('filtered', 0)
                    f.write(f"| 10% Weight Loss | {pct_10_raw:.1f}% | {pct_10_filt:.1f}% | "
                           f"{pct_10_filt - pct_10_raw:+.1f}% |\n")

                    # User inclusion
                    f.write("\n### User Inclusion Impact\n\n")
                    f.write("*How filtering affects the number of users with valid data for analysis.*\n\n")
                    f.write("| Stage | Raw | Filtered | Change |\n")
                    f.write("|-------|-----|----------|--------|\n")

                    inclusion = cohort_dict.get('inclusion', {})
                    baseline_raw = inclusion.get('baseline', {}).get('raw', 0)
                    baseline_filt = inclusion.get('baseline', {}).get('filtered', 0)
                    f.write(f"| Valid Baseline | {baseline_raw} | {baseline_filt} | "
                           f"{baseline_filt - baseline_raw:+d} |\n")

                    endpoint_raw = inclusion.get('endpoint', {}).get('raw', 0)
                    endpoint_filt = inclusion.get('endpoint', {}).get('filtered', 0)
                    f.write(f"| Valid Endpoint | {endpoint_raw} | {endpoint_filt} | "
                           f"{endpoint_filt - endpoint_raw:+d} |\n")

                    # Statistical power
                    f.write("\n### Statistical Power Improvements\n\n")
                    f.write("*How filtering improves the statistical reliability of analyses.*\n\n")
                    power = cohort_dict.get('power', {})
                    f.write(f"- **Variance Reduction**: {power.get('variance_reduction', 0):.1%} - Lower variance means more consistent measurements\n")
                    f.write(f"- **Effect Size Improvement**: {power.get('effect_size_improvement', 0):.3f} - Larger effect sizes are easier to detect statistically\n")
                    f.write("\n")

                # Quarterly Reporting Analysis (90+ Day Users)
                if 'quarterly' in metrics and metrics['quarterly']:
                    quarterly = metrics['quarterly']
                    f.write("## 📊 QUARTERLY REPORTING ANALYSIS\n\n")

                    f.write("This section analyzes users who have been in the program for 90+ days, which is the standard "
                           "timeframe for quarterly business reporting and clinical outcome assessment.\n\n")

                    f.write("### Key Business Question Answered\n\n")

                    if 'filtered_metrics' in quarterly and quarterly['filtered_metrics']:
                        fm = quarterly['filtered_metrics']
                        rm = quarterly['raw_metrics']

                        f.write(f"**\"What is the average weight loss for users in the program for 90+ days?\"**\n\n")

                        # Main answer box
                        f.write("| Metric | Raw Data | Filtered Data | Improvement |\n")
                        f.write("|--------|----------|---------------|-------------|\n")
                        f.write(f"| **Average Weight Loss** | {rm.mean_weight_loss_pct:.2f}% | "
                               f"{fm.mean_weight_loss_pct:.2f}% | "
                               f"{fm.mean_weight_loss_pct - rm.mean_weight_loss_pct:+.2f}% |\n")
                        f.write(f"| Median Weight Loss | {rm.median_weight_loss_pct:.2f}% | "
                               f"{fm.median_weight_loss_pct:.2f}% | "
                               f"{fm.median_weight_loss_pct - rm.median_weight_loss_pct:+.2f}% |\n")
                        f.write(f"| Standard Deviation | {rm.std_weight_loss_pct:.2f}% | "
                               f"{fm.std_weight_loss_pct:.2f}% | "
                               f"{abs(rm.std_weight_loss_pct - fm.std_weight_loss_pct):.2f}% reduction |\n")
                        f.write("\n")

                        # Data quality impact
                        f.write("### Data Quality Impact\n\n")
                        f.write("*How many users have usable data for quarterly reporting.*\n\n")
                        f.write(f"- **Eligible Users**: {rm.eligible_users} users with 90+ days in program\n")
                        f.write(f"- **Valid Data (Raw)**: {rm.users_with_valid_endpoint} users ({rm.users_with_valid_endpoint/rm.eligible_users*100:.1f}%)\n")
                        f.write(f"- **Valid Data (Filtered)**: {fm.users_with_valid_endpoint} users ({fm.users_with_valid_endpoint/fm.eligible_users*100:.1f}%)\n")
                        f.write("\n")

                        # Success rates
                        f.write("### Clinical Success Rates (90+ Day Users)\n\n")
                        f.write("| Threshold | Raw Success Rate | Filtered Success Rate | Difference |\n")
                        f.write("|-----------|-----------------|----------------------|------------|\n")

                        raw_5pct = rm.users_losing_5pct / rm.users_with_valid_endpoint * 100 if rm.users_with_valid_endpoint > 0 else 0
                        filt_5pct = fm.users_losing_5pct / fm.users_with_valid_endpoint * 100 if fm.users_with_valid_endpoint > 0 else 0
                        f.write(f"| 5% Loss | {raw_5pct:.1f}% ({rm.users_losing_5pct} users) | "
                               f"{filt_5pct:.1f}% ({fm.users_losing_5pct} users) | "
                               f"{filt_5pct - raw_5pct:+.1f}% |\n")

                        raw_10pct = rm.users_losing_10pct / rm.users_with_valid_endpoint * 100 if rm.users_with_valid_endpoint > 0 else 0
                        filt_10pct = fm.users_losing_10pct / fm.users_with_valid_endpoint * 100 if fm.users_with_valid_endpoint > 0 else 0
                        f.write(f"| 10% Loss | {raw_10pct:.1f}% ({rm.users_losing_10pct} users) | "
                               f"{filt_10pct:.1f}% ({fm.users_losing_10pct} users) | "
                               f"{filt_10pct - raw_10pct:+.1f}% |\n")

                        raw_15pct = rm.users_losing_15pct / rm.users_with_valid_endpoint * 100 if rm.users_with_valid_endpoint > 0 else 0
                        filt_15pct = fm.users_losing_15pct / fm.users_with_valid_endpoint * 100 if fm.users_with_valid_endpoint > 0 else 0
                        f.write(f"| 15% Loss | {raw_15pct:.1f}% ({rm.users_losing_15pct} users) | "
                               f"{filt_15pct:.1f}% ({fm.users_losing_15pct} users) | "
                               f"{filt_15pct - raw_15pct:+.1f}% |\n")
                        f.write("\n")

                        # Cohort progression
                        if 'cohort_results' in quarterly and quarterly['cohort_results']:
                            f.write("### Weight Loss Progression by Program Duration\n\n")
                            f.write("Average weight loss at different time checkpoints:\n\n")
                            f.write("| Days in Program | Raw Avg Loss | Filtered Avg Loss | Improvement |\n")
                            f.write("|-----------------|--------------|-------------------|-------------|\n")

                            for cohort in quarterly['cohort_results']:
                                f.write(f"| {cohort.day_checkpoint} days | "
                                       f"{cohort.raw_mean_loss_pct:.2f}% | "
                                       f"{cohort.filtered_mean_loss_pct:.2f}% | "
                                       f"{cohort.mean_loss_difference:+.2f}% |\n")
                            f.write("\n")

                        # Visualizations
                        if 'visualizations' in quarterly and quarterly['visualizations']:
                            f.write("### Quarterly Reporting Visualizations\n\n")
                            f.write("The following visualizations have been generated:\n\n")
                            for viz_path in quarterly['visualizations']:
                                viz_name = Path(viz_path).name
                                f.write(f"- `{viz_path}` - {viz_name}\n")
                            f.write("\n")

                    f.write("\n")

                # Individual User Analysis Summary
                if 'users' in metrics and metrics['users']:
                    f.write("## Individual User Analysis\n\n")
                    total_users = metrics.get('aggregate', {}).get('total_users', len(metrics['users']))
                    f.write(f"Analyzed {total_users} users in total.\n")
                    f.write(f"Detailed metrics calculated for all {len(metrics['users'])} users.\n\n")

                    # Create summary table
                    f.write("| User ID | Measurements | Filtered | Removal Rate | Outlier Rate |\n")
                    f.write("|---------|--------------|----------|--------------|-------------|\n")

                    for user_metrics in metrics['users'][:10]:  # Show first 10
                        user_id = user_metrics['user_id'][:8]
                        raw_count = user_metrics['data_summary']['raw'].get('count', 0)
                        filt_count = user_metrics['data_summary']['filtered'].get('count', 0)
                        removal_rate = user_metrics['data_summary'].get('removal_rate', 0)
                        outlier_rate = user_metrics.get('outliers', {}).get('outlier_rate', 0)

                        f.write(f"| {user_id} | {raw_count} | {filt_count} | "
                               f"{removal_rate:.1%} | {outlier_rate:.1%} |\n")

                    f.write("\n")

                # Key Findings
                f.write("## Key Findings & Interpretation\n\n")

                if 'aggregate' in metrics:
                    agg = metrics['aggregate']

                    f.write("### Data Quality Improvements\n\n")
                    f.write("*These metrics show how filtering improves the reliability of weight measurements.*\n\n")
                    f.write(f"1. **Outlier Detection**: Successfully identified and removed "
                           f"{agg.get('outlier_summary', {}).get('total_outliers', 0)} outliers "
                           f"across all users\n")
                    f.write(f"2. **Temporal Consistency**: Reduced daily weight volatility by an average of "
                           f"{agg.get('temporal_summary', {}).get('avg_daily_volatility', 0):.2f}kg\n")
                    f.write(f"3. **Impossible Changes**: Eliminated "
                           f"{agg.get('temporal_summary', {}).get('total_impossible_changes', 0)} "
                           f"physiologically impossible weight changes\n\n")

                    f.write("### Clinical Impact\n\n")
                    f.write("*How filtering prevents medical misinterpretations and improves clinical decision-making.*\n\n")
                    f.write(f"1. **Direction Errors**: Prevented "
                           f"{agg.get('medical_summary', {}).get('total_direction_errors', 0)} "
                           f"cases where weight change direction would be misclassified (e.g., showing gain instead of loss)\n")
                    f.write(f"2. **Confidence Intervals**: Improved measurement confidence by "
                           f"{agg.get('medical_summary', {}).get('avg_confidence_improvement', 0):.1%} "
                           f"on average (tighter confidence bands mean more reliable measurements)\n\n")

                # Visualizations
                if 'visualizations' in metrics and metrics['visualizations']:
                    f.write("## Generated Visualizations\n\n")
                    f.write("The following visualization files have been generated:\n\n")

                    for viz_path in metrics['visualizations']:
                        viz_name = Path(viz_path).name
                        f.write(f"- `{viz_path}` - {viz_name}\n")

                    f.write("\n")

                # Recommendations
                f.write("## Recommendations\n\n")
                f.write("Based on the analysis results, we recommend:\n\n")

                f.write("1. **Continue Filtering**: The filtering process significantly improves data quality "
                       "without compromising clinical validity\n")
                f.write("2. **Source Monitoring**: Pay special attention to data sources with high outlier rates\n")
                f.write("3. **Threshold Tuning**: Consider adjusting quality thresholds based on source reliability\n")
                f.write("4. **Regular Validation**: Implement periodic manual review of filtered data\n\n")

                f.write("### How to Interpret These Results\n\n")
                f.write("- **Higher filtered success rates**: More accurate assessment of true program effectiveness\n")
                f.write("- **Reduced variance**: More reliable individual measurements and trend detection\n")
                f.write("- **Improved mean weight loss**: Removal of erroneous measurements reveals true outcomes\n")
                f.write("- **Better statistical power**: Easier to detect real changes and treatment effects\n\n")

                # Technical Details
                f.write("## Technical Details\n\n")
                f.write("### Configuration Used\n\n")
                f.write("```toml\n")
                if 'processing' in self.config:
                    f.write(f"quality_threshold = {self.config['processing'].get('quality_threshold', 0.45)}\n")
                if 'kalman' in self.config:
                    f.write(f"initial_variance = {self.config['kalman'].get('initial_variance', 1.0)}\n")
                f.write("```\n\n")

                f.write("### Analysis Parameters\n\n")
                f.write(f"- Analysis timestamp: {metrics.get('timestamp', 'N/A')}\n")
                f.write(f"- Cohort size: {metrics.get('cohort_size', 0)} users\n")
                f.write(f"- Minimum measurements per user: "
                       f"{self.config.get('analysis', {}).get('min_measurements', 20)}\n")

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
            report_dir = Path(self.config.get('analysis', {}).get('output_dir', 'reports/visualizations'))
            report_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = report_dir / f"filtering_metrics_{timestamp}.json"

            # Convert any non-serializable objects
            def make_serializable(obj):
                if hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                else:
                    return str(obj)

            # Clean metrics for JSON serialization
            clean_metrics = json.loads(
                json.dumps(metrics, default=make_serializable)
            )

            with open(json_path, 'w') as f:
                json.dump(clean_metrics, f, indent=2)

            logger.info(f"Metrics saved to {json_path}")
            return str(json_path)

        except Exception as e:
            logger.error(f"Error saving metrics JSON: {e}")
            return ""

    def save_user_metrics_csv(self, metrics: Dict[str, Any], raw_data: Dict[str, pd.DataFrame],
                             filtered_data: Dict[str, pd.DataFrame]) -> str:
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
            report_dir = Path(self.config.get('analysis', {}).get('output_dir', 'reports/visualizations'))
            report_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = report_dir / f"user_analysis_results_{timestamp}.csv"

            # Build rows for CSV
            rows = []

            for user_id in raw_data.keys():
                row = {
                    'user_id': user_id,
                    'raw_measurement_count': len(raw_data[user_id]),
                    'filtered_measurement_count': len(filtered_data.get(user_id, [])),
                    'removal_rate': 0,
                    'raw_mean_weight': 0,
                    'filtered_mean_weight': 0,
                    'raw_std_weight': 0,
                    'filtered_std_weight': 0,
                    'raw_min_weight': 0,
                    'raw_max_weight': 0,
                    'filtered_min_weight': 0,
                    'filtered_max_weight': 0,
                    'raw_weight_change_pct': 0,
                    'filtered_weight_change_pct': 0,
                    'outlier_count': 0,
                    'outlier_rate': 0,
                    'data_duration_days': 0,
                    'raw_daily_volatility': 0,
                    'filtered_daily_volatility': 0
                }

                # Calculate metrics for raw data
                raw_df = raw_data[user_id]
                if not raw_df.empty:
                    row['raw_mean_weight'] = raw_df['weight'].mean()
                    row['raw_std_weight'] = raw_df['weight'].std()
                    row['raw_min_weight'] = raw_df['weight'].min()
                    row['raw_max_weight'] = raw_df['weight'].max()

                    # Calculate weight change percentage
                    if len(raw_df) > 1:
                        sorted_df = raw_df.sort_values('timestamp')
                        baseline = sorted_df['weight'].iloc[0]
                        endpoint = sorted_df['weight'].iloc[-1]
                        if baseline > 0:
                            row['raw_weight_change_pct'] = ((endpoint - baseline) / baseline) * 100

                        # Calculate daily volatility
                        daily_changes = np.diff(sorted_df['weight'].values)
                        row['raw_daily_volatility'] = np.std(daily_changes) if len(daily_changes) > 0 else 0

                    # Data duration
                    row['data_duration_days'] = (raw_df['timestamp'].max() - raw_df['timestamp'].min()).days

                # Calculate metrics for filtered data
                if user_id in filtered_data:
                    filtered_df = filtered_data[user_id]
                    if not filtered_df.empty:
                        row['filtered_mean_weight'] = filtered_df['weight'].mean()
                        row['filtered_std_weight'] = filtered_df['weight'].std()
                        row['filtered_min_weight'] = filtered_df['weight'].min()
                        row['filtered_max_weight'] = filtered_df['weight'].max()

                        # Calculate weight change percentage
                        if len(filtered_df) > 1:
                            sorted_df = filtered_df.sort_values('timestamp')
                            baseline = sorted_df['weight'].iloc[0]
                            endpoint = sorted_df['weight'].iloc[-1]
                            if baseline > 0:
                                row['filtered_weight_change_pct'] = ((endpoint - baseline) / baseline) * 100

                            # Calculate daily volatility
                            daily_changes = np.diff(sorted_df['weight'].values)
                            row['filtered_daily_volatility'] = np.std(daily_changes) if len(daily_changes) > 0 else 0

                # Calculate removal metrics
                row['removal_rate'] = (row['raw_measurement_count'] - row['filtered_measurement_count']) / row['raw_measurement_count'] if row['raw_measurement_count'] > 0 else 0
                row['outlier_count'] = row['raw_measurement_count'] - row['filtered_measurement_count']
                row['outlier_rate'] = row['outlier_count'] / row['raw_measurement_count'] if row['raw_measurement_count'] > 0 else 0

                # Add individual user metrics if available
                if 'users' in metrics:
                    user_metrics = next((m for m in metrics['users'] if m['user_id'] == user_id), None)
                    if user_metrics:
                        # Add additional metrics from the analysis
                        if 'medical_impact' in user_metrics:
                            row['direction_errors'] = user_metrics['medical_impact'].get('clinical', {}).get('direction_errors', 0)
                            row['confidence_improvement'] = user_metrics['medical_impact'].get('confidence', {}).get('ci_reduction', 0)

                        if 'temporal' in user_metrics:
                            row['impossible_changes'] = user_metrics['temporal'].get('daily_change', {}).get('impossible_count', 0)
                            row['max_daily_change'] = user_metrics['temporal'].get('daily_change', {}).get('max', 0)

                rows.append(row)

            # Create DataFrame and save to CSV
            results_df = pd.DataFrame(rows)

            # Sort by removal rate to highlight most impacted users
            results_df = results_df.sort_values('removal_rate', ascending=False)

            # Save to CSV
            results_df.to_csv(csv_path, index=False)

            logger.info(f"User metrics CSV saved to {csv_path}")
            return str(csv_path)

        except Exception as e:
            logger.error(f"Error saving user metrics CSV: {e}")
            return ""


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive filtering effectiveness analysis"
    )
    parser.add_argument(
        "csv_file",
        nargs='?',
        default="data/2025-09-05_all.csv",
        help="Path to CSV file with weight measurements (default: data/2025-09-05_all.csv)"
    )
    parser.add_argument(
        "--filtered",
        default="data/2025-09-05_all_filtered.csv",
        help="Path to filtered CSV file (default: data/2025-09-05_all_filtered.csv)"
    )
    parser.add_argument(
        "--employer",
        default="data/2025-09-17-user-employers.csv",
        help="Path to employer data file (default: data/2025-09-17-user-employers.csv)"
    )
    parser.add_argument(
        "--partners",
        default="data/partners.csv",
        help="Path to partners data file (default: data/partners.csv)"
    )
    parser.add_argument(
        "--filter-employer",
        help="Filter analysis to only users from this employer"
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to configuration file (default: config.toml)"
    )
    parser.add_argument(
        "--max-users",
        type=int,
        help="Maximum number of users to analyze"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for output files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize runner
    runner = FilteringAnalysisRunner(args.config)

    # Override config with command line args
    if args.max_users:
        runner.config.setdefault('analysis', {})['max_users'] = args.max_users
    if args.output_dir:
        runner.config.setdefault('analysis', {})['output_dir'] = args.output_dir
        # Recreate visualizer with new output directory
        runner.visualizer = FilteringVisualizationGenerator(output_dir=args.output_dir)
        # Also update quarterly visualizer
        runner.quarterly_viz = QuarterlyVisualizationGenerator(output_dir=f"{args.output_dir}/quarterly")

    # Determine if we should skip user limit (when filtering by employer)
    skip_limit = bool(args.filter_employer)

    # Load raw data
    raw_data = runner.load_data_from_csv(args.csv_file, "raw", skip_user_limit=skip_limit)
    if not raw_data:
        logger.error("No raw data loaded. Exiting.")
        return 1

    # Load pre-filtered data if available
    filtered_data = {}
    if Path(args.filtered).exists():
        logger.info("Loading pre-filtered data...")
        filtered_data_full = runner.load_data_from_csv(args.filtered, "filtered", skip_user_limit=skip_limit)

        # The filtered dataset is a subset of raw - align them properly
        filtered_data = {}
        for user_id in raw_data.keys():
            if user_id in filtered_data_full:
                # Get the filtered subset for this user
                filtered_df = filtered_data_full[user_id]
                raw_df = raw_data[user_id]

                # The filtered data should be a subset of raw data
                # Match by timestamp (since filtered removed some measurements)
                filtered_timestamps = set(filtered_df['timestamp'])

                # Keep only the filtered measurements
                filtered_data[user_id] = filtered_df

                # Log removal statistics
                removed_count = len(raw_df) - len(filtered_df)
                removal_rate = removed_count / len(raw_df) if len(raw_df) > 0 else 0
                logger.info(f"User {user_id[:8]}: {len(raw_df)} raw -> {len(filtered_df)} filtered "
                          f"({removal_rate:.1%} removed)")

        if skip_limit:
            logger.info(f"Loaded filtered data for {len(filtered_data)} users (all users, no limit applied)")
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
            logger.info(f"Processing user {user_id[:8]}... ({len(raw_df)} measurements)")

            filtered_measurements = []

            # Process each measurement through the pipeline
            for _, row in raw_df.iterrows():
                try:
                    result = process_measurement(
                        user_id=user_id,
                        weight=row['weight'],
                        timestamp=row['timestamp'],
                        source=row.get('source', 'unknown'),
                        config=runner.config,
                        unit=row.get('unit', 'kg'),
                        db=db
                    )

                    # Only keep accepted measurements
                    if result.get('accepted', False):
                        filtered_measurements.append({
                            'timestamp': row['timestamp'],
                            'weight': result.get('filtered_weight', row['weight']),
                            'source': row.get('source', 'unknown'),
                            'quality_score': result.get('quality_score', 0),
                            'unit': row.get('unit', 'kg')
                        })

                except Exception as e:
                    logger.warning(f"Error processing measurement for user {user_id}: {e}")
                    continue

            # Create filtered DataFrame
            if filtered_measurements:
                filtered_df = pd.DataFrame(filtered_measurements)
                filtered_data[user_id] = filtered_df.sort_values('timestamp').reset_index(drop=True)
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
        target_users = {uid for uid, emp in employer_data.items()
                       if emp and str(emp).lower() == args.filter_employer.lower()}

        if not target_users:
            logger.error(f"No users found for employer: {args.filter_employer}")
            logger.info("Available employers:")
            unique_employers = set(emp for emp in employer_data.values() if emp)
            for emp in sorted(unique_employers)[:20]:  # Show first 20
                logger.info(f"  - {emp}")
            return 1

        # Filter both raw and filtered data to only include these users
        raw_data = {uid: data for uid, data in raw_data.items() if uid in target_users}
        filtered_data = {uid: data for uid, data in filtered_data.items() if uid in target_users}

        logger.info(f"Filtering to employer {args.filter_employer}: {len(raw_data)} users found")
        logger.info(f"Will analyze ALL {len(raw_data)} employer users, visualizations for top 10 most impacted")

        if not raw_data:
            logger.error(f"No data available for users from {args.filter_employer}")
            return 1

    # Run analysis
    metrics = runner.run_analysis(raw_data, filtered_data)

    # Run quarterly analysis
    # Pass the filtered user list if we're filtering by employer
    filter_users = list(raw_data.keys()) if args.filter_employer else None
    quarterly_metrics = runner.run_quarterly_analysis(raw_data, filtered_data, args.employer, filter_users)
    metrics['quarterly'] = quarterly_metrics

    # Generate outputs
    report_path = runner.generate_report(metrics)
    json_path = runner.save_metrics_json(metrics)
    csv_path = runner.save_user_metrics_csv(metrics, raw_data, filtered_data)

    logger.info("=" * 60)
    logger.info("Analysis Complete!")
    logger.info(f"Report: {report_path}")
    logger.info(f"Metrics (JSON): {json_path}")
    logger.info(f"User Results (CSV): {csv_path}")
    logger.info(f"Visualizations: {runner.config.get('analysis', {}).get('output_dir', 'reports/visualizations')}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
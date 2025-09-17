#!/usr/bin/env python3
"""
Weight Loss Interval Analysis Report Generator

This script analyzes weight loss progress at 30-day intervals,
comparing raw measurements against Kalman-filtered data to
quantify the impact of outlier rejection.
"""

import argparse
import logging
import sys
import csv
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, Optional, List
import time

from src.config_loader import load_config as load_config_with_profiles
from src.database.db_wrapper import Database
from src.analysis.interval_analyzer_fast import FastIntervalAnalyzer
from src.analysis.interval_analyzer_parallel import ParallelIntervalAnalyzer
from src.analysis.weight_selector import WeightSelector
from src.analysis.statistics import StatisticsCalculator
from src.analysis.csv_generator import CSVGenerator
from src.analysis.analysis_visualizer import AnalysisVisualizer
from src.analysis.markdown_reporter import MarkdownReporter
from src.analysis.daily_weight_analyzer import DailyWeightAnalyzer




class WeightLossReport:
    """Main orchestrator for weight loss analysis reporting"""

    def __init__(self, config_path: str, output_dir: str):
        """Initialize report generator with configuration"""
        self.config = load_config_with_profiles(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.data_dir = self.output_dir / "data"
        self.viz_dir = self.output_dir / "visualizations"
        self.data_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)

        # Note: Database is optional now, only used if needed
        # db_path = self.config.get("database", {}).get("path", "data/weight_tracking.db")
        # self.db = Database(db_path)

        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("WeightLossReport")
        logger.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        log_file = self.output_dir / "report_generation.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def generate_report(self,
                       raw_csv: str,
                       filtered_csv: str,
                       top_n: int = 200,
                       interval_days: int = 30,
                       window_days: int = 7,
                       use_parallel: bool = False) -> Dict:
        """
        Generate comprehensive weight loss analysis report

        Args:
            raw_csv: Path to raw weight measurements CSV
            filtered_csv: Path to filtered weight measurements CSV (from main.py)
            top_n: Number of top divergent users to visualize
            interval_days: Interval between measurements (default: 30)
            window_days: ±window for selecting measurements (default: 7)
            use_parallel: Use parallel processing for interval calculation (default: False)

        Returns:
            Dictionary with report metadata and file paths
        """
        self.logger.info(f"Starting report generation")
        self.logger.info(f"Raw data: {raw_csv}")
        self.logger.info(f"Filtered data: {filtered_csv}")
        self.logger.info(f"Parameters: top_n={top_n}, interval={interval_days}d, window=±{window_days}d")

        # Track phase timings for performance report
        phase_timings = {}

        try:
            # Phase 1: Load raw data
            phase1_start = time.time()
            self.logger.info("Phase 1: Loading raw data...")
            raw_data = self._load_raw_data(raw_csv)
            phase1_time = time.time() - phase1_start
            phase_timings['phase1'] = phase1_time
            self.logger.info(f"Loaded {len(raw_data)} raw measurements in {phase1_time:.2f}s")

            # Phase 2: Load filtered data
            phase2_start = time.time()
            self.logger.info("Phase 2: Loading filtered data...")
            filtered_data = self._load_raw_data(filtered_csv)  # Same format as raw
            phase2_time = time.time() - phase2_start
            phase_timings['phase2'] = phase2_time
            self.logger.info(f"Loaded {len(filtered_data)} filtered measurements in {phase2_time:.2f}s")

            # Phase 3: Calculate intervals for each user (using optimized or parallel version)
            phase3_start = time.time()
            if use_parallel:
                self.logger.info("Phase 3: Calculating intervals (parallel)...")
                interval_analyzer = ParallelIntervalAnalyzer(
                    interval_days, window_days
                )
            else:
                self.logger.info("Phase 3: Calculating intervals (optimized)...")
                interval_analyzer = FastIntervalAnalyzer(
                    interval_days, window_days
                )
            user_intervals = interval_analyzer.calculate_all_users(
                raw_data, filtered_data
            )
            phase3_time = time.time() - phase3_start
            phase_timings['phase3'] = phase3_time
            self.logger.info(f"Calculated intervals for {len(user_intervals)} users in {phase3_time:.2f}s")

            # Phase 4: Generate statistics
            phase4_start = time.time()
            self.logger.info("Phase 4: Generating statistics...")
            stats_calc = StatisticsCalculator()
            statistics = stats_calc.calculate_all_statistics(user_intervals)
            phase4_time = time.time() - phase4_start
            phase_timings['phase4'] = phase4_time
            self.logger.info(f"Generated statistics in {phase4_time:.2f}s")

            # Phase 5: Identify top divergent users
            phase5_start = time.time()
            self.logger.info("Phase 5: Identifying top divergent users...")
            top_users = self._identify_top_divergent_users(
                user_intervals, statistics, top_n
            )
            phase5_time = time.time() - phase5_start
            phase_timings['phase5'] = phase5_time
            self.logger.info(f"Identified top {len(top_users)} divergent users in {phase5_time:.2f}s")

            # Phase 6: Generate CSV outputs
            phase6_start = time.time()
            self.logger.info("Phase 6: Generating CSV files...")
            csv_gen = CSVGenerator(self.data_dir)
            csv_files = csv_gen.generate_all_csvs(
                raw_data, filtered_data, user_intervals,
                statistics, top_users
            )
            phase6_time = time.time() - phase6_start
            phase_timings['phase6'] = phase6_time
            self.logger.info(f"Generated {len(csv_files)} CSV files in {phase6_time:.2f}s")

            # Phase 7: Daily weight analysis
            phase7_start = time.time()
            self.logger.info("Phase 7: Performing daily weight analysis...")
            daily_analyzer = DailyWeightAnalyzer()
            daily_results = daily_analyzer.calculate_daily_weights(raw_data, filtered_data)
            dramatic_users = daily_analyzer.get_most_dramatic_users(daily_results, top_n=20)
            daily_stats = daily_analyzer.generate_daily_statistics(daily_results)
            phase7_time = time.time() - phase7_start
            phase_timings['phase7'] = phase7_time
            self.logger.info(f"Daily analysis completed in {phase7_time:.2f}s")
            self.logger.info(f"Found {len(dramatic_users)} users with dramatic daily differences")
            if dramatic_users:
                top_dramatic = dramatic_users[0]
                self.logger.info(f"Most dramatic: Score={top_dramatic['dramatic_score']:.1f}, "
                               f"Max diff={top_dramatic['max_daily_difference']:.2f} lbs")

            # Phase 8: Generate visualizations (DISABLED)
            phase8_start = time.time()
            self.logger.info("Phase 8: Skipping visualizations (disabled)...")
            viz_files = {}  # Empty dictionary since visualizations are disabled

            # Commented out visualization generation
            # viz = AnalysisVisualizer(self.viz_dir)
            # viz_files = viz.generate_all_visualizations(
            #     user_intervals, statistics, top_users
            # )
            # if daily_results and dramatic_users:
            #     daily_viz_files = viz.generate_daily_comparison_visualizations(
            #         daily_results, dramatic_users
            #     )
            #     viz_files.update(daily_viz_files)

            phase8_time = time.time() - phase8_start
            phase_timings['phase8'] = phase8_time
            self.logger.info(f"Visualization generation disabled")

            # Phase 9: Generate summary report
            phase9_start = time.time()
            self.logger.info("Phase 9: Generating summary report...")
            summary = self._generate_summary_report(
                statistics, csv_files, viz_files, daily_stats
            )
            phase9_time = time.time() - phase9_start
            phase_timings['phase9'] = phase9_time

            # Phase 10: Generate comprehensive Markdown report
            phase10_start = time.time()
            self.logger.info("Phase 10: Generating comprehensive Markdown report...")
            md_reporter = MarkdownReporter(self.output_dir)
            # Add daily analysis to statistics for report
            statistics['daily_analysis'] = daily_stats
            statistics['dramatic_users'] = dramatic_users

            md_report_path = md_reporter.generate_comprehensive_report(
                raw_data, filtered_data, user_intervals,
                statistics, top_users, csv_files, phase_timings
            )
            phase10_time = time.time() - phase10_start
            phase_timings['phase10'] = phase10_time
            self.logger.info(f"Generated Markdown report: {md_report_path} in {phase10_time:.2f}s")

            total_time = time.time() - phase1_start
            self.logger.info(f"Report generation complete! Total time: {total_time:.2f}s")
            self.logger.info(f"Phase breakdown: "
                           f"P1={phase1_time:.2f}s, P2={phase2_time:.2f}s, "
                           f"P3={phase3_time:.2f}s, P4={phase4_time:.2f}s, "
                           f"P5={phase5_time:.2f}s, P6={phase6_time:.2f}s, "
                           f"P7={phase7_time:.2f}s, P8={phase8_time:.2f}s, "
                           f"P9={phase9_time:.2f}s, P10={phase10_time:.2f}s")

            return {
                'status': 'success',
                'csv_files': csv_files,
                'visualizations': viz_files,
                'summary': summary,
                'markdown_report': md_report_path,
                'output_directory': str(self.output_dir)
            }

        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'output_directory': str(self.output_dir)
            }

    def _load_raw_data(self, csv_path: str) -> pd.DataFrame:
        """Load raw measurement data from CSV"""
        df = pd.read_csv(csv_path)

        # Handle different column name variations
        # Map common variations to standard names
        column_mappings = {
            'source': ['source', 'source_type'],
            'timestamp': ['timestamp', 'effectiveDateTime', 'datetime', 'date']
        }

        for standard_name, variations in column_mappings.items():
            if standard_name not in df.columns:
                for variant in variations:
                    if variant in df.columns:
                        df[standard_name] = df[variant]
                        break

        # Ensure required columns exist
        required_cols = ['user_id', 'weight', 'timestamp', 'source']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            self.logger.error(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert timestamp to datetime with error handling for mixed formats
        # First fix common date errors (year 0025 should be 2025)
        df['timestamp'] = df['timestamp'].str.replace(r'^0025-', '2025-', regex=True)

        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
        except:
            try:
                # Fall back to mixed format parsing
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=False)
            except:
                # If still failing, parse with errors='coerce' to handle bad dates
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                # Drop rows with unparseable dates
                bad_dates = df['timestamp'].isna().sum()
                if bad_dates > 0:
                    self.logger.warning(f"Dropped {bad_dates} rows with unparseable dates")
                    df = df.dropna(subset=['timestamp'])

        # Sort by user and timestamp
        df = df.sort_values(['user_id', 'timestamp'])

        return df


    def _identify_top_divergent_users(self,
                                     user_intervals: Dict,
                                     statistics: Dict,
                                     top_n: int) -> list:
        """Identify users with highest divergence between raw and filtered"""
        divergence_scores = []

        for user_id, intervals in user_intervals.items():
            # Calculate divergence score with more detailed metrics
            total_diff = 0
            squared_diff = 0  # For RMS calculation
            max_diff = 0
            valid_intervals = 0
            raw_weights = []
            filtered_weights = []

            for interval_data in intervals['intervals']:
                if interval_data['raw_weight'] and interval_data['filtered_weight']:
                    raw_w = interval_data['raw_weight']
                    filt_w = interval_data['filtered_weight']
                    diff = abs(raw_w - filt_w)

                    total_diff += diff
                    squared_diff += diff ** 2
                    max_diff = max(max_diff, diff)
                    valid_intervals += 1
                    raw_weights.append(raw_w)
                    filtered_weights.append(filt_w)

            if valid_intervals >= 3:  # Need at least 3 intervals for meaningful comparison
                avg_diff = total_diff / valid_intervals
                rms_diff = (squared_diff / valid_intervals) ** 0.5

                # Calculate trajectory difference (slope difference)
                if len(raw_weights) >= 2:
                    raw_trend = (raw_weights[-1] - raw_weights[0]) / len(raw_weights)
                    filt_trend = (filtered_weights[-1] - filtered_weights[0]) / len(filtered_weights)
                    trend_diff = abs(raw_trend - filt_trend)
                else:
                    trend_diff = 0

                # Composite score: weighted average of different metrics
                composite_score = (avg_diff * 0.5) + (rms_diff * 0.3) + (max_diff * 0.2)

                divergence_scores.append({
                    'user_id': user_id,
                    'divergence_score': composite_score,  # Main sorting criterion
                    'avg_diff': avg_diff,
                    'rms_diff': rms_diff,
                    'max_diff': max_diff,
                    'total_diff': total_diff,
                    'trend_diff': trend_diff,
                    'num_intervals': valid_intervals
                })

        # Sort by composite divergence score and return top N
        divergence_scores.sort(key=lambda x: x['divergence_score'], reverse=True)

        self.logger.info(f"Top divergent users analysis: {len(divergence_scores)} users with 3+ intervals")
        if divergence_scores:
            top_user = divergence_scores[0]
            self.logger.info(f"Highest divergence: User {top_user['user_id'][:8]}... "
                           f"Score={top_user['divergence_score']:.2f}, "
                           f"Avg={top_user['avg_diff']:.2f}, "
                           f"Max={top_user['max_diff']:.2f}")

        return divergence_scores[:top_n]

    def _generate_summary_report(self,
                                statistics: Dict,
                                csv_files: Dict,
                                viz_files: Dict,
                                daily_stats: Dict = None) -> Dict:
        """Generate summary report with key findings"""
        summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_users_analyzed': statistics.get('total_users', 0),
            'intervals_analyzed': statistics.get('intervals_analyzed', []),
            'average_weight_loss': {
                'raw': statistics.get('avg_loss_raw', {}),
                'filtered': statistics.get('avg_loss_filtered', {})
            },
            'outlier_impact': {
                'avg_measurements_rejected': statistics.get('avg_outliers_per_user', 0),
                'max_single_user_impact': statistics.get('max_impact', 0)
            },
            'daily_analysis': daily_stats if daily_stats else {},
            'files_generated': {
                'csv_files': list(csv_files.keys()),
                'visualization_files': list(viz_files.keys())
            }
        }

        # Save summary as JSON
        import json
        summary_path = self.output_dir / "report_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        return summary



def main():
    """Main entry point for report generation"""
    parser = argparse.ArgumentParser(
        description="Generate weight loss interval analysis report"
    )

    parser.add_argument(
        'raw_csv',
        help='Path to raw CSV file with weight measurements'
    )
    parser.add_argument(
        'filtered_csv',
        help='Path to filtered CSV file (from main.py --filtered-output)'
    )
    parser.add_argument(
        '--config',
        default='config.toml',
        help='Path to configuration file (default: config.toml)'
    )
    parser.add_argument(
        '--output-dir',
        default='reports_analysis',
        help='Output directory for reports (default: reports)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=200,
        help='Number of top divergent users to visualize (default: 200)'
    )
    parser.add_argument(
        '--interval-days',
        type=int,
        default=5,
        help='Interval between measurements in days (default: 5)'
    )
    parser.add_argument(
        '--window-days',
        type=float,
        default=2.5,
        help='Window tolerance for measurements in days (default: ±2.5)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Use parallel processing for interval calculation (uses multiple CPU cores)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate input files exist
    if not Path(args.raw_csv).exists():
        print(f"Error: Raw CSV file '{args.raw_csv}' not found")
        sys.exit(1)

    if not Path(args.filtered_csv).exists():
        print(f"Error: Filtered CSV file '{args.filtered_csv}' not found")
        print("Please generate it first using: python main.py --filtered-output <path>")
        sys.exit(1)

    # Generate report
    report = WeightLossReport(args.config, args.output_dir)
    result = report.generate_report(
        raw_csv=args.raw_csv,
        filtered_csv=args.filtered_csv,
        top_n=args.top_n,
        interval_days=args.interval_days,
        window_days=args.window_days,
        use_parallel=args.parallel
    )

    if result['status'] == 'success':
        print(f"\n✓ Report generated successfully!")
        print(f"  Output directory: {result['output_directory']}")
        print(f"  Main report: {Path(result['markdown_report']).name}")
        print(f"  CSV files: {len(result['csv_files'])}")
        print(f"  Visualizations: {len(result['visualizations'])}")
        print(f"\n📊 Open the Markdown report for detailed insights and analysis!")
    else:
        print(f"\n✗ Report generation failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()

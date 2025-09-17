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

from src.config_loader import load_config as load_config_with_profiles
from src.database.db_wrapper import Database
from src.analysis.interval_analyzer import IntervalAnalyzer
from src.analysis.weight_selector import WeightSelector
from src.analysis.statistics import StatisticsCalculator
from src.analysis.csv_generator import CSVGenerator
from src.analysis.analysis_visualizer import AnalysisVisualizer




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

        # Initialize database connection
        db_path = self.config.get("database", {}).get("path", "data/weight_tracking.db")
        self.db = Database(db_path)

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
                       window_days: int = 7) -> Dict:
        """
        Generate comprehensive weight loss analysis report

        Args:
            raw_csv: Path to raw weight measurements CSV
            filtered_csv: Path to filtered weight measurements CSV (from main.py)
            top_n: Number of top divergent users to visualize
            interval_days: Interval between measurements (default: 30)
            window_days: ±window for selecting measurements (default: 7)

        Returns:
            Dictionary with report metadata and file paths
        """
        self.logger.info(f"Starting report generation")
        self.logger.info(f"Raw data: {raw_csv}")
        self.logger.info(f"Filtered data: {filtered_csv}")
        self.logger.info(f"Parameters: top_n={top_n}, interval={interval_days}d, window=±{window_days}d")

        try:
            # Phase 1: Load raw data
            self.logger.info("Phase 1: Loading raw data...")
            raw_data = self._load_raw_data(raw_csv)
            self.logger.info(f"Loaded {len(raw_data)} raw measurements")

            # Phase 2: Load filtered data
            self.logger.info("Phase 2: Loading filtered data...")
            filtered_data = self._load_raw_data(filtered_csv)  # Same format as raw
            self.logger.info(f"Loaded {len(filtered_data)} filtered measurements")

            # Phase 3: Calculate intervals for each user
            self.logger.info("Phase 3: Calculating intervals...")
            interval_analyzer = IntervalAnalyzer(
                self.db, interval_days, window_days
            )
            user_intervals = interval_analyzer.calculate_all_users(
                raw_data, filtered_data
            )
            self.logger.info(f"Calculated intervals for {len(user_intervals)} users")

            # Phase 4: Generate statistics
            self.logger.info("Phase 4: Generating statistics...")
            stats_calc = StatisticsCalculator()
            statistics = stats_calc.calculate_all_statistics(user_intervals)

            # Phase 5: Identify top divergent users
            self.logger.info("Phase 5: Identifying top divergent users...")
            top_users = self._identify_top_divergent_users(
                user_intervals, statistics, top_n
            )
            self.logger.info(f"Identified top {len(top_users)} divergent users")

            # Phase 6: Generate CSV outputs
            self.logger.info("Phase 6: Generating CSV files...")
            csv_gen = CSVGenerator(self.data_dir)
            csv_files = csv_gen.generate_all_csvs(
                raw_data, filtered_data, user_intervals,
                statistics, top_users
            )
            self.logger.info(f"Generated {len(csv_files)} CSV files")

            # Phase 7: Generate visualizations
            self.logger.info("Phase 7: Generating visualizations...")
            viz = AnalysisVisualizer(self.viz_dir)
            viz_files = viz.generate_all_visualizations(
                user_intervals, statistics, top_users
            )
            self.logger.info(f"Generated {len(viz_files)} visualization files")

            # Phase 8: Generate summary report
            self.logger.info("Phase 8: Generating summary report...")
            summary = self._generate_summary_report(
                statistics, csv_files, viz_files
            )

            self.logger.info("Report generation complete!")

            return {
                'status': 'success',
                'csv_files': csv_files,
                'visualizations': viz_files,
                'summary': summary,
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

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by user and timestamp
        df = df.sort_values(['user_id', 'timestamp'])

        return df

    def _get_filtered_data_from_db(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Get filtered (Kalman-processed) data from database"""
        # Get unique users
        users = raw_data['user_id'].unique()

        filtered_records = []
        for user_id in users:
            # Get user's processed measurements from database
            measurements = self.db.get_user_measurements(user_id)

            for m in measurements:
                # Only include measurements that weren't rejected
                if not m.get('is_outlier', False):
                    filtered_records.append({
                        'user_id': user_id,
                        'weight': m.get('filtered_weight', m['weight']),
                        'timestamp': pd.to_datetime(m['timestamp']),
                        'source': m['source'],
                        'quality_score': m.get('quality_score', 0.5),
                        'is_outlier': False
                    })

        if not filtered_records:
            # If no database records, use raw data without outliers
            self.logger.warning("No filtered data in database, using raw data")
            return raw_data.copy()

        filtered_df = pd.DataFrame(filtered_records)
        return filtered_df.sort_values(['user_id', 'timestamp'])

    def _identify_top_divergent_users(self,
                                     user_intervals: Dict,
                                     statistics: Dict,
                                     top_n: int) -> list:
        """Identify users with highest divergence between raw and filtered"""
        divergence_scores = []

        for user_id, intervals in user_intervals.items():
            if user_id not in statistics.get('user_statistics', {}):
                continue

            user_stats = statistics['user_statistics'][user_id]

            # Calculate divergence score
            total_diff = 0
            valid_intervals = 0

            for interval_data in intervals['intervals']:
                if interval_data['raw_weight'] and interval_data['filtered_weight']:
                    diff = abs(interval_data['raw_weight'] - interval_data['filtered_weight'])
                    total_diff += diff
                    valid_intervals += 1

            if valid_intervals > 0:
                avg_diff = total_diff / valid_intervals
                divergence_scores.append({
                    'user_id': user_id,
                    'divergence_score': avg_diff,
                    'total_diff': total_diff,
                    'num_intervals': valid_intervals
                })

        # Sort by divergence score and return top N
        divergence_scores.sort(key=lambda x: x['divergence_score'], reverse=True)
        return divergence_scores[:top_n]

    def _generate_summary_report(self,
                                statistics: Dict,
                                csv_files: Dict,
                                viz_files: Dict) -> Dict:
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
        default='reports',
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
        default=30,
        help='Interval between measurements in days (default: 30)'
    )
    parser.add_argument(
        '--window-days',
        type=int,
        default=7,
        help='Window tolerance for measurements in days (default: ±7)'
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
        window_days=args.window_days
    )

    if result['status'] == 'success':
        print(f"\n✓ Report generated successfully!")
        print(f"  Output directory: {result['output_directory']}")
        print(f"  CSV files: {len(result['csv_files'])}")
        print(f"  Visualizations: {len(result['visualizations'])}")
    else:
        print(f"\n✗ Report generation failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
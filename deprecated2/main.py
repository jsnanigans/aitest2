#!/usr/bin/env python3
"""
Weight Stream Processor v2.1 with Robust Kalman Filter
Enhanced with outlier-resistant state estimation for extreme values.
True streaming with proper layered architecture.
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from src.core import config_loader, get_logger
from src.core.logger_config import setup_logging
from src.core.progress import ProcessingStatus, ProgressIndicator, count_csv_lines
from src.processing import UserProcessor

# Setup enhanced logging
setup_logging()
logger = get_logger(__name__)


class StreamProcessor:
    """Main streaming processor with clean architecture."""

    def __init__(self, config_path: str = "config.toml"):
        self.config = config_loader.load_config(config_path)
        self.users: Dict[str, UserProcessor] = {}
        self.stats = {
            'total_rows': 0,
            'total_users': 0,
            'rows_processed': 0,
            'start_time': None,
            'end_time': None
        }

        # Output configuration - create structured folders
        self.output_dir = Path(self.config.get('output', {}).get('directory', 'output'))
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories for organization
        self.results_dir = self.output_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)

        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)

        # Visualization directory from config
        viz_config_dir = self.config.get('visualization', {}).get('output_dir', 'output/visualizations')
        self.viz_dir = Path(viz_config_dir)
        # Don't create viz_dir here - let visualization module handle it when needed

    def process_file(self, filepath: str):
        """
        Process CSV file line by line.
        True streaming - each user processed as data arrives.
        """
        ProcessingStatus.section("Weight Stream Processing")
        ProcessingStatus.info(f"Processing file: {filepath}")

        # Count lines for progress bar
        ProcessingStatus.processing("Counting rows...")
        total_lines = count_csv_lines(filepath)
        if total_lines:
            ProcessingStatus.success(f"Found {total_lines:,} rows to process")

        # Check max_users configuration
        max_users = self.config.get('processing', {}).get('max_users', 0)
        if max_users > 0:
            ProcessingStatus.info(f"Processing limited to {max_users} users")
            # When max_users is set, use it as the total for progress tracking
            progress = ProgressIndicator(total_items=max_users, show_eta=True)
            progress.start(f"Processing weight data (up to {max_users} users)")
        else:
            # Otherwise use total lines for progress
            progress = ProgressIndicator(total_items=total_lines, show_eta=True)
            progress.start("Processing weight data")

        self.stats['start_time'] = time.time()

        current_user_id = None
        current_processor = None
        max_users = self.config.get('processing', {}).get('max_users', 0)

        with open(filepath) as csvfile:
            reader = csv.DictReader(csvfile)

            for row_num, row in enumerate(reader, 1):
                self.stats['total_rows'] = row_num

                # Update progress - use users as primary metric if max_users is set
                if max_users > 0:
                    progress_value = min(self.stats['total_users'], max_users)
                else:
                    progress_value = row_num
                    
                progress.update(progress_value, {
                    'rows_processed': row_num,
                    'users_processed': self.stats['total_users'],
                    'current_user': current_user_id,
                    'accepted': self.stats.get('total_accepted', 0),
                    'rejected': self.stats.get('total_rejected', 0)
                })

                # Get user ID
                user_id = row.get('user_id')
                if not user_id:
                    continue

                # Filter for test users if configured
                filter_test_users = self.config.get('processing', {}).get('filter_test_users', False)
                test_users = self.config.get('processing', {}).get('test_users', [])
                
                if filter_test_users and test_users:
                    if user_id not in test_users:
                        continue  # Skip users not in test list

                # Check if we've switched to a new user
                if user_id != current_user_id:
                    # Save previous user's results if exists
                    if current_processor and current_processor.is_initialized:
                        self._save_user_results(current_user_id, current_processor)
                        # Create visualization immediately for this user
                        if self.config.get('visualization', {}).get('enabled', True):
                            self._create_user_visualization(current_user_id, current_processor)

                    # Check if this is a new user we haven't seen before
                    if user_id not in self.users:
                        # Check if we've reached max users limit BEFORE adding the new user
                        if max_users > 0 and len(self.users) >= max_users:
                            # We've reached the limit - stop processing entirely
                            logger.info(f"Reached max_users limit ({max_users}), stopping processing")
                            break

                        # Create processor for new user
                        self.users[user_id] = UserProcessor(
                            user_id,
                            self.config.get('processing', {})
                        )
                        self.stats['total_users'] = len(self.users)

                    # Switch to user (whether new or existing)
                    current_user_id = user_id
                    current_processor = self.users[user_id]

                # Process the reading
                try:
                    processed = current_processor.process_reading(row)

                    if processed:
                        self.stats['rows_processed'] += 1

                        # Track statistics
                        if processed.is_valid:
                            self.stats['total_accepted'] = self.stats.get('total_accepted', 0) + 1
                        else:
                            self.stats['total_rejected'] = self.stats.get('total_rejected', 0) + 1

                        # Log significant events (reduced verbosity for cleaner progress)
                        if not processed.is_valid:
                            logger.debug(f"User {user_id}: Rejected {processed.measurement.weight:.1f}kg - "
                                       f"{processed.rejection_reason}")
                        elif processed.trend_kg_per_day and abs(processed.trend_kg_per_day) > 0.1:
                            logger.debug(f"User {user_id}: Significant trend {processed.trend_kg_per_day:.3f} kg/day")

                except Exception as e:
                    logger.error(f"Error processing row {row_num} for user {user_id}: {e}")
                    self.stats['errors'] = self.stats.get('errors', 0) + 1
                    continue

        # Save last user's results and create visualization
        if current_processor and current_processor.is_initialized:
            self._save_user_results(current_user_id, current_processor)
            # Create visualization for last user
            if self.config.get('visualization', {}).get('enabled', True):
                self._create_user_visualization(current_user_id, current_processor)

        # Finish progress
        progress.finish(f"Processed {self.stats['total_rows']:,} rows from {self.stats['total_users']} users")

        self.stats['end_time'] = time.time()
        self._save_summary()

    def _save_user_results(self, user_id: str, processor: UserProcessor):
        """Save individual user results."""
        results = processor.get_results()

        # Save to individual file in results subdirectory
        user_file = self.results_dir / f"user_{user_id}.json"
        with open(user_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    def _create_user_visualization(self, user_id: str, processor: UserProcessor):
        """Create visualization for a single user."""
        try:
            results = processor.get_results()

            # Use enhanced dashboard if configured
            use_enhanced = self.config.get('visualization', {}).get('use_enhanced', True)

            if use_enhanced:
                from src.visualization.enhanced_dashboard import EnhancedDashboard
                dashboard = EnhancedDashboard(user_id, results, str(self.viz_dir))
            else:
                from src.visualization import UserDashboard
                dashboard = UserDashboard(user_id, results, str(self.viz_dir))

            path = dashboard.create_dashboard()
            if path:
                logger.debug(f"Created visualization for user {user_id}: {path}")
                return path
        except Exception as e:
            logger.warning(f"Could not create visualization for user {user_id}: {e}")

    def _save_summary(self):
        """Save processing summary and aggregate results."""
        elapsed = self.stats['end_time'] - self.stats['start_time']

        # Aggregate user results
        all_results = {}
        total_accepted = 0
        total_rejected = 0
        outlier_types = {}

        for user_id, processor in self.users.items():
            if processor.is_initialized:
                results = processor.get_results()
                all_results[user_id] = results

                # Aggregate statistics
                total_accepted += results['stats']['accepted_readings']
                total_rejected += results['stats']['rejected_readings']

                for outlier_type, count in results['stats'].get('outliers_by_type', {}).items():
                    outlier_types[outlier_type] = outlier_types.get(outlier_type, 0) + count

        # Summary statistics
        summary = {
            'processing_stats': {
                'total_rows': self.stats['total_rows'],
                'total_users': self.stats['total_users'],
                'initialized_users': len(all_results),
                'processing_time_seconds': elapsed,
                'rows_per_second': self.stats['total_rows'] / elapsed if elapsed > 0 else 0,
                'users_per_second': self.stats['total_users'] / elapsed if elapsed > 0 else 0
            },
            'data_quality': {
                'total_accepted': total_accepted,
                'total_rejected': total_rejected,
                'acceptance_rate': total_accepted / (total_accepted + total_rejected) if (total_accepted + total_rejected) > 0 else 0,
                'outlier_types': outlier_types
            },
            'timestamp': datetime.now().isoformat()
        }

        # Save summary to results directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = self.results_dir / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save all results to results directory
        results_file = self.results_dir / f"all_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        # Count existing visualizations
        viz_count = 0
        if self.config.get('visualization', {}).get('enabled', True) and self.viz_dir.exists():
            viz_files = list(self.viz_dir.glob("dashboard_*.png"))
            viz_count = len(viz_files)

        # Print summary with colored status
        ProcessingStatus.section("Processing Complete")

        # Main statistics
        ProcessingStatus.success(f"Total rows processed: {self.stats['total_rows']:,}")
        ProcessingStatus.success(f"Total users: {self.stats['total_users']:,}")
        ProcessingStatus.success(f"Initialized users: {len(all_results):,}")

        # Performance metrics
        ProcessingStatus.info(f"Processing time: {elapsed:.1f} seconds")
        ProcessingStatus.info(f"Processing rate: {self.stats['total_rows'] / elapsed:.0f} rows/sec")

        # Data quality
        acceptance_rate = summary['data_quality']['acceptance_rate']
        if acceptance_rate >= 0.9:
            ProcessingStatus.success(f"Acceptance rate: {acceptance_rate:.1%}")
        elif acceptance_rate >= 0.7:
            ProcessingStatus.warning(f"Acceptance rate: {acceptance_rate:.1%}")
        else:
            ProcessingStatus.error(f"Low acceptance rate: {acceptance_rate:.1%}")

        # Output files
        print("\n📁 Output Files Created:")
        print(f"  • Results: {self.results_dir}")
        print(f"    - Summary: {summary_file.name}")
        print(f"    - All results: {results_file.name}")
        print(f"    - User files: {len(all_results)} files")

        if self.config.get('visualization', {}).get('enabled', True) and viz_count > 0:
            print(f"  • Visualizations: {self.viz_dir}")
            print(f"    - Dashboards: {viz_count} files")

        print(f"  • Logs: {self.logs_dir}/app.log")

        ProcessingStatus.section("Done")


def main():
    """Main entry point."""
    # Load configuration
    config = config_loader.load_config()

    # Get source file
    source_file = config.get('source_file')
    if not source_file or not Path(source_file).exists():
        logger.error(f"Source file not found: {source_file}")
        return

    # Create processor
    processor = StreamProcessor()

    # Process file
    processor.process_file(source_file)


if __name__ == "__main__":
    main()


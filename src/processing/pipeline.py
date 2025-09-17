"""
Shared Processing Pipeline

Reusable functions for processing weight measurements through the full
Kalman filtering, quality scoring, and outlier detection pipeline.

This module is used by both main.py and report.py to ensure consistent processing.
"""

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

from src.database.database import get_state_db
from src.processing.processor import process_measurement
from src.processing.validation import DataQualityPreprocessor


class ProcessingPipeline:
    """Shared pipeline for processing weight measurements"""

    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        """
        Initialize processing pipeline

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.db = None
        self.replay_components = None

    def initialize(self) -> None:
        """Initialize pipeline components"""
        # Load height data for BMI calculations
        DataQualityPreprocessor.load_height_data()

        # Initialize database
        self.db = get_state_db()

        # Initialize replay components if enabled
        self.replay_components = self._initialize_replay_processing()

    def _initialize_replay_processing(self) -> Optional[Dict]:
        """Initialize replay processing components if enabled"""
        replay_config = self.config.get("replay", {})
        replay_enabled = replay_config.get("enabled", False)

        if not replay_enabled:
            return None

        try:
            from src.processing.buffer_factory import get_factory
            from src.processing.outlier_detection import OutlierDetector
            from src.replay.replay_manager import ReplayManager

            # Create buffer using factory
            buffer_factory = get_factory()
            buffer_factory.set_default_config(replay_config)
            replay_buffer = buffer_factory.create_buffer('default', replay_config)

            outlier_detector = OutlierDetector(
                replay_config.get("outlier_detection", {}),
                db=self.db
            )
            replay_manager = ReplayManager(
                self.db,
                replay_config.get("safety", {})
            )

            self.logger.info("Replay processing initialized")
            self.logger.info(f"  Buffer window: {replay_config.get('buffer_hours', 72)} hours")
            self.logger.info(f"  Trigger mode: {replay_config.get('trigger_mode', 'time_based')}")

            return {
                'enabled': True,
                'buffer': replay_buffer,
                'outlier_detector': outlier_detector,
                'replay_manager': replay_manager,
                'config': replay_config
            }

        except ImportError as e:
            self.logger.warning(f"Could not initialize replay processing: {e}")
            return None

    def process_csv_file(self,
                        csv_path: str,
                        output_callback: Optional[callable] = None,
                        filter_rejected: bool = True,
                        max_users: int = 0,
                        min_readings: int = 0,
                        user_offset: int = 0,
                        test_users: Optional[List[str]] = None) -> Dict:
        """
        Process a CSV file through the full pipeline

        Args:
            csv_path: Path to CSV file
            output_callback: Optional callback for each processed measurement
            filter_rejected: If True, only return accepted measurements
            max_users: Maximum number of users to process (0 = all)
            min_readings: Minimum readings required per user
            user_offset: Skip this many eligible users (for pagination)
            test_users: Optional list of specific users to process

        Returns:
            Dictionary with processed data and statistics
        """
        if not self.db:
            self.initialize()

        # Parse date filters from config
        min_date_str = self.config.get("data", {}).get("min_date", "")
        max_date_str = self.config.get("data", {}).get("max_date", "")
        min_date = self._parse_timestamp(min_date_str) if min_date_str else None
        max_date = self._parse_timestamp(max_date_str) if max_date_str else None

        # Initialize tracking
        stats = {
            'total_rows': 0,
            'accepted': 0,
            'rejected': 0,
            'outliers': 0,
            'date_filtered': 0,
            'parse_errors': 0,
            'invalid_weight': 0,
            'processing_errors': 0,
            'users': set()
        }

        processed_data = []
        rejected_data = []
        user_results = {}

        # Determine eligible users if filtering needed
        eligible_users = None
        if min_readings > 0 or max_users > 0 or user_offset > 0:
            eligible_users = self._get_eligible_users(
                csv_path, min_readings, max_users, user_offset, test_users,
                min_date, max_date
            )
            self.logger.info(f"Processing {len(eligible_users)} eligible users")

        # Main processing loop
        with open(csv_path) as f:
            reader = csv.DictReader(f)

            for row in reader:
                stats['total_rows'] += 1

                # Progress update
                if stats['total_rows'] % 1000 == 0:
                    self.logger.debug(f"Processed {stats['total_rows']:,} rows...")

                # Parse and validate row
                parsed = self._parse_measurement_row(row)
                if not parsed:
                    stats['parse_errors'] += 1
                    continue

                user_id, weight, timestamp, source, unit = parsed

                # User filtering
                if eligible_users and user_id not in eligible_users:
                    continue

                # Date filtering
                if min_date and timestamp < min_date:
                    stats['date_filtered'] += 1
                    continue
                if max_date and timestamp > max_date:
                    stats['date_filtered'] += 1
                    continue

                # Weight validation
                if not self._validate_weight(weight):
                    stats['invalid_weight'] += 1
                    continue

                # Process through full pipeline
                try:
                    result = self._process_single_measurement(
                        user_id, weight, timestamp, source, unit
                    )

                    if result:
                        stats['users'].add(user_id)

                        # Track user results
                        if user_id not in user_results:
                            user_results[user_id] = []
                        user_results[user_id].append(result)

                        # Handle replay processing
                        if self.replay_components and self.replay_components['enabled']:
                            self._add_to_replay_buffer(
                                user_id, weight, timestamp, source, unit, result
                            )
                            # Check for replay trigger
                            self._check_replay_trigger(user_id)

                        # Categorize result
                        if result.get('is_outlier', False):
                            stats['outliers'] += 1
                            stats['rejected'] += 1
                            rejected_data.append(self._create_output_row(
                                row, user_id, weight, timestamp, source, result
                            ))
                        elif result.get('accepted', True):
                            stats['accepted'] += 1
                            processed_data.append(self._create_output_row(
                                row, user_id, weight, timestamp, source, result
                            ))
                        else:
                            stats['rejected'] += 1
                            rejected_data.append(self._create_output_row(
                                row, user_id, weight, timestamp, source, result
                            ))

                        # Call output callback if provided
                        if output_callback:
                            output_callback(user_id, result)

                except Exception as e:
                    stats['processing_errors'] += 1
                    self.logger.error(f"Error processing {user_id}: {e}")

        return {
            'processed_data': processed_data,
            'rejected_data': rejected_data,
            'user_results': user_results,
            'statistics': stats
        }

    def _parse_measurement_row(self, row: Dict) -> Optional[Tuple]:
        """Parse a CSV row into measurement components"""
        user_id = row.get("user_id")
        if not user_id:
            return None

        # Parse weight
        weight_str = row.get("weight", "").strip()
        if not weight_str or weight_str.upper() == "NULL":
            return None

        try:
            weight = float(weight_str)
        except (ValueError, TypeError):
            return None

        # Parse metadata
        date_str = row.get("timestamp") or row.get("effectiveDateTime")
        source = row.get("source") or row.get("source_type", "unknown")
        unit = (row.get("unit") or "kg").lower().strip()

        # Skip BSA measurements
        if 'BSA' in source.upper() or 'm2' in unit or 'm²' in unit:
            return None

        # Parse timestamp
        try:
            timestamp = self._parse_timestamp(date_str)
        except:
            timestamp = datetime.now()

        return user_id, weight, timestamp, source, unit

    def _validate_weight(self, weight: float) -> bool:
        """Validate weight value"""
        if weight <= 0 or weight > 1000:
            return False
        if math.isnan(weight) or math.isinf(weight):
            return False
        return True

    def _process_single_measurement(self,
                                   user_id: str,
                                   weight: float,
                                   timestamp: datetime,
                                   source: str,
                                   unit: str) -> Optional[Dict]:
        """Process a single measurement through the pipeline"""
        # Merge quality scoring config
        full_config = self.config.copy()
        if 'quality_scoring' not in full_config:
            full_config['quality_scoring'] = {}

        result = process_measurement(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source,
            config=full_config,
            unit=unit,
            db=self.db
        )

        return result

    def _add_to_replay_buffer(self,
                            user_id: str,
                            weight: float,
                            timestamp: datetime,
                            source: str,
                            unit: str,
                            result: Dict) -> None:
        """Add measurement to replay buffer if enabled"""
        if not self.replay_components or not self.replay_components['buffer']:
            return

        measurement_data = {
            'weight': weight,
            'timestamp': timestamp,
            'source': source,
            'unit': unit,
            'metadata': {
                'accepted': result.get('accepted', False),
                'rejection_reason': result.get('rejection_reason', None),
                'quality_score': result.get('quality_score', None),
                'quality_components': result.get('quality_components', None)
            }
        }

        self.replay_components['buffer'].add_measurement(user_id, measurement_data)

    def _check_replay_trigger(self, user_id: str) -> None:
        """Check if replay should be triggered for a user"""
        if not self.replay_components:
            return

        buffer = self.replay_components['buffer']
        trigger_mode = self.replay_components['config'].get('trigger_mode', 'time_based')

        should_trigger = False
        trigger_reason = None

        if trigger_mode == 'time_based':
            if buffer.should_trigger(user_id):
                should_trigger = True
                trigger_reason = 'time_threshold'
        elif trigger_mode == 'count_based':
            count_threshold = self.replay_components['config'].get('count_threshold', 10)
            if len(buffer.get_measurements(user_id)) >= count_threshold:
                should_trigger = True
                trigger_reason = f'count_threshold ({count_threshold})'

        if should_trigger:
            self._perform_replay(user_id, trigger_reason)

    def _perform_replay(self, user_id: str, trigger_reason: str) -> None:
        """Perform replay processing for a user"""
        if not self.replay_components:
            return

        self.logger.debug(f"Triggering replay for {user_id}: {trigger_reason}")

        buffer = self.replay_components['buffer']
        outlier_detector = self.replay_components['outlier_detector']
        replay_manager = self.replay_components['replay_manager']

        # Get buffered measurements
        measurements = buffer.get_measurements(user_id)
        if not measurements:
            return

        # Run outlier detection
        outliers = outlier_detector.detect_outliers(user_id, measurements)

        # Apply safety checks
        safe_to_process = replay_manager.validate_replay(user_id, measurements, outliers)

        if safe_to_process:
            # Process replay
            replay_manager.process_replay(user_id, measurements, outliers)

        # Clear buffer after processing
        buffer.clear_user(user_id)

    def _create_output_row(self,
                         original_row: Dict,
                         user_id: str,
                         weight: float,
                         timestamp: datetime,
                         source: str,
                         result: Dict) -> Dict:
        """Create output row for CSV"""
        return {
            'user_id': user_id,
            'weight': result.get('filtered_weight', weight),
            'raw_weight': weight,
            'timestamp': timestamp.isoformat(),
            'source': source,
            'quality_score': result.get('quality_score', 0.5),
            'bmi': result.get('bmi'),
            'kalman_weight': result.get('filtered_weight'),
            'kalman_trend': result.get('kalman_trend'),
            'is_outlier': result.get('is_outlier', False),
            'rejection_reason': result.get('rejection_reason'),
            'measurement_id': original_row.get('id', '')
        }

    def _get_eligible_users(self,
                          csv_path: str,
                          min_readings: int,
                          max_users: int,
                          user_offset: int,
                          test_users: Optional[List[str]],
                          min_date: Optional[datetime],
                          max_date: Optional[datetime]) -> set:
        """Determine eligible users based on criteria"""
        if test_users:
            return set(test_users)

        user_counts = {}

        # Count readings per user
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = self._parse_measurement_row(row)
                if not parsed:
                    continue

                user_id, weight, timestamp, _, _ = parsed

                # Apply date filter
                if min_date and timestamp < min_date:
                    continue
                if max_date and timestamp > max_date:
                    continue

                # Validate weight
                if not self._validate_weight(weight):
                    continue

                user_counts[user_id] = user_counts.get(user_id, 0) + 1

        # Filter by min_readings
        eligible = []
        for user_id, count in sorted(user_counts.items()):
            if count >= min_readings:
                eligible.append(user_id)

        # Apply user_offset
        if user_offset > 0 and user_offset < len(eligible):
            eligible = eligible[user_offset:]
        elif user_offset >= len(eligible):
            eligible = []  # Offset beyond available users

        # Apply max_users limit
        if max_users > 0 and len(eligible) > max_users:
            eligible = eligible[:max_users]

        return set(eligible)

    def _parse_timestamp(self, date_str: str) -> datetime:
        """Parse various timestamp formats"""
        if isinstance(date_str, datetime):
            return date_str

        # Try different formats
        formats = [
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue

        # If all fail, try pandas
        import pandas as pd
        return pd.to_datetime(date_str)
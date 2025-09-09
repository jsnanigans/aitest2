"""
User-level processor that manages the pipeline for individual users.
Handles streaming data and maintains per-user state.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.types import ProcessedMeasurement, WeightMeasurement
from src.processing.weight_pipeline import WeightProcessingPipeline

logger = logging.getLogger(__name__)


class UserProcessor:
    """Manages weight processing pipeline for a single user."""

    def __init__(self, user_id: str, config: Optional[Dict[str, Any]] = None):
        self.user_id = user_id
        self.config = config or {}

        # Initialize pipeline
        self.pipeline = WeightProcessingPipeline(config)

        # User statistics
        self.stats = {
            'total_readings': 0,
            'accepted_readings': 0,
            'rejected_readings': 0,
            'outliers_by_type': {},
            'first_reading': None,
            'last_reading': None,
            'min_weight': float('inf'),
            'max_weight': float('-inf')
        }

        # Store raw and processed measurements
        self.raw_measurements: List[WeightMeasurement] = []
        self.processed_measurements: List[ProcessedMeasurement] = []

        # Initialization state
        self.is_initialized = False
        self.initialization_buffer: List[WeightMeasurement] = []
        self.min_init_readings = config.get('min_init_readings', 3)

    def process_reading(self, row: Dict[str, Any]) -> Optional[ProcessedMeasurement]:
        """
        Process a single reading from CSV row.

        Args:
            row: Dictionary containing weight data

        Returns:
            ProcessedMeasurement or None if initialization pending
        """
        # Handle NULL or invalid weight values
        weight_str = row.get('weight', '').strip()
        if not weight_str or weight_str.upper() == 'NULL':
            return None

        try:
            weight = float(weight_str)
        except (ValueError, TypeError):
            logger.debug(f"Invalid weight value: {weight_str}")
            return None

        # Create measurement object
        # Handle both possible column names for date and source
        date_value = row.get('effectivDateTime') or row.get('date')
        source_value = row.get('source_type') or row.get('source')

        # Debug logging for first few readings
        if self.stats['total_readings'] < 3:
            logger.debug(f"Processing row for {self.user_id}: date={date_value}, source={source_value}, weight={weight}")

        measurement = WeightMeasurement(
            weight=weight,
            timestamp=self._parse_timestamp(date_value),
            source_type=source_value,
            user_id=self.user_id,
            raw_data=row
        )

        # Update basic stats
        self.stats['total_readings'] += 1
        self.stats['min_weight'] = min(self.stats['min_weight'], measurement.weight)
        self.stats['max_weight'] = max(self.stats['max_weight'], measurement.weight)

        if not self.stats['first_reading']:
            self.stats['first_reading'] = measurement.timestamp
        self.stats['last_reading'] = measurement.timestamp

        # Store raw measurement
        self.raw_measurements.append(measurement)

        # Handle initialization phase
        if not self.is_initialized:
            self.initialization_buffer.append(measurement)

            if len(self.initialization_buffer) >= self.min_init_readings:
                # Attempt initialization
                baseline_result = self.pipeline.initialize_user(self.initialization_buffer)

                if baseline_result.success:
                    self.is_initialized = True
                    logger.info(f"User {self.user_id} initialized with baseline {baseline_result.baseline_weight:.1f}kg")

                    # Process buffered measurements
                    for buffered in self.initialization_buffer:
                        processed = self.pipeline.process_measurement(buffered)
                        self._update_stats(processed)
                        self.processed_measurements.append(processed)

                    # Clear buffer
                    self.initialization_buffer = []

                    # Return last processed measurement
                    return self.processed_measurements[-1] if self.processed_measurements else None
                else:
                    logger.debug(f"User {self.user_id} initialization pending: {baseline_result.error}")

            return None

        # Normal processing (post-initialization)
        processed = self.pipeline.process_measurement(measurement)
        self._update_stats(processed)
        self.processed_measurements.append(processed)

        return processed

    def _update_stats(self, processed: ProcessedMeasurement):
        """Update statistics based on processed measurement."""
        if processed.is_valid:
            self.stats['accepted_readings'] += 1
        else:
            self.stats['rejected_readings'] += 1
            if processed.outlier_type:
                outlier_type = processed.outlier_type.value
                self.stats['outliers_by_type'][outlier_type] = \
                    self.stats['outliers_by_type'].get(outlier_type, 0) + 1

    def _parse_timestamp(self, date_str: Optional[str]) -> datetime:
        """Parse timestamp from various formats."""
        if not date_str:
            logger.debug("Empty date string, using current time")
            return datetime.now()

        try:
            # Try ISO format with T separator
            if 'T' in date_str:
                parsed = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                logger.debug(f"Parsed ISO date: {date_str} -> {parsed}")
                return parsed
            # Try datetime format with space separator (our CSV format)
            elif ' ' in date_str:
                parsed = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                logger.debug(f"Parsed datetime: {date_str} -> {parsed}")
                return parsed
            # Try simple date format
            else:
                parsed = datetime.strptime(date_str, '%Y-%m-%d')
                logger.debug(f"Parsed date: {date_str} -> {parsed}")
                return parsed
        except Exception as e:
            logger.warning(f"Failed to parse timestamp '{date_str}': {e}")
            return datetime.now()

    def get_results(self) -> Dict[str, Any]:
        """Get complete processing results for the user."""
        if not self.is_initialized:
            return {
                'user_id': self.user_id,
                'initialized': False,
                'readings_buffered': len(self.initialization_buffer),
                'stats': self.stats
            }

        # Get pipeline state
        pipeline_state = self.pipeline.get_state_summary()

        # Build time series data
        time_series = []
        for pm in self.processed_measurements:
            time_series.append({
                'date': pm.measurement.timestamp.isoformat(),
                'weight': pm.measurement.weight,
                'filtered_weight': pm.filtered_weight,
                'predicted_weight': pm.predicted_weight,
                'is_valid': pm.is_valid,
                'confidence': pm.confidence,
                'trend_kg_per_day': pm.trend_kg_per_day,
                'outlier_type': pm.outlier_type.value if pm.outlier_type else None,
                'source': pm.measurement.source_type
            })

        # Calculate derived statistics
        valid_weights = [pm.measurement.weight for pm in self.processed_measurements if pm.is_valid]

        result = {
            'user_id': self.user_id,
            'initialized': True,
            'baseline': pipeline_state['baseline'],
            'current_state': pipeline_state['current_state'],
            'stats': {
                **self.stats,
                'acceptance_rate': pipeline_state['acceptance_rate'],
                'average_weight': sum(valid_weights) / len(valid_weights) if valid_weights else 0,
                'weight_range': self.stats['max_weight'] - self.stats['min_weight']
            },
            'time_series': time_series
        }

        return result

    def should_reinitialize(self, gap_days: int = 90) -> bool:
        """
        Check if user should be reinitialized due to long gap.
        Framework suggests considering gaps > 30 days.
        """
        if not self.processed_measurements:
            return False

        last_measurement = self.processed_measurements[-1].measurement
        current_time = datetime.now()

        gap = (current_time - last_measurement.timestamp).days
        return gap > gap_days


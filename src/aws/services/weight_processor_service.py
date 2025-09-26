"""Service layer for weight processing operations."""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.aws.api.models import (
    Measurement,
    MeasurementResult,
    ProcessResponseData,
    CleanupResponseData,
    StateInfo,
    HistoricalConflictDetails,
    HistoricalConflictResponse,
)
from src.core.database.base import StateStore
from src.core.processing.processor import process_measurement
from src.aws.config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class HistoricalConflictError(Exception):
    """Exception raised when measurements are before last processed timestamp."""

    def __init__(self, conflict_response: HistoricalConflictResponse):
        self.conflict_response = conflict_response
        super().__init__(conflict_response.error)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.conflict_response.model_dump()


class WeightProcessorService:
    """Service layer for weight processing operations."""

    def __init__(self, state_store: StateStore = None, config: Dict[str, Any] = None):
        """
        Initialize service.

        Args:
            state_store: State storage backend
            config: Configuration dictionary
        """
        # Use factory pattern if not provided
        if state_store is None:
            from ..database import get_state_db

            state_store = get_state_db()

        self.state_store = state_store
        self.config = config or ConfigManager.load_config()

    def process_batch(
        self, user_id: str, measurements: List[Measurement]
    ) -> ProcessResponseData:
        """
        Process a batch of measurements for a user.

        Args:
            user_id: User identifier
            measurements: List of measurements to process

        Returns:
            ProcessResponseData with results for all measurements

        Raises:
            HistoricalConflictError: If measurements are before last processed timestamp
        """
        # Sort measurements chronologically
        sorted_measurements = sorted(measurements, key=lambda m: m.measured_at)

        # Check for historical conflicts
        conflict = self._check_historical_conflict(user_id, sorted_measurements)
        if conflict:
            raise HistoricalConflictError(conflict)

        # Get initial state
        current_state = self.state_store.get_state(user_id)
        previous_weight = (
            current_state.get("last_raw_weight") if current_state else None
        )

        # Process each measurement
        results = []
        accepted_count = 0
        rejected_count = 0

        for measurement in sorted_measurements:
            try:
                result = self._process_single(user_id, measurement)
                results.append(result)

                if result.accepted:
                    accepted_count += 1
                else:
                    rejected_count += 1

            except Exception as e:
                logger.error(f"Error processing {measurement.measurement_id}: {e}")
                results.append(
                    MeasurementResult(
                        measurement_id=measurement.measurement_id,
                        accepted=False,
                        rejection_reason=str(e),
                        processing_stage="processing",
                    )
                )
                rejected_count += 1

        # Get final state
        final_state = self.state_store.get_state(user_id)
        current_weight = final_state.get("last_raw_weight") if final_state else None

        # Create state update
        state_update = None
        if sorted_measurements:
            state_update = StateInfo(
                user_id=user_id,
                previous_weight=previous_weight,
                current_weight=current_weight,
                last_processed_at=sorted_measurements[-1].measured_at,
            )

        return ProcessResponseData(
            user_id=user_id,
            measurements_processed=len(results),
            measurements_accepted=accepted_count,
            measurements_rejected=rejected_count,
            results=results,
            state_update=state_update,
        )

    def cleanup(
        self, user_id: str, measurements: List[Measurement], reset_state: bool = True
    ) -> CleanupResponseData:
        """
        Perform one-time cleanup for a user.

        Args:
            user_id: User identifier
            measurements: All historical measurements
            reset_state: Whether to reset state before processing

        Returns:
            CleanupResponseData with results for all measurements
        """
        # Reset state if requested
        if reset_state:
            self.state_store.delete_state(user_id)
            logger.info(f"Reset state for user {user_id}")

        # Sort measurements chronologically
        sorted_measurements = sorted(measurements, key=lambda m: m.measured_at)

        # Process all measurements
        results = []
        accepted_count = 0
        rejected_count = 0

        for measurement in sorted_measurements:
            try:
                result = self._process_single(user_id, measurement)
                results.append(result)

                if result.accepted:
                    accepted_count += 1
                else:
                    rejected_count += 1

            except Exception as e:
                logger.error(f"Error processing {measurement.measurement_id}: {e}")
                results.append(
                    MeasurementResult(
                        measurement_id=measurement.measurement_id,
                        accepted=False,
                        rejection_reason=str(e),
                        processing_stage="processing",
                    )
                )
                rejected_count += 1

        # Get final state
        final_state_data = self.state_store.get_state(user_id)
        final_state = None

        if final_state_data:
            final_state = StateInfo(
                user_id=user_id,
                current_weight=final_state_data.get("last_raw_weight", 0),
                previous_weight=None,
                last_processed_at=final_state_data.get(
                    "last_timestamp", datetime.now()
                ),
                measurements_count=len(results),
                last_source=None,
                adaptation_state="converged"
                if final_state_data.get("measurements_since_reset", 0) > 10
                else "adapting",
            )

        return CleanupResponseData(
            user_id=user_id,
            cleanup_type="reset_adaptive" if reset_state else "cleanup",
            measurements_processed=len(results),
            state_cleared=reset_state,
            message=f"Processed {len(results)} measurements"
        )

    def _process_single(
        self, user_id: str, measurement: Measurement
    ) -> MeasurementResult:
        """Process a single measurement."""
        # Call the existing processor
        result = process_measurement(
            user_id=user_id,
            weight=measurement.weight_value,
            timestamp=measurement.measured_at,
            source=measurement.source,
            unit=measurement.weight_unit,
            config=self.config,
            db=self.state_store,
        )

        # Convert to API model
        return MeasurementResult(
            measurement_id=measurement.measurement_id,
            accepted=result.get("accepted", False),
            quality_score=result.get("quality_score"),
            kalman_estimate=result.get("kalman_estimate"),
            kalman_uncertainty=result.get("kalman_uncertainty"),
            rejection_reason=result.get("reason"),
            processing_stage=result.get("stage"),
            reset_triggered=result.get("reset_triggered", False),
            quality_components=result.get("quality_components"),
        )

    def _check_historical_conflict(
        self, user_id: str, measurements: List[Measurement]
    ) -> Optional[HistoricalConflictResponse]:
        """Check if any measurements are before last processed timestamp."""
        current_state = self.state_store.get_state(user_id)

        if not current_state or not current_state.get("last_timestamp"):
            return None  # No conflict if no previous state

        last_timestamp = current_state["last_timestamp"]
        if isinstance(last_timestamp, str):
            last_timestamp = datetime.fromisoformat(last_timestamp)

        # Find conflicting measurements
        conflicting = [
            str(m.measurement_id) for m in measurements if m.measured_at < last_timestamp
        ]

        if not conflicting:
            return None  # No conflict

        # Get earliest measurement
        earliest = min(measurements, key=lambda m: m.measured_at)

        # Check for available snapshot
        snapshot = self.state_store.get_snapshot(user_id, earliest.measured_at)
        snapshot_time = None
        if snapshot and "snapshotTime" in snapshot:
            snapshot_time = datetime.fromisoformat(snapshot["snapshotTime"])

        return HistoricalConflictResponse(
            error="One or more measurements are before last processed timestamp",
            details=HistoricalConflictDetails(
                earliest_measurement_timestamp=earliest.measured_at,
                last_processed_timestamp=last_timestamp,
                replay_from_timestamp=earliest.measured_at,
                snapshot_available=snapshot_time,
                conflicting_measurements=conflicting,
            ),
        )

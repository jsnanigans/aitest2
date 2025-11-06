"""Service layer for weight processing operations."""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from src.aws.api.models import (
    Measurement,
    MeasurementResult,
    ProcessResponseData,
    CleanupResponseData,
    StateInfo,
    HistoricalConflictDetails,
    HistoricalConflictResponse,
    ReplayWindowInfo,
    ReplayResultData,
    ReplayTriggerCheckResponse,
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
        self, user_id: str, measurements: List[Measurement], user_height_m: Optional[float] = None
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

        # Check if buffered replay is enabled
        buffered_replay_enabled = self.config.get("replay", {}).get("buffered_replay_enabled", True)

        # Initialize buffer for replay processing (only if enabled)
        buffer: List[Measurement] = []
        buffer_start_time: Optional[datetime] = None
        replay_metadata: List[Dict[str, Any]] = []

        # Process each measurement
        results = []
        accepted_count = 0
        rejected_count = 0

        for i, measurement in enumerate(sorted_measurements):
            try:
                result = self._process_single(user_id, measurement, user_height_m)
                results.append(result)

                if result.accepted:
                    accepted_count += 1
                else:
                    rejected_count += 1

                # Buffer management: Buffer ALL measurements (accepted or rejected) for replay
                # Replay will re-evaluate them together to determine best accept/reject decisions
                if buffered_replay_enabled:
                    if not buffer:
                        buffer_start_time = measurement.measured_at
                        self.state_store.save_state_snapshot(user_id, buffer_start_time)
                        logger.info(f"Created snapshot for user {user_id} at {buffer_start_time}")

                    # Add ALL measurements to buffer (not just accepted ones)
                    # Replay will re-score them all against the snapshot state
                    buffer.append(measurement)

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

            # Check if replay should be triggered (only if feature is enabled)
            if buffered_replay_enabled:
                # Filter buffer to only keep measurements within the time window (sliding window)
                buffer_hours = self.config.get("replay", {}).get("buffer_hours", 24)
                cutoff_time = measurement.measured_at - timedelta(hours=buffer_hours)
                buffer = [m for m in buffer if m.measured_at >= cutoff_time]

                logger.debug(f"  Buffer size after filtering: {len(buffer)} measurements")

                is_last = (i == len(sorted_measurements) - 1)
                should_replay = self._should_trigger_replay(buffer, measurement.measured_at, is_last)

                if should_replay and buffer:
                    # Determine trigger reason
                    if is_last:
                        trigger_reason = "batch_end"
                    elif len(buffer) >= self.config.get("replay", {}).get("max_buffer_measurements", 100):
                        trigger_reason = "buffer_overflow"
                    else:
                        trigger_reason = "time_window"

                    # Console output to match TypeScript logging
                    print(
                        f"Triggering replay for user {user_id}: trigger={trigger_reason}, "
                        f"buffer_size={len(buffer)}, time_range={buffer[0].measured_at} to {buffer[-1].measured_at}"
                    )
                    logger.info(
                        f"Triggering replay for user {user_id}: trigger={trigger_reason}, "
                        f"buffer_size={len(buffer)}, time_range={buffer[0].measured_at} to {buffer[-1].measured_at}"
                    )

                    # Execute replay
                    replay_output = self._execute_buffered_replay(
                        user_id, buffer, buffer_start_time, user_height_m
                    )

                    # Merge replay results into original results
                    results = self._merge_replay_results(results, replay_output, buffer)

                    # Track replay metadata
                    replay_metadata.append({
                        "trigger": trigger_reason,
                        "buffer_size": len(buffer),
                        "replay_from": buffer_start_time.isoformat(),
                        "replay_to": buffer[-1].measured_at.isoformat(),
                        "measurements_replayed": len(buffer),
                        "duration_seconds": replay_output.get("duration_seconds", 0),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                    # Clear buffer for next window
                    buffer.clear()
                    buffer_start_time = None

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
            replay_metadata=replay_metadata if replay_metadata else None,
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
                    "last_timestamp", datetime.now(timezone.utc)
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
        self, user_id: str, measurement: Measurement, user_height_m: Optional[float] = None
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
            user_height_m=user_height_m,
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
            # Ensure timezone-aware datetime
            if "+" in last_timestamp or "Z" in last_timestamp:
                last_timestamp = datetime.fromisoformat(last_timestamp.replace("Z", "+00:00"))
            else:
                # Assume UTC if no timezone info
                last_timestamp = datetime.fromisoformat(last_timestamp)
                if last_timestamp.tzinfo is None:
                    last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)

        # Find conflicting measurements
        conflicting = [
            m.measurement_id for m in measurements if m.measured_at < last_timestamp
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

    def should_trigger_replay(
        self,
        user_id: str,
        current_timestamp: datetime,
        buffer_hours: int = None
    ) -> ReplayTriggerCheckResponse:
        """
        Check if replay should trigger after processing a measurement.

        This method provides information to help the CALLER decide whether
        to trigger replay. It does NOT execute replay automatically.

        Replay should trigger when there are measurements in the buffer window
        before the current timestamp.

        Args:
            user_id: User identifier
            current_timestamp: Timestamp of measurement just processed
            buffer_hours: Size of replay window in hours (default: from config)

        Returns:
            ReplayTriggerCheckResponse with trigger recommendation and window info
        """
        # Get buffer hours from config if not provided
        if buffer_hours is None:
            buffer_hours = self.config.get("replay", {}).get("buffer_hours", 24)

        # Calculate window boundaries
        from datetime import timedelta
        window_start = current_timestamp - timedelta(hours=buffer_hours)
        window_end = current_timestamp

        # Query measurements in window
        measurements = self.state_store.get_measurements_in_window(
            user_id, window_start, window_end
        )

        # Trigger if there are measurements in window
        should_trigger = len(measurements) > 0

        if should_trigger:
            # Extract measurement IDs for tracking
            measurement_ids = [
                m.get("metadata", {}).get("measurement_id", "")
                for m in measurements
                if m.get("metadata")
            ]

            window_info = ReplayWindowInfo(
                window_start=window_start,
                window_end=window_end,
                measurements_in_window=len(measurements),
                measurement_ids=measurement_ids
            )

            return ReplayTriggerCheckResponse(
                should_trigger=True,
                window_info=window_info
            )
        else:
            return ReplayTriggerCheckResponse(
                should_trigger=False,
                window_info=None
            )

    def execute_replay(
        self,
        user_id: str,
        window_info: ReplayWindowInfo,
        measurements_to_replay: Optional[List[Measurement]] = None
    ) -> ReplayResultData:
        """
        Execute replay for a measurement window.

        The CALLER triggers this method and must handle the results by updating
        acceptance tracking. This method:
        1. Restores state to before window
        2. Detects outliers using pre-window state
        3. Replays clean measurements chronologically
        4. Returns NEW acceptance results for caller to process

        Args:
            user_id: User identifier
            window_info: Window information from should_trigger_replay()
            measurements_to_replay: Optional list of measurements (if None, queries from DB)

        Returns:
            ReplayResultData containing NEW acceptance results

        IMPORTANT: Caller must update acceptance tracking based on results!
        """
        try:
            # Import replay components
            from src.core.replay.replay_manager import ReplayManager
            from src.core.processing.outlier_detection import OutlierDetector

            # Get measurements if not provided
            if measurements_to_replay is None:
                measurements_dict = self.state_store.get_measurements_in_window(
                    user_id, window_info.window_start, window_info.window_end
                )
            else:
                # Convert Measurement objects to dict format
                measurements_dict = [
                    {
                        "weight": m.weight_value,
                        "timestamp": m.measured_at,
                        "source": m.source,
                        "unit": m.weight_unit,
                        "metadata": m.metadata or {}
                    }
                    for m in measurements_to_replay
                ]

            if not measurements_dict:
                return ReplayResultData(
                    user_id=user_id,
                    success=False,
                    window_start=window_info.window_start,
                    window_end=window_info.window_end,
                    measurement_results=[],
                    error="No measurements found in window"
                )

            # Initialize replay manager
            replay_config = self.config.get("replay", {})
            replay_manager = ReplayManager(self.state_store, replay_config.get("safety", {}))

            # Execute replay using existing replay manager logic
            replay_result = replay_manager.replay_clean_measurements(
                user_id=user_id,
                clean_measurements=measurements_dict,
                buffer_start_time=window_info.window_start
            )

            if not replay_result.get("success"):
                return ReplayResultData(
                    user_id=user_id,
                    success=False,
                    window_start=window_info.window_start,
                    window_end=window_info.window_end,
                    measurement_results=[],
                    error=replay_result.get("error", "Replay failed")
                )

            # Get the final state to determine which measurements were accepted
            # Re-process measurements to get acceptance results
            measurement_results = []
            for m_dict in measurements_dict:
                result = self._process_single(
                    user_id,
                    Measurement(
                        measurement_id=m_dict.get("metadata", {}).get("measurement_id", ""),
                        weight_value=m_dict["weight"],
                        weight_unit=m_dict["unit"],
                        measured_at=m_dict["timestamp"],
                        source=m_dict["source"],
                        metadata=m_dict.get("metadata")
                    )
                )
                measurement_results.append(result)

            # Calculate corrections (accepted vs rejected changes)
            corrections_made = sum(1 for r in measurement_results if r.accepted)

            return ReplayResultData(
                user_id=user_id,
                success=True,
                window_start=window_info.window_start,
                window_end=window_info.window_end,
                measurement_results=measurement_results,
                outliers_detected=[],  # Would need to track from outlier detection
                outliers_count=0,
                corrections_made=corrections_made,
                state_restored_to=window_info.window_start
            )

        except Exception as e:
            logger.error(f"Error executing replay for user {user_id}: {e}")
            return ReplayResultData(
                user_id=user_id,
                success=False,
                window_start=window_info.window_start,
                window_end=window_info.window_end,
                measurement_results=[],
                error=str(e)
            )

    def _should_trigger_replay(
        self, buffer: List[Measurement], current_timestamp: datetime, is_last: bool
    ) -> bool:
        """
        Determine if replay should be triggered for the current buffer.

        Replay is triggered when:
        1. Last measurement in batch (is_last=True) AND buffer has >= 2 measurements
        2. Buffer has enough measurements for meaningful replay (>= min_replay_buffer_size)
        3. Buffer size limit reached (safety)

        Args:
            buffer: List of buffered measurements (already filtered to time window)
            current_timestamp: Timestamp of current measurement being processed
            is_last: Whether this is the last measurement in the batch

        Returns:
            True if replay should be triggered, False otherwise
        """
        # Minimum buffer size: need at least 2 measurements to replay
        if len(buffer) < 2:
            return False

        # Trigger 1: Last measurement in batch
        if is_last:
            return True

        # Trigger 2: Buffer has enough measurements within the time window for meaningful replay
        # This ensures we trigger replay when we have clustered measurements that should be
        # evaluated together, rather than waiting until batch end when they might be filtered out
        min_replay_size = self.config.get("replay", {}).get("min_replay_buffer_size", 2)
        if len(buffer) >= min_replay_size:
            return True

        # Trigger 3: Buffer size limit (safety)
        max_buffer = self.config.get("replay", {}).get("max_buffer_measurements", 100)
        if len(buffer) >= max_buffer:
            return True

        return False

    def _execute_buffered_replay(
        self,
        user_id: str,
        buffer: List[Measurement],
        buffer_start_time: datetime,
        user_height_m: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute replay for buffered measurements.

        Args:
            user_id: User identifier
            buffer: List of buffered measurements to replay
            buffer_start_time: Timestamp to replay from (snapshot timestamp)
            user_height_m: User height in meters (optional)

        Returns:
            Replay result dictionary with processing results (includes 'duration_seconds')

        Raises:
            Exception: If replay fails, exception is logged and re-raised
        """
        import time

        try:
            print(
                f"Executing buffered replay for user {user_id}: "
                f"buffer_size={len(buffer)}, replay_from={buffer_start_time}"
            )
            logger.info(
                f"Executing buffered replay for user {user_id}: "
                f"buffer_size={len(buffer)}, replay_from={buffer_start_time}"
            )

            # Track replay performance
            replay_start = time.time()

            # Import replay service
            from src.aws.services.replay_service import replay_measurements

            replay_output = replay_measurements(
                user_id=user_id,
                measurements=buffer,
                replay_from=buffer_start_time,
                state_store=self.state_store,
                config=self.config,
                user_height_m=user_height_m,
            )

            replay_duration = time.time() - replay_start

            if not replay_output.get("success", False):
                error_msg = replay_output.get("error", "Unknown error")
                logger.error(f"Replay failed for user {user_id}: {error_msg}")
                raise Exception(f"Replay failed: {error_msg}")

            logger.info(
                f"Replay completed for user {user_id}: "
                f"processed={replay_output.get('processed_count', 0)}, "
                f"accepted={replay_output.get('accepted_count', 0)}, "
                f"duration={replay_duration:.2f}s"
            )

            # Add duration to output
            replay_output["duration_seconds"] = round(replay_duration, 2)

            return replay_output

        except Exception as e:
            logger.error(f"Replay execution failed for user {user_id}: {str(e)}", exc_info=True)
            raise

    def _merge_replay_results(
        self,
        original_results: List[MeasurementResult],
        replay_output: Dict[str, Any],
        buffer: List[Measurement],
    ) -> List[MeasurementResult]:
        """
        Merge replay results back into original results list.

        Args:
            original_results: Original processing results
            replay_output: Replay service output dictionary
            buffer: List of buffered measurements that were replayed

        Returns:
            Updated list of MeasurementResult with replay data merged
        """
        # Create lookup map: measurement_id -> replay result
        replay_map = {
            r["uuid"]: r for r in replay_output.get("results", [])
        }

        # Create set of buffered measurement IDs for quick lookup
        buffered_ids = {m.measurement_id for m in buffer}

        # Update original results with replay data
        updated_results = []
        for original in original_results:
            # Check if this measurement was in the buffer and has replay data
            if original.measurement_id in buffered_ids and original.measurement_id in replay_map:
                replay_data = replay_map[original.measurement_id]

                # Create updated result with replay data
                # Use model_dump() and model_validate() to properly create new instance
                updated_dict = original.model_dump()
                updated_dict.update({
                    "accepted": replay_data.get("accepted", original.accepted),
                    "quality_score": replay_data.get("quality_score", original.quality_score),
                    "kalman_estimate": replay_data.get("kalman_estimate", original.kalman_estimate),
                    # Note: replay service doesn't return these fields, so keep originals
                    # kalman_uncertainty, rejection_reason, processing_stage, reset_triggered, quality_components
                })

                updated_result = MeasurementResult.model_validate(updated_dict)
                updated_results.append(updated_result)

                logger.debug(
                    f"Updated result for measurement {original.measurement_id}: "
                    f"accepted={updated_result.accepted}, quality_score={updated_result.quality_score}"
                )
            else:
                # Keep original result unchanged
                updated_results.append(original)

        return updated_results

"""Service layer for weight processing operations."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from aws.api.models import (CleanupResponseData, HistoricalConflictDetails,
                                HistoricalConflictResponse, Measurement,
                                MeasurementResult, ProcessResponseData,
                                StateInfo)
from aws.config.config_manager import ConfigManager
from aws.services.replay_service import replay_measurements
from weight_processor_lib.core.database.base import StateStore
from weight_processor_lib.core.processing.processor import process_measurement

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
            from weight_processor_lib.core.database import get_state_db

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
            # Check if replay should be triggered BEFORE processing current measurement
            if buffered_replay_enabled and buffer:
                buffer_hours = self.config.get("replay", {}).get("buffer_hours", 24)

                # Check if current measurement is outside the time window from the last buffered measurement
                last_buffered_time = buffer[-1].measured_at
                time_gap_hours = (measurement.measured_at - last_buffered_time).total_seconds() / 3600

                # If time gap exceeds buffer window
                if time_gap_hours >= buffer_hours:
                    # Trigger replay if we have enough measurements
                    if len(buffer) >= 2:
                        logger.info(
                            f"Triggering replay for user {user_id}: trigger=time_gap, "
                            f"buffer_size={len(buffer)}, time_gap={time_gap_hours:.1f}h, "
                            f"buffer_range={buffer[0].measured_at} to {buffer[-1].measured_at}"
                        )

                        # Execute replay
                        replay_output = self._execute_buffered_replay(
                            user_id, buffer, buffer_start_time, user_height_m
                        )

                        # Merge replay results into original results
                        results = self._merge_replay_results(results, replay_output, buffer)

                        # Track replay metadata
                        replay_metadata.append({
                            "trigger": "time_gap",
                            "buffer_size": len(buffer),
                            "replay_from": buffer_start_time.isoformat(),
                            "replay_to": buffer[-1].measured_at.isoformat(),
                            "measurements_replayed": len(buffer),
                            "duration_seconds": replay_output.get("duration_seconds", 0),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                    else:
                        logger.info(
                            f"Time gap {time_gap_hours:.1f}h exceeds buffer window but only {len(buffer)} measurement(s) in buffer - no replay"
                        )

                    # Clear buffer for next window (regardless of whether replay triggered)
                    buffer.clear()
                    buffer_start_time = None

            try:
                result = self._process_single(user_id, measurement, user_height_m)
                results.append(result)

                if result.accepted:
                    accepted_count += 1
                else:
                    rejected_count += 1

                # Buffer management: Add ALL measurements to buffer (accepted or rejected)
                # This allows replays to reconsider rejected measurements with better context
                if buffered_replay_enabled:
                    # Create snapshot before first buffered measurement in the window
                    if not buffer:
                        buffer_start_time = measurement.measured_at
                        self.state_store.save_state_snapshot(user_id, buffer_start_time)
                        logger.info(f"Created snapshot for user {user_id} at {buffer_start_time}")

                    # Add measurement to buffer (both accepted and rejected)
                    buffer.append(measurement)

            except Exception as e:
                logger.error(f"Error processing {measurement.measurement_id}: {e}")
                results.append(
                    MeasurementResult(
                        measurement_id=measurement.measurement_id,
                        accepted=False,
                        value=measurement.weight_value,
                        unit=measurement.weight_unit,
                        effective_date_time=measurement.measured_at,
                        source_type=measurement.source,
                        rejection_reason=str(e),
                        processing_stage="processing",
                    )
                )
                rejected_count += 1

            # Check if replay should be triggered at batch end
            if buffered_replay_enabled:
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
                        value=measurement.weight_value,
                        unit=measurement.weight_unit,
                        effective_date_time=measurement.measured_at,
                        source_type=measurement.source,
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
        # Extract reset event if present
        reset_event_data = result.get("reset_event")
        reset_event = None
        if reset_event_data:
            from aws.api.models import ResetEvent
            reset_event = ResetEvent(
                type=reset_event_data.get("type", "unknown"),
                gap_days=reset_event_data.get("gap_days"),
                reason=reset_event_data.get("reason", "unknown"),
            )

        return MeasurementResult(
            measurement_id=measurement.measurement_id,
            accepted=result.get("accepted", False),
            value=measurement.weight_value,
            unit=measurement.weight_unit,
            effective_date_time=measurement.measured_at,
            source_type=measurement.source,
            quality_score=result.get("quality_score"),
            kalman_estimate=result.get("kalman_estimate"),
            kalman_uncertainty=result.get("kalman_uncertainty"),
            rejection_reason=result.get("reason"),
            processing_stage=result.get("stage"),
            reset_triggered=result.get("was_reset", False) or result.get("reset_event") is not None,
            reset_event=reset_event,
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
                conflicting_measurement_ids=conflicting,
            ),
        )

    def _should_trigger_replay(
        self, buffer: List[Measurement], current_timestamp: datetime, is_last: bool
    ) -> bool:
        """
        Determine if replay should be triggered for the current buffer.

        Replay is triggered when:
        1. Last measurement in batch (is_last=True) AND buffer has >= 2 measurements
        2. Time window exceeded (buffer_hours) AND buffer has >= 2 measurements
        3. Buffer size limit reached AND buffer has >= 2 measurements

        Args:
            buffer: List of buffered measurements
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

        # Trigger 2: Time window exceeded
        buffer_hours = self.config.get("replay", {}).get("buffer_hours", 24)
        first_timestamp = buffer[0].measured_at
        hours_elapsed = (current_timestamp - first_timestamp).total_seconds() / 3600

        if hours_elapsed >= buffer_hours:
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
            logger.info(
                f"Executing buffered replay for user {user_id}: "
                f"buffer_size={len(buffer)}, replay_from={buffer_start_time}"
            )

            # Track replay performance
            replay_start = time.time()

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

                # Update with replay results - use replay data for all processing fields
                updated_dict.update({
                    "accepted": replay_data.get("accepted", original.accepted),
                    "quality_score": replay_data.get("quality_score", original.quality_score),
                    "kalman_estimate": replay_data.get("kalman_estimate", original.kalman_estimate),
                    "rejection_reason": replay_data.get("rejection_reason", original.rejection_reason),
                    "processing_stage": replay_data.get("processing_stage", original.processing_stage),
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


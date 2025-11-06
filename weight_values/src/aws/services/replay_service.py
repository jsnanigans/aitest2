"""Simple replay service for MVP."""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from src.aws.api.models import Measurement
from src.core.processing.processor import process_measurement

logger = logging.getLogger(__name__)


def replay_measurements(
    user_id: str,
    measurements: List[Measurement],
    replay_from: datetime,
    state_store,
    config: Dict[str, Any],
    user_height_m: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Simple replay: restore state and reprocess measurements.

    Args:
        user_id: User identifier
        measurements: All measurements to replay
        replay_from: Timestamp to replay from
        state_store: Database instance
        config: Configuration

    Returns:
        Result dictionary with processing results
    """
    try:
        # Step 1: Get snapshot before replay_from
        snapshot = state_store.get_snapshot(user_id, replay_from)

        if snapshot:
            # Restore from snapshot
            state_store.save_state(user_id, snapshot)
            logger.info(f"Restored state from snapshot at {replay_from}")
        else:
            # No snapshot - reset state
            state_store.delete_state(user_id)
            logger.info(f"No snapshot found, starting fresh")

        # Step 2: Filter and sort measurements
        # The snapshot is AT the first buffered measurement's timestamp,
        # so we need to replay measurements AFTER the snapshot (not including it)
        replay_measurements = [
            m for m in measurements if m.measured_at > replay_from
        ]
        replay_measurements.sort(key=lambda m: m.measured_at)

        # If no measurements to replay (snapshot was after all buffered measurements),
        # include measurements at exactly replay_from timestamp
        if not replay_measurements:
            replay_measurements = [
                m for m in measurements if m.measured_at >= replay_from
            ]
            replay_measurements.sort(key=lambda m: m.measured_at)

        # Step 3: Process measurements with multi-pass approach
        # Pass 1: Score all measurements against the snapshot state
        baseline_state = state_store.get_state(user_id)
        measurement_scores = []

        logger.info(f"Pass 1: Scoring {len(replay_measurements)} measurements against snapshot state")
        print(f"Pass 1: Scoring {len(replay_measurements)} measurements against snapshot state")

        for measurement in replay_measurements:
            # Temporarily restore snapshot for each measurement
            state_store.save_state(user_id, snapshot if snapshot else {})

            result = process_measurement(
                user_id=user_id,
                weight=measurement.weight_value,
                timestamp=measurement.measured_at,
                source=measurement.source,
                unit=measurement.weight_unit,
                config=config,
                db=state_store,
                user_height_m=user_height_m,
            )

            quality_score = result.get("quality_score", 0.0)
            logger.info(f"  {measurement.weight_value} kg -> quality_score={quality_score:.4f}")
            print(f"  {measurement.weight_value} kg -> quality_score={quality_score:.4f}")

            measurement_scores.append({
                "measurement": measurement,
                "quality_score": quality_score,
                "result": result
            })

        # Pass 2: Process in order of quality score (best first)
        # This ensures we accept the best measurements first
        measurement_scores.sort(key=lambda x: x["quality_score"], reverse=True)

        # Restore snapshot for final processing
        state_store.save_state(user_id, snapshot if snapshot else {})

        results = []
        accepted_count = 0
        rejected_count = 0

        # Process in quality order, then re-sort results by timestamp
        for item in measurement_scores:
            measurement = item["measurement"]
            result = process_measurement(
                user_id=user_id,
                weight=measurement.weight_value,
                timestamp=measurement.measured_at,
                source=measurement.source,
                unit=measurement.weight_unit,
                config=config,
                db=state_store,
                user_height_m=user_height_m,
            )

            results.append(
                {
                    "uuid": measurement.measurement_id,
                    "accepted": result.get("accepted", False),
                    "quality_score": result.get("quality_score"),
                    "kalman_estimate": result.get("kalman_estimate"),
                    "timestamp": measurement.measured_at,
                }
            )

            if result.get("accepted"):
                accepted_count += 1
            else:
                rejected_count += 1

        # Sort results back to chronological order
        results.sort(key=lambda x: x["timestamp"])

        # Step 4: Create snapshot after replay
        state_store.save_state_snapshot(user_id, datetime.now(timezone.utc))

        return {
            "success": True,
            "processed_count": len(replay_measurements),
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "results": results,
        }

    except Exception as e:
        logger.exception(f"Replay failed for user {user_id}")
        return {"success": False, "error": str(e)}

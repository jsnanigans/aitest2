"""Simple replay service for MVP."""

import logging
from datetime import datetime
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
        replay_measurements = [
            m for m in measurements if m.measured_at >= replay_from
        ]
        replay_measurements.sort(key=lambda m: m.measured_at)

        # Step 3: Process measurements
        results = []
        accepted_count = 0
        rejected_count = 0

        for measurement in replay_measurements:
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
                }
            )

            if result.get("accepted"):
                accepted_count += 1
            else:
                rejected_count += 1

        # Step 4: Create snapshot after replay
        state_store.save_state_snapshot(user_id, datetime.utcnow())

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

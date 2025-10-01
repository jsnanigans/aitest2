"""
Simplified Replay Processor - Temporal consistency filtering only.

Handles the complete replay workflow:
1. Temporal consistency analysis (single filter)
2. State restoration and reprocessing
3. Metrics tracking
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

from .temporal_consistency_analyzer import TemporalConsistencyAnalyzer
from .replay_manager import ReplayManager

logger = logging.getLogger(__name__)


class ReplayProcessor:
    """
    Orchestrates replay processing with temporal consistency filtering.
    Simplified from previous multi-method approach.
    """

    def __init__(self, db, config: Optional[Dict[str, Any]] = None):
        """
        Initialize replay processor.

        Args:
            db: Database instance
            config: Replay configuration
        """
        self.db = db
        self.config = config or {}

        # Initialize components
        self.analyzer = TemporalConsistencyAnalyzer(
            config.get("temporal_consistency", {})
        )
        self.replay_manager = ReplayManager(db, config.get("safety", {}))

        # Metrics tracking
        self.metrics = {
            "buffers_processed": 0,
            "measurements_analyzed": 0,
            "outliers_found": 0,
            "corrections_made": 0,
            "replay_time_total": 0.0,
            "concurrent_replay_prevented": 0,
        }

    def process_buffer(
        self,
        user_id: str,
        buffered_measurements: List[Dict[str, Any]],
        buffer_start_time: datetime,
    ) -> Dict[str, Any]:
        """
        Process a buffer of measurements with temporal consistency analysis.

        Args:
            user_id: User identifier
            buffered_measurements: List of measurements from buffer
            buffer_start_time: Start time of the buffer window

        Returns:
            Processing result dictionary
        """
        start_time = time.time()

        try:
            # Update metrics
            self.metrics["buffers_processed"] += 1
            self.metrics["measurements_analyzed"] += len(buffered_measurements)

            # Step 1: Analyze measurements for temporal consistency
            logger.info(
                f"Analyzing {len(buffered_measurements)} measurements for {user_id}"
            )

            analysis = self.analyzer.analyze_and_filter(buffered_measurements, user_id)

            if not analysis["success"]:
                return {
                    "success": False,
                    "error": f"Analysis failed: {analysis.get('error')}",
                    "user_id": user_id,
                }

            # Update metrics
            outliers_found = analysis["outliers_found"]
            self.metrics["outliers_found"] += outliers_found

            # Step 2: If outliers found, replay clean measurements
            if outliers_found > 0:
                clean_measurements = analysis["clean_measurements"]
                logger.info(
                    f"Found {outliers_found} outliers for {user_id}, "
                    f"replaying {len(clean_measurements)} clean measurements"
                )

                replay_result = self.replay_manager.replay_clean_measurements(
                    user_id=user_id,
                    clean_measurements=clean_measurements,
                    buffer_start_time=buffer_start_time,
                )

                # Check if concurrent replay was prevented
                if replay_result.get("reason") == "concurrent_replay_prevented":
                    self.metrics["concurrent_replay_prevented"] += 1

                if replay_result["success"]:
                    self.metrics["corrections_made"] += 1

                # Add analysis details
                replay_result["analysis"] = {
                    "total_measurements": analysis["total_measurements"],
                    "outliers_found": outliers_found,
                    "clean_measurements": len(clean_measurements),
                    "statistics": self.analyzer.get_statistics(analysis),
                }

                # Track processing time
                processing_time = time.time() - start_time
                self.metrics["replay_time_total"] += processing_time
                replay_result["processing_time"] = processing_time

                return replay_result
            else:
                logger.info(f"No outliers found for {user_id}, skipping replay")

                return {
                    "success": True,
                    "user_id": user_id,
                    "message": "No temporal outliers found, replay not needed",
                    "analysis": {
                        "total_measurements": analysis["total_measurements"],
                        "outliers_found": 0,
                        "clean_measurements": len(buffered_measurements),
                    },
                    "processing_time": time.time() - start_time,
                }

        except Exception as e:
            logger.error(f"Error processing buffer for {user_id}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id,
                "processing_time": time.time() - start_time,
            }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get replay processing metrics.

        Returns:
            Metrics dictionary
        """
        metrics = self.metrics.copy()

        # Calculate derived metrics
        if metrics["buffers_processed"] > 0:
            metrics["avg_processing_time"] = (
                metrics["replay_time_total"] / metrics["buffers_processed"]
            )
            metrics["correction_rate"] = (
                metrics["corrections_made"] / metrics["buffers_processed"]
            )
        else:
            metrics["avg_processing_time"] = 0.0
            metrics["correction_rate"] = 0.0

        if metrics["measurements_analyzed"] > 0:
            metrics["outlier_rate"] = (
                metrics["outliers_found"] / metrics["measurements_analyzed"]
            )
        else:
            metrics["outlier_rate"] = 0.0

        return metrics

    def reset_metrics(self):
        """Reset all metrics to zero."""
        self.metrics = {
            "buffers_processed": 0,
            "measurements_analyzed": 0,
            "outliers_found": 0,
            "corrections_made": 0,
            "replay_time_total": 0.0,
            "concurrent_replay_prevented": 0,
        }

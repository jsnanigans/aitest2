#!/usr/bin/env python3
"""
Hyper-Speed Local Weight Stream Processor

Similar to api_main.py but uses direct method calls instead of API
and in-memory storage instead of database for maximum performance.
Processes weight measurements from CSV data and outputs a filtered CSV
with only accepted (non-rejected) measurements.

Implements real-time replay processing that triggers during measurement
processing when anomalies are detected within the replay window.
"""

import argparse
import csv
import json
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add weight_values/src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "weight_values" / "src"))

from core.database.database import ProcessorStateDB
from core.processing.processor import process_measurement
from core.replay.replay_manager import ReplayManager
from core.replay.replay_buffer import ReplayBuffer
from core.processing.outlier_detection import OutlierDetector
from core.processing.buffer_factory import get_factory


@dataclass
class ProcessingResult:
    """Result of processing measurements."""
    measurements_processed: int
    measurements_accepted: int
    measurements_rejected: int
    replays_triggered: int
    errors: List[str]

    @property
    def is_success(self) -> bool:
        return len(self.errors) == 0


class InMemoryProcessor:
    """Direct processor using in-memory storage for hyper-speed processing with real-time replay."""

    def __init__(self, config_path: Optional[str] = None, replay_config_override: Optional[Dict[str, Any]] = None):
        self.db = ProcessorStateDB()  # Already in-memory
        self.data_config = {}  # Initialize data_config

        # Load configuration from TOML file if provided, otherwise use defaults
        if config_path and Path(config_path).exists():
            self.config = self._load_config_from_toml(config_path)
        else:
            self.config = self._get_default_config()

        # Extract replay config from loaded config or use override
        if replay_config_override:
            replay_config = replay_config_override
        else:
            replay_config = self.config.get("replay", {})

        # Initialize replay components if enabled
        self.replay_enabled = replay_config and replay_config.get("enabled", False)
        self.replay_buffer = None
        self.outlier_detector = None
        self.replay_manager = None
        self.buffer_factory = None

        if self.replay_enabled:
            # Use BufferFactory for better testability and instance management
            self.buffer_factory = get_factory()
            self.buffer_factory.set_default_config(replay_config)
            self.replay_buffer = self.buffer_factory.create_buffer("default", replay_config)

            self.outlier_detector = OutlierDetector(
                replay_config.get("outlier_detection", {}), db=self.db
            )
            self.replay_manager = ReplayManager(self.db, replay_config.get("safety", {}))

            self.replay_window_hours = replay_config.get("buffer_hours", 72)

    def _load_config_from_toml(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from TOML file and convert to expected format."""
        with open(config_path, "rb") as f:
            toml_config = tomllib.load(f)

        # Convert TOML config to the format expected by the processor
        config = {
            "kalman": {
                "process_noise": toml_config.get("kalman", {}).get("process_noise", 0.01),
                "measurement_noise": toml_config.get("kalman", {}).get("observation_noise", 1.0),
                "initial_uncertainty": toml_config.get("kalman", {}).get("initial_covariance", 10.0),
                "adaptation_enabled": toml_config.get("kalman", {}).get("adaptation_enabled", True),
                "reset_window_hours": toml_config.get("kalman", {}).get("reset_window_hours", 720),
                # Additional parameters from TOML
                "initial_variance": toml_config.get("kalman", {}).get("initial_variance", 0.364),
                "transition_covariance_weight": toml_config.get("kalman", {}).get("transition_covariance_weight", 0.018),
                "transition_covariance_trend": toml_config.get("kalman", {}).get("transition_covariance_trend", 0.00015),
                "observation_covariance": toml_config.get("kalman", {}).get("observation_covariance", 3.4),
            },
            "quality_scoring": {
                "enabled": toml_config.get("quality_scoring", {}).get("enabled", True),
                "use_harmonic_mean": toml_config.get("quality_scoring", {}).get("use_harmonic_mean", True),
                "acceptance_threshold": toml_config.get("quality_scoring", {}).get("threshold", 0.5),
                # Component weights
                "plausibility_weight": toml_config.get("quality_scoring", {}).get("component_weights", {}).get("physiological", 0.15),
                "temporal_weight": toml_config.get("quality_scoring", {}).get("component_weights", {}).get("temporal_consistency", 0.20),
                "statistical_weight": toml_config.get("quality_scoring", {}).get("component_weights", {}).get("statistical", 0.10),
                "source_weight": toml_config.get("quality_scoring", {}).get("component_weights", {}).get("source_reliability", 0.20),
                "kalman_weight": toml_config.get("quality_scoring", {}).get("component_weights", {}).get("kalman_fit", 0.25),
                "frequency_weight": toml_config.get("quality_scoring", {}).get("component_weights", {}).get("frequency", 0.10),
                # Thresholds
                "thresholds": toml_config.get("quality_scoring", {}).get("thresholds", {}),
                # Temporal parameters
                "temporal": toml_config.get("quality_scoring", {}).get("temporal", {}),
                # Trend alignment
                "trend_alignment": toml_config.get("quality_scoring", {}).get("trend_alignment", {}),
            },
            "processing": {
                "extreme_threshold": toml_config.get("processing", {}).get("extreme_threshold", 0.15),
                "max_daily_change": toml_config.get("processing", {}).get("max_daily_change", 2.0),
                "min_weight": toml_config.get("processing", {}).get("min_weight", 20.0),
                "max_weight": toml_config.get("processing", {}).get("max_weight", 500.0),
            },
            "replay": toml_config.get("replay", {
                "enabled": True,
                "buffer_hours": 72,
                "trigger_mode": "time_based"
            }),
            # Include reset configurations
            "kalman.reset.initial": toml_config.get("kalman", {}).get("reset", {}).get("initial", {}),
            "kalman.reset.hard": toml_config.get("kalman", {}).get("reset", {}).get("hard", {}),
            "kalman.reset.soft": toml_config.get("kalman", {}).get("reset", {}).get("soft", {}),
            # Include other sections
            "adaptive_ranges": toml_config.get("adaptive_ranges", {}),
            "adaptive_noise": toml_config.get("adaptive_noise", {}),
            "circuit_breaker": toml_config.get("circuit_breaker", {}),
            "database": toml_config.get("database", {}),
            "visualization": toml_config.get("visualization", {}),
            "analysis": toml_config.get("analysis", {}),
            "logging": toml_config.get("logging", {}),
            "service": toml_config.get("service", {}),
        }

        # Store the data configuration for later use
        self.data_config = toml_config.get("data", {})

        return config

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default processing configuration."""
        return {
            "kalman": {
                "process_noise": 0.01,
                "measurement_noise": 1.0,
                "initial_uncertainty": 10.0
            },
            "quality_scoring": {
                "plausibility_weight": 0.3,
                "temporal_weight": 0.3,
                "statistical_weight": 0.2,
                "source_weight": 0.2,
                "acceptance_threshold": 0.3
            },
            "processing": {
                "extreme_threshold": 0.15
            },
            "replay": {
                "enabled": True,
                "buffer_hours": 72,
                "trigger_mode": "time_based"
            }
        }

    def process_measurements_chronologically(
        self,
        user_id: str,
        measurements: List[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None,
        acceptance_tracker: Optional['AcceptanceTracker'] = None
    ) -> ProcessingResult:
        """
        Process weight measurements for a user in chronological order with real-time replay.

        This simulates how measurements would be processed in production:
        1. Each measurement is processed one by one in time order
        2. Each measurement is added to a replay buffer
        3. When the buffer detects anomalies or is full, replay is triggered immediately
        4. After replay, the buffer is cleared and processing continues

        The key insight: We don't write to CSV during processing. Instead, we track
        acceptance in memory and write the CSV at the end. This allows replay to
        properly update acceptance status.

        Args:
            user_id: User identifier
            measurements: List of measurements to process
            options: Optional processing options
            acceptance_tracker: Optional tracker to update with final acceptance

        Returns:
            ProcessingResult with summary of processing including replay triggers
        """
        result = ProcessingResult(
            measurements_processed=0,
            measurements_accepted=0,
            measurements_rejected=0,
            replays_triggered=0,
            errors=[]
        )

        # Merge options into config
        config = self.config.copy()
        if options:
            config.update(options)

        # Sort measurements chronologically - CRITICAL for realistic simulation
        sorted_measurements = sorted(
            measurements,
            key=lambda m: self._parse_timestamp(m["effectiveDateTime"])
        )

        # Track acceptance status for each measurement by index
        # This is more reliable than timestamp matching
        acceptance_status = {}  # measurement index -> accepted bool

        # Process each measurement in chronological order
        for idx, measurement in enumerate(sorted_measurements):
            try:
                timestamp = self._parse_timestamp(measurement["effectiveDateTime"])

                # Process single measurement
                process_result = process_measurement(
                    user_id=user_id,
                    weight=measurement["weight"],
                    timestamp=timestamp,
                    source=measurement.get("source", "unknown"),
                    config=config,
                    unit=measurement.get("unit", "kg"),
                    db=self.db
                )

                result.measurements_processed += 1

                # Track initial acceptance status
                accepted = process_result.get("accepted", False)
                acceptance_status[idx] = accepted

                if accepted:
                    result.measurements_accepted += 1
                else:
                    result.measurements_rejected += 1

                # Store quality score for later use
                measurement["_quality_score"] = process_result.get("quality_score", 0.0)
                measurement["_idx"] = idx  # Track index for replay updates

                # Add to replay buffer if enabled (simulating real-time processing)
                if self.replay_enabled and self.replay_buffer:
                    measurement_data = {
                        "weight": measurement["weight"],
                        "timestamp": timestamp,
                        "source": measurement.get("source", "unknown"),
                        "unit": measurement.get("unit", "kg"),
                        "metadata": {
                            "accepted": accepted,
                            "rejection_reason": process_result.get("rejection_reason", None),
                            "quality_score": process_result.get("quality_score", None),
                            "quality_components": process_result.get("quality_components", None),
                            "measurement_idx": idx  # Track index for updates
                        }
                    }

                    buffer_result = self.replay_buffer.add_measurement(user_id, measurement_data)

                    # Check if buffer is ready for replay processing
                    if buffer_result.get("buffer_ready", False):
                        # Save state snapshot before replay
                        self.db.save_state_snapshot(user_id, timestamp)

                        # Get measurements in the buffer before replay
                        buffered_measurements = self.replay_buffer.get_buffer_measurements(user_id)
                        buffered_indices = [m["metadata"].get("measurement_idx") for m in buffered_measurements if m.get("metadata")]

                        # Trigger replay immediately - this is the key difference!
                        replay_success = self._process_replay_buffer(
                            user_id=user_id,
                            buffer_timestamp=timestamp
                        )

                        if replay_success:
                            result.replays_triggered += 1

                            # After replay, reprocess buffered measurements to get corrected acceptance
                            # The replay has restored state, so now we need to re-evaluate
                            for buffer_idx in buffered_indices:
                                if buffer_idx is not None and 0 <= buffer_idx < len(sorted_measurements):
                                    m = sorted_measurements[buffer_idx]
                                    m_timestamp = self._parse_timestamp(m["effectiveDateTime"])

                                    # Re-evaluate after replay with restored state
                                    re_result = process_measurement(
                                        user_id=user_id,
                                        weight=m["weight"],
                                        timestamp=m_timestamp,
                                        source=m.get("source", "unknown"),
                                        config=config,
                                        unit=m.get("unit", "kg"),
                                        db=self.db
                                    )

                                    new_accepted = re_result.get("accepted", False)
                                    old_accepted = acceptance_status.get(buffer_idx, False)

                                    # Update acceptance status and counts
                                    if new_accepted != old_accepted:
                                        acceptance_status[buffer_idx] = new_accepted
                                        if new_accepted:
                                            result.measurements_accepted += 1
                                            result.measurements_rejected -= 1
                                        else:
                                            result.measurements_accepted -= 1
                                            result.measurements_rejected += 1

                                    # Update quality score
                                    m["_quality_score"] = re_result.get("quality_score", 0.0)

                        # Clear buffer after processing
                        self.replay_buffer.clear_buffer(user_id)

            except Exception as e:
                result.errors.append(f"Error processing measurement at {measurement.get('effectiveDateTime', 'unknown')}: {str(e)}")

        # Process any remaining buffers at the end
        if self.replay_enabled and self.replay_buffer:
            ready_buffers = self.replay_buffer.get_ready_buffers()
            for buffer_user_id in ready_buffers:
                if buffer_user_id == user_id:
                    # Process final buffer
                    self.db.save_state_snapshot(user_id, datetime.now(timezone.utc))
                    buffered_measurements = self.replay_buffer.get_buffer_measurements(user_id)
                    buffered_indices = [m["metadata"].get("measurement_idx") for m in buffered_measurements if m.get("metadata")]

                    if self._process_replay_buffer(user_id, datetime.now(timezone.utc)):
                        result.replays_triggered += 1

                        # Re-evaluate buffered measurements
                        for buffer_idx in buffered_indices:
                            if buffer_idx is not None and 0 <= buffer_idx < len(sorted_measurements):
                                m = sorted_measurements[buffer_idx]
                                m_timestamp = self._parse_timestamp(m["effectiveDateTime"])

                                re_result = process_measurement(
                                    user_id=user_id,
                                    weight=m["weight"],
                                    timestamp=m_timestamp,
                                    source=m.get("source", "unknown"),
                                    config=config,
                                    unit=m.get("unit", "kg"),
                                    db=self.db
                                )

                                new_accepted = re_result.get("accepted", False)
                                old_accepted = acceptance_status.get(buffer_idx, False)

                                if new_accepted != old_accepted:
                                    acceptance_status[buffer_idx] = new_accepted
                                    if new_accepted:
                                        result.measurements_accepted += 1
                                        result.measurements_rejected -= 1
                                    else:
                                        result.measurements_accepted -= 1
                                        result.measurements_rejected += 1

                                m["_quality_score"] = re_result.get("quality_score", 0.0)

                    self.replay_buffer.clear_buffer(user_id)

        # Update measurements with final acceptance status
        for idx, measurement in enumerate(sorted_measurements):
            measurement["_accepted"] = acceptance_status.get(idx, False)

        # Update acceptance tracker if provided
        if acceptance_tracker:
            acceptance_tracker.track_results(user_id, sorted_measurements)

        return result

    def _process_replay_buffer(self, user_id: str, buffer_timestamp: datetime) -> bool:
        """
        Process replay buffer when anomalies are detected.
        Uses the same logic as local_old.py's _process_replay_buffer function.

        Args:
            user_id: User identifier
            buffer_timestamp: Current timestamp when replay was triggered

        Returns:
            True if replay was successful
        """
        try:
            # Get buffered measurements
            buffered_measurements = self.replay_buffer.get_buffer_measurements(user_id)
            if not buffered_measurements:
                return False

            buffer_info = self.replay_buffer.get_buffer_info(user_id)
            buffer_start_time = buffer_info["first_timestamp"] if buffer_info else None

            # Try enhanced replay processor first if available
            try:
                from core.database import get_state_db
                from core.replay.replay_processor import ReplayProcessor

                replay_config = {
                    "analysis": {
                        "kalman_deviation_threshold": 0.10,
                        "temporal_change_threshold": 0.05,
                        "outlier_score_threshold": 0.4,
                        "reset_reevaluation_threshold": 0.6,
                    },
                    "safety": self.replay_manager.config if hasattr(self.replay_manager, "config") else {}
                }

                processor = ReplayProcessor(self.db, replay_config)
                result = processor.process_buffer(user_id, buffered_measurements, buffer_start_time)

                return result.get("success", False)

            except ImportError:
                # Fall back to basic replay logic
                # Detect outliers in the buffer
                clean_measurements, outlier_indices = self.outlier_detector.get_clean_measurements(
                    buffered_measurements, user_id=user_id
                )

                # If we found outliers, replay from buffer start
                if len(outlier_indices) > 0 and clean_measurements and buffer_start_time:
                    # Restore to snapshot before buffer
                    self.db.restore_state_snapshot(user_id)

                    # Replay clean measurements
                    replay_result = self.replay_manager.replay_clean_measurements(
                        user_id=user_id,
                        clean_measurements=clean_measurements,
                        buffer_start_time=buffer_start_time
                    )

                    return replay_result.get("success", False)

                return False

        except Exception as e:
            print(f"Error in replay processing for {user_id}: {e}")
            return False

    def get_user_state(self, user_id: str) -> Dict[str, Any]:
        """Get user processing state."""
        state = self.db.get_state(user_id)
        if state:
            return {"success": True, "state": state}
        return {"success": False, "error": "User not found"}

    def cleanup_user(self, user_id: str, cleanup_type: str = "reset_adaptive") -> Dict[str, Any]:
        """Cleanup/reset user state."""
        if cleanup_type == "reset_adaptive":
            # Delete state to trigger fresh start
            deleted = self.db.delete_state(user_id)
            if self.replay_buffer and deleted:
                self.replay_buffer.clear_buffer(user_id)
            return {"success": deleted}
        return {"success": False, "error": f"Unknown cleanup type: {cleanup_type}"}

    def cleanup(self):
        """Clean up resources."""
        if self.buffer_factory:
            try:
                buffer_stats = self.buffer_factory.get_stats()
                if buffer_stats["total_instances"] > 0:
                    self.buffer_factory.clear_all(force=True)
            except:
                pass

    def _parse_timestamp(self, date_str: str) -> datetime:
        """Parse various timestamp formats and ensure timezone-aware datetimes."""
        if not date_str:
            return datetime.now(timezone.utc)

        try:
            if "T" in date_str:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            elif " " in date_str:
                dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            else:
                dt = datetime.strptime(date_str, "%Y-%m-%d")

            # Ensure timezone-aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            return dt
        except Exception:
            return datetime.now(timezone.utc)


class AcceptanceTracker:
    """Tracks which measurements were accepted during processing."""

    def __init__(self):
        self.accepted_measurements = {}  # (user_id, timestamp) -> quality_score
        self.user_acceptance_details = {}   # user_id -> list of acceptance info

    def track_results(self, user_id: str, measurements: List[Dict[str, Any]]):
        """Track acceptance results from processed measurements."""
        if user_id not in self.user_acceptance_details:
            self.user_acceptance_details[user_id] = []

        for measurement in measurements:
            timestamp = measurement["effectiveDateTime"]
            normalized_timestamp = parse_timestamp(timestamp).isoformat()

            if measurement.get("_accepted", False):
                quality_score = measurement.get("_quality_score", 0.0)
                self.accepted_measurements[(user_id, normalized_timestamp)] = quality_score

                self.user_acceptance_details[user_id].append({
                    "timestamp": timestamp,
                    "accepted": True,
                    "quality_score": quality_score
                })

    def is_accepted(self, user_id: str, timestamp: str) -> bool:
        """Check if a measurement was accepted."""
        normalized_timestamp = parse_timestamp(timestamp).isoformat()
        return (user_id, normalized_timestamp) in self.accepted_measurements

    def get_quality_score(self, user_id: str, timestamp: str) -> float:
        """Get quality score for an accepted measurement."""
        normalized_timestamp = parse_timestamp(timestamp).isoformat()
        return self.accepted_measurements.get((user_id, normalized_timestamp), 0.0)


def parse_timestamp(date_str: str) -> datetime:
    """Parse various timestamp formats and ensure timezone-aware datetimes."""
    if not date_str:
        return datetime.now(timezone.utc)

    try:
        if "T" in date_str:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        elif " " in date_str:
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        else:
            dt = datetime.strptime(date_str, "%Y-%m-%d")

        # Ensure all datetimes are timezone-aware (assume UTC if missing)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt
    except Exception:
        return datetime.now(timezone.utc)


def load_csv_data(csv_path: str, max_users: int = 0, max_rows: int = 0) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Load CSV data and group measurements by user_id.

    Args:
        csv_path: Path to CSV file
        max_users: Maximum number of users to process (0 for no limit)
        max_rows: Maximum number of rows to read (0 for no limit)

    Returns:
        Tuple of (user_measurements dict, original_rows list)
    """
    user_measurements = {}
    original_rows = []
    selected_users = set()  # Track selected users for early stopping

    print(f"Loading data from {csv_path}...")
    if max_users > 0:
        print(f"Will stop after loading {max_users} users")

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        row_count = 0

        for row in reader:
            row_count += 1
            if max_rows > 0 and row_count > max_rows:
                break

            # Handle both old and new column names for ID
            measurement_id = row.get("id") or row.get("measurement_id")
            user_id = row.get("user_id")
            if not user_id or not measurement_id:
                continue

            # Early stopping when we have enough users
            if max_users > 0:
                if user_id not in selected_users:
                    if len(selected_users) >= max_users:
                        # Skip rows from new users once we have enough
                        continue
                    selected_users.add(user_id)
                # Always process rows from already selected users

            # Parse and validate weight - handle both old and new column names
            weight_str = row.get("value_quantity", "") or row.get("weight", "")
            weight_str = weight_str.strip()
            if not weight_str or weight_str.upper() == "NULL":
                continue

            try:
                weight = float(weight_str)
                if weight <= 0 or weight > 1000:
                    continue
            except (ValueError, TypeError):
                continue

            # Parse other fields - handle both old and new column names
            date_str = row.get("effective_date_time", "") or row.get("effectiveDateTime", "")
            source = row.get("source_type", "unknown")
            unit = row.get("unit", "kg")

            # Store original row with unique identifier for tracking
            original_row = row.copy()
            original_row["_row_index"] = row_count
            original_rows.append(original_row)

            # Convert to measurement format for processing
            measurement = {
                "uuid": measurement_id,
                "weight": weight,
                "unit": unit,
                "effectiveDateTime": parse_timestamp(date_str).isoformat() if date_str else datetime.now(timezone.utc).isoformat(),
                "source": source,
                "metadata": {
                    "original_row_index": row_count,
                    "csv_row": original_row
                }
            }

            if user_id not in user_measurements:
                user_measurements[user_id] = []
            user_measurements[user_id].append(measurement)

            # Progress update
            if row_count % 10000 == 0:
                print(f"  Loaded {row_count:,} rows, {len(user_measurements):,} users...")
                if max_users > 0 and len(selected_users) >= max_users:
                    print(f"  Reached max users limit ({max_users}), continuing to load their remaining measurements...")

    print(f"Loaded {len(user_measurements):,} users with {sum(len(m) for m in user_measurements.values()):,} total measurements")

    return user_measurements, original_rows


def process_users_chronologically(
    processor: InMemoryProcessor,
    user_measurements: Dict[str, List[Dict[str, Any]]],
    acceptance_tracker: AcceptanceTracker
) -> Dict[str, Dict[str, Any]]:
    """
    Process measurements chronologically for each user with real-time replay.

    This simulates production behavior where measurements arrive in real-time
    and replay is triggered immediately when anomalies are detected.

    Args:
        processor: InMemoryProcessor instance
        user_measurements: Dict mapping user_id to measurements
        acceptance_tracker: Tracker for accepted measurements

    Returns:
        Dict mapping user_id to processing results
    """
    results = {}
    total_users = len(user_measurements)
    total_measurements = sum(len(measurements) for measurements in user_measurements.values())

    print(f"\nProcessing measurements chronologically with real-time replay...")
    print(f"Total users: {total_users:,}")
    print(f"Total measurements: {total_measurements:,}")
    if processor.replay_enabled:
        print(f"Replay window: {processor.replay_window_hours} hours")

    processed_measurements = 0
    successful_users = 0
    failed_users = 0
    total_replays = 0

    for i, (user_id, measurements) in enumerate(user_measurements.items(), 1):
        if i % 100 == 0:
            print(f"[{i}/{total_users}] Processing users...")

        # Process all measurements chronologically for this user
        # Pass acceptance_tracker to update it during processing
        result = processor.process_measurements_chronologically(
            user_id, measurements, acceptance_tracker=acceptance_tracker
        )

        user_results = {
            "measurements_processed": result.measurements_processed,
            "measurements_accepted": result.measurements_accepted,
            "measurements_rejected": result.measurements_rejected,
            "replays_triggered": result.replays_triggered,
            "errors": result.errors
        }

        processed_measurements += result.measurements_processed
        total_replays += result.replays_triggered

        results[user_id] = user_results

        if user_results["errors"]:
            failed_users += 1
        else:
            successful_users += 1

        # Progress update
        if i % 500 == 0 or i == total_users:
            print(f"  Progress: {i}/{total_users} users, {processed_measurements:,}/{total_measurements:,} measurements, {total_replays} replays triggered")

    print("\nProcessing complete:")
    print(f"  Successful users: {successful_users:,}")
    print(f"  Failed users: {failed_users:,}")
    print(f"  Total measurements processed: {processed_measurements:,}")
    print(f"  Total replay triggers: {total_replays:,}")

    return results


def write_filtered_csv(
    original_rows: List[Dict[str, Any]],
    acceptance_tracker: AcceptanceTracker,
    output_path: str
) -> int:
    """
    Write filtered CSV with only accepted measurements.
    Similar to local_old.py but reads from acceptance tracker instead of real-time writing.

    Args:
        original_rows: Original CSV rows with tracking info
        acceptance_tracker: Tracker containing acceptance information
        output_path: Path to write filtered CSV

    Returns:
        Number of accepted rows written
    """
    if not original_rows:
        print("No original rows to filter")
        return 0

    print(f"\nWriting filtered CSV to {output_path}...")

    # Get fieldnames from first row (excluding internal tracking fields)
    fieldnames = [k for k in original_rows[0].keys() if not k.startswith('_')]

    # Add quality_score column like local_old.py does
    if 'quality_score' not in fieldnames:
        fieldnames.append('quality_score')

    accepted_count = 0
    total_count = len(original_rows)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in original_rows:
            user_id = row.get("user_id")
            # Handle both old and new column names for timestamp
            timestamp = row.get("effective_date_time") or row.get("effectiveDateTime")

            if user_id and timestamp and acceptance_tracker.is_accepted(user_id, timestamp):
                # Create filtered row without internal tracking fields
                filtered_row = {k: v for k, v in row.items() if not k.startswith('_')}

                # Add quality score
                filtered_row['quality_score'] = acceptance_tracker.get_quality_score(user_id, timestamp)

                writer.writerow(filtered_row)
                accepted_count += 1

    print(f"Filtered CSV written: {accepted_count:,}/{total_count:,} measurements accepted ({accepted_count/total_count*100:.1f}%)")

    return accepted_count


def main():
    parser = argparse.ArgumentParser(description="Hyper-Speed Local Weight Stream Processor with Real-Time Replay")
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to configuration TOML file (default: local/config.toml)"
    )
    parser.add_argument(
        "--csv-file",
        help="CSV file to process (overrides config file)"
    )
    parser.add_argument(
        "--max-users",
        type=int,
        help="Maximum users to process (overrides config file)"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Maximum CSV rows to read (overrides config file)"
    )
    parser.add_argument(
        "--enable-replay",
        action="store_true",
        help="Enable real-time replay processing (overrides config file)"
    )
    parser.add_argument(
        "--disable-replay",
        action="store_true",
        help="Disable real-time replay processing (overrides config file)"
    )
    parser.add_argument(
        "--replay-window",
        type=int,
        help="Replay window in hours (overrides config file)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for results (overrides config file)"
    )
    parser.add_argument(
        "--filtered-csv",
        help="Output path for filtered CSV (default: output_dir/filtered_TIMESTAMP.csv)"
    )

    args = parser.parse_args()

    # Load configuration file first
    config_path = Path(args.config)
    if config_path.exists():
        processor = InMemoryProcessor(config_path=str(config_path))
        # Get data config from loaded TOML
        data_config = processor.data_config
        toml_replay_config = processor.config.get("replay", {})
    else:
        print(f"Warning: Config file not found: {config_path}, using defaults")
        processor = InMemoryProcessor()
        data_config = {}
        toml_replay_config = {}

    # Override config values with command-line arguments
    csv_file = args.csv_file or data_config.get("csv_file", "data/2025-09-29_weights_all.csv")
    max_users = args.max_users if args.max_users is not None else data_config.get("max_users", 0)
    max_rows = args.max_rows if args.max_rows is not None else data_config.get("max_rows", 0)
    output_dir = args.output_dir or data_config.get("output_dir", "output_local")

    # Handle replay configuration
    if args.disable_replay:
        replay_enabled = False
    elif args.enable_replay:
        replay_enabled = True
    else:
        replay_enabled = toml_replay_config.get("enabled", False)

    replay_window = args.replay_window or toml_replay_config.get("buffer_hours", 72)

    # Validate inputs
    if not Path(csv_file).exists():
        print(f"Error: CSV file not found: {csv_file}")
        return 1

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Configure replay if enabled
    if replay_enabled:
        replay_config = {
            "enabled": True,
            "buffer_hours": replay_window,
            "trigger_mode": toml_replay_config.get("trigger_mode", "time_based"),
            "outlier_detection": toml_replay_config.get("outlier_detection", {
                "mad_threshold": 3.0,
                "quality_threshold": 0.3
            }),
            "safety": toml_replay_config.get("safety", {
                "max_replay_attempts": 3,
                "min_measurements": 5
            })
        }
        # Re-initialize processor with updated replay config if needed
        if not processor.replay_enabled:
            processor = InMemoryProcessor(config_path=str(config_path) if config_path.exists() else None, replay_config_override=replay_config)
    else:
        replay_config = None

    print("=" * 60)
    print("HYPER-SPEED LOCAL PROCESSOR WITH REAL-TIME REPLAY")
    print("Using in-memory storage and direct method calls")
    if config_path.exists():
        print(f"Configuration loaded from: {config_path}")
    if replay_enabled:
        print(f"Real-time replay enabled with {replay_window}h window")
    else:
        print("Real-time replay disabled")
    print("=" * 60)

    # Load CSV data
    user_measurements, original_rows = load_csv_data(
        csv_file,
        max_users=max_users,
        max_rows=max_rows
    )

    if not user_measurements:
        print("No valid measurements found in CSV file")
        return 1

    # Initialize acceptance tracker
    acceptance_tracker = AcceptanceTracker()

    # Track overall results
    start_time = datetime.now()
    overall_results = {
        "start_time": start_time.isoformat(),
        "csv_file": csv_file,
        "config_file": str(config_path) if config_path.exists() else None,
        "mode": "hyper-speed-local-realtime",
        "replay_enabled": replay_enabled,
        "replay_window_hours": replay_window if replay_enabled else None,
        "users_loaded": len(user_measurements),
        "total_measurements": sum(len(m) for m in user_measurements.values()),
        "processing_results": None
    }

    # Process measurements chronologically with real-time replay
    processing_results = process_users_chronologically(
        processor,
        user_measurements,
        acceptance_tracker
    )
    overall_results["processing_results"] = processing_results

    # Write filtered CSV
    timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
    filtered_csv_path = args.filtered_csv or str(output_dir / f"filtered_{timestamp_str}.csv")
    accepted_count = write_filtered_csv(original_rows, acceptance_tracker, filtered_csv_path)

    # Finalize results
    end_time = datetime.now()
    overall_results["end_time"] = end_time.isoformat()
    overall_results["duration_seconds"] = (end_time - start_time).total_seconds()
    overall_results["filtered_csv_path"] = filtered_csv_path
    overall_results["accepted_measurements"] = accepted_count
    overall_results["total_original_measurements"] = len(original_rows)

    # Save results to file
    results_file = output_dir / f"local_processing_results_{timestamp_str}.json"

    with open(results_file, 'w') as f:
        json.dump(overall_results, f, indent=2, default=str)

    print("\n=== Processing Complete ===")
    print(f"Duration: {overall_results['duration_seconds']:.1f} seconds")
    if overall_results['duration_seconds'] > 0:
        print(f"Speed: {len(original_rows) / overall_results['duration_seconds']:.0f} rows/second")
    print(f"Results saved to: {results_file}")
    print(f"Filtered CSV saved to: {filtered_csv_path}")

    # Print summary statistics
    if overall_results["processing_results"]:
        stats = overall_results["processing_results"]
        total_processed = sum(r["measurements_processed"] for r in stats.values())
        total_accepted = sum(r["measurements_accepted"] for r in stats.values())
        total_replays = sum(r.get("replays_triggered", 0) for r in stats.values())
        print(f"Processing summary: {total_processed:,} processed, {total_accepted:,} accepted")
        if replay_enabled:
            print(f"Real-time replays triggered: {total_replays:,}")

    print(f"Filtered output: {accepted_count:,} accepted measurements written")

    # Cleanup
    processor.cleanup()

    return 0


if __name__ == "__main__":
    exit(main())

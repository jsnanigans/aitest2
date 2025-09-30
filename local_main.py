#!/usr/bin/env python3
"""
Local Weight Stream Processor

Processes weight measurements from CSV data using direct method calls instead of API.
Uses in-memory storage for state management.
Outputs a filtered CSV with only accepted (non-rejected) measurements.
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add weight_values to path for imports
sys.path.insert(0, str(Path(__file__).parent / "weight_values"))

from weight_values.src.aws.api.models import Measurement, ProcessResponseData
from weight_values.src.aws.services.weight_processor_service import WeightProcessorService
from weight_values.src.core.database.database import ProcessorStateDB
from weight_values.src.aws.config.config_manager import ConfigManager
from weight_values.src.core.constants import SUPPORTED_WEIGHT_UNITS


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration based on lambda.env.template.

    Returns:
        Configuration dictionary matching production settings
    """
    return {
        "database": {
            "backend": "memory",  # Override for local processing
            "table_name": "weight-processor-state",
            "region": "us-east-1",
        },
        "kalman": {
            "enabled": True,
            "adaptive": True,
            "process_noise": 0.1,
            "observation_noise": 1.0,
            "initial_covariance": 1.0,
            "adaptation": {
                "enabled": True,
                "initial_multiplier": 10.0,
                "decay_rate": 0.1,
            },
            "resets": {
                "enabled": True,
                "hard_gap_days": 30,
                "window_hours": 720,
                "soft_sources": ["questionnaire"],
            },
        },
        "quality_scoring": {
            "enabled": True,
            "weights": {
                "kalman": 0.25,
                "temporal": 0.20,
                "source": 0.20,
                "physiological": 0.15,
                "statistical": 0.10,
                "frequency": 0.10,
            },
            "thresholds": {
                "high": 0.8,
                "medium": 0.5,
                "outlier_override": 0.85,
                "acceptance": 0.3,
            },
        },
        "processing": {
            "extreme_threshold": 0.15,
            "max_daily_change_kg": 2.0,
            "min_weight_kg": 20.0,
            "max_weight_kg": 500.0,
        },
        "replay": {
            "enabled": True,
            "buffer_hours": 72,
            "trigger_mode": "time_based",
            "outlier_methods": ["iqr", "mad"],
            "iqr_multiplier": 1.5,
            "mad_threshold": 3.0,
            "max_attempts": 3,
            "min_measurements": 10,
            "rollback_on_error": True,
        },
        "outlier_detection": {
            "enabled": True,
            "iqr_multiplier": 1.5,
            "mad_threshold": 3.0,
        },
        "snapshot": {
            "periodic_enabled": True,
            "interval_hours": 24,
            "retention_days": 10,
        },
        "circuit_breaker": {
            "enabled": True,
            "failure_threshold": 5,
            "timeout_seconds": 60,
            "half_open_attempts": 3,
        },
        "logging": {
            "level": "INFO",
            "verbose": False,
        },
        "service": {
            "environment": "local",
            "version": "1.0.0",
        },
    }


def parse_timestamp(date_str: str) -> datetime:
    """Parse various timestamp formats and return timezone-aware datetime in UTC."""
    if not date_str:
        return datetime.now(timezone.utc)

    try:
        if "T" in date_str:
            # Parse ISO format
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            # Ensure it has UTC timezone
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        elif " " in date_str:
            # Parse as naive and add UTC timezone
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)
        else:
            # Parse date only and add UTC timezone
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc)
    except Exception:
        # Fallback to current time if parsing fails
        return datetime.now(timezone.utc)


def load_csv_data(csv_path: str, max_users: int = 0, max_rows: int = 0) -> Tuple[Dict[str, List[Measurement]], List[Dict[str, Any]]]:
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

    # Statistics for rejected data
    stats = {
        "total_rows": 0,
        "invalid_weight": 0,
        "parse_errors": 0,
        "unit_rejected": 0,
        "rejected_units": {},
        "bsa_measurements": 0,
        "missing_data": 0,
    }

    print(f"Loading data from {csv_path}...")

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        row_count = 0

        for row in reader:
            row_count += 1
            stats["total_rows"] += 1

            if max_rows > 0 and row_count > max_rows:
                break

            # Handle both old and new column names for ID
            measurement_id = row.get("id") or row.get("measurement_id")
            user_id = row.get("user_id")
            if not user_id or not measurement_id:
                stats["missing_data"] += 1
                continue

            # Parse and validate weight - handle both old and new column names
            weight_str = row.get("value_quantity", "") or row.get("weight", "")
            weight_str = weight_str.strip()
            if not weight_str or weight_str.upper() == "NULL":
                stats["missing_data"] += 1
                continue

            try:
                weight = float(weight_str)
                # Validate weight is reasonable (basic sanity check)
                if weight <= 0 or weight > 1000:
                    stats["invalid_weight"] += 1
                    continue
                # Check for NaN and Inf
                if math.isnan(weight) or math.isinf(weight):
                    stats["invalid_weight"] += 1
                    continue
            except (ValueError, TypeError):
                stats["parse_errors"] += 1
                continue

            # Parse other fields - handle both old and new column names
            date_str = row.get("effective_date_time", "") or row.get("effectiveDateTime", "")
            source = row.get("source_type", "unknown")
            unit = row.get("unit", "").strip()  # NO DEFAULT - must be explicit

            # Skip BSA measurements (Body Surface Area)
            if "BSA" in source.upper() or "m2" in unit or "m²" in unit:
                stats["bsa_measurements"] += 1
                continue

            # Early unit validation - check against whitelist
            if not unit:
                stats["unit_rejected"] += 1
                stats["rejected_units"]["<missing>"] = stats["rejected_units"].get("<missing>", 0) + 1
                continue

            unit_lower = unit.lower().strip()
            if unit_lower not in SUPPORTED_WEIGHT_UNITS:
                stats["unit_rejected"] += 1
                stats["rejected_units"][unit] = stats["rejected_units"].get(unit, 0) + 1
                continue

            # Store original row with unique identifier for tracking
            original_row = row.copy()
            original_row["_row_index"] = row_count
            original_row["_accepted"] = False  # Will be updated during processing
            original_rows.append(original_row)

            # Parse timestamp with error handling
            try:
                timestamp = parse_timestamp(date_str) if date_str else datetime.now(timezone.utc)
            except Exception:
                # Fallback to current time if parsing fails
                timestamp = datetime.now(timezone.utc)

            # Convert to Measurement model
            try:
                measurement = Measurement(
                    uuid=measurement_id,
                    weight=weight,
                    unit=unit,
                    effectiveDateTime=timestamp,
                    source=source,
                    metadata={
                        "original_row_index": row_count,
                        "csv_row": original_row
                    }
                )
            except Exception as e:
                stats["parse_errors"] += 1
                continue

            if user_id not in user_measurements:
                user_measurements[user_id] = []
            user_measurements[user_id].append(measurement)

            # Progress update
            if row_count % 10000 == 0:
                print(f"  Loaded {row_count:,} rows, {len(user_measurements):,} users...")

    # Apply user limit
    if max_users > 0 and len(user_measurements) > max_users:
        # Take first N users by sorted order for consistency
        sorted_users = sorted(user_measurements.keys())[:max_users]
        user_measurements = {uid: user_measurements[uid] for uid in sorted_users}

        # Filter original_rows to match selected users
        selected_user_set = set(sorted_users)
        original_rows = [row for row in original_rows if row.get("user_id") in selected_user_set]

    # Calculate valid measurements loaded
    total_measurements = sum(len(m) for m in user_measurements.values())

    print(f"Loaded {len(user_measurements):,} users with {total_measurements:,} total measurements")

    # Report data quality statistics
    if stats["total_rows"] > 0:
        print(f"\nData Quality Statistics:")
        print(f"  Total rows read: {stats['total_rows']:,}")
        print(f"  Valid measurements: {total_measurements:,}")

        rejected_total = (
            stats["invalid_weight"] +
            stats["parse_errors"] +
            stats["unit_rejected"] +
            stats["bsa_measurements"] +
            stats["missing_data"]
        )
        print(f"  Rejected measurements: {rejected_total:,}")

        if stats["invalid_weight"] > 0:
            print(f"    Invalid weight values: {stats['invalid_weight']:,}")
        if stats["parse_errors"] > 0:
            print(f"    Parse errors: {stats['parse_errors']:,}")
        if stats["unit_rejected"] > 0:
            print(f"    Invalid/unsupported units: {stats['unit_rejected']:,}")
        if stats["bsa_measurements"] > 0:
            print(f"    BSA measurements (filtered): {stats['bsa_measurements']:,}")
        if stats["missing_data"] > 0:
            print(f"    Missing required data: {stats['missing_data']:,}")

        # Report rejected units breakdown
        if stats["rejected_units"]:
            print(f"\n  Top rejected units:")
            for unit, count in sorted(stats["rejected_units"].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    '{unit}': {count:,} measurements")

    return user_measurements, original_rows


class AcceptanceTracker:
    """Tracks which measurements were accepted during processing."""

    def __init__(self):
        self.accepted_measurements = set()  # Track by (user_id, timestamp)
        self.user_acceptance_details = {}   # user_id -> list of acceptance info

    def clear(self):
        """Clear all tracked acceptances (used for replay)."""
        self.accepted_measurements.clear()
        self.user_acceptance_details.clear()

    def mark_measurement_accepted(self, user_id: str, timestamp: str, additional_info: Dict[str, Any] = None):
        """Mark a measurement as accepted."""
        self.accepted_measurements.add((user_id, timestamp))
        if user_id not in self.user_acceptance_details:
            self.user_acceptance_details[user_id] = []

        info = {"timestamp": timestamp, "accepted": True}
        if additional_info:
            info.update(additional_info)
        self.user_acceptance_details[user_id].append(info)

    def mark_batch_results(self, user_id: str, measurements: List[Measurement], response_data: ProcessResponseData):
        """Mark batch measurement results based on processing response."""
        # Extract results from response
        for i, result in enumerate(response_data.results):
            if result.accepted and i < len(measurements):
                timestamp = measurements[i].measured_at.isoformat()
                self.mark_measurement_accepted(user_id, timestamp, {
                    "quality_score": result.quality_score,
                    "kalman_estimate": result.kalman_estimate,
                    "processing_result": result.model_dump()
                })

    def is_accepted(self, user_id: str, timestamp: str) -> bool:
        """Check if a measurement was accepted."""
        return (user_id, timestamp) in self.accepted_measurements


def process_individual_measurements(
    service: WeightProcessorService,
    user_measurements: Dict[str, List[Measurement]],
    acceptance_tracker: AcceptanceTracker,
    batch_size: int = 1
) -> Dict[str, Dict[str, Any]]:
    """
    Process measurements individually (or in small batches) for each user.

    Args:
        service: Weight processor service instance
        user_measurements: Dict mapping user_id to measurements
        acceptance_tracker: Tracker for accepted measurements
        batch_size: Number of measurements to send per call

    Returns:
        Dict mapping user_id to processing results
    """
    results = {}
    total_users = len(user_measurements)
    total_measurements = sum(len(measurements) for measurements in user_measurements.values())

    print(f"\nProcessing measurements individually (batch_size={batch_size})...")
    print(f"Total users: {total_users:,}")
    print(f"Total measurements: {total_measurements:,}")

    processed_measurements = 0
    successful_users = 0
    failed_users = 0

    for i, (user_id, measurements) in enumerate(user_measurements.items(), 1):
        print(f"[{i}/{total_users}] Processing user {user_id[:12]}... ({len(measurements)} measurements)")

        user_results = {
            "measurements_processed": 0,
            "measurements_accepted": 0,
            "measurements_rejected": 0,
            "api_calls": 0,
            "errors": []
        }

        # Sort measurements by timestamp
        sorted_measurements = sorted(measurements, key=lambda m: m.measured_at)

        # Process in batches
        for batch_start in range(0, len(sorted_measurements), batch_size):
            batch = sorted_measurements[batch_start:batch_start + batch_size]

            try:
                response: ProcessResponseData = service.process_batch(user_id, batch)
                user_results["api_calls"] += 1

                user_results["measurements_processed"] += response.measurements_processed
                user_results["measurements_accepted"] += response.measurements_accepted
                user_results["measurements_rejected"] += response.measurements_rejected

                processed_measurements += response.measurements_processed

                # Track accepted measurements
                acceptance_tracker.mark_batch_results(user_id, batch, response)

            except Exception as e:
                error_msg = str(e)
                user_results["errors"].append(f"Batch {batch_start//batch_size + 1}: {error_msg}")
                print(f"  Error in batch {batch_start//batch_size + 1}: {error_msg}")

        results[user_id] = user_results

        if user_results["errors"]:
            failed_users += 1
        else:
            successful_users += 1

        # Progress update
        if i % 10 == 0 or i == total_users:
            print(f"  Progress: {i}/{total_users} users, {processed_measurements:,}/{total_measurements:,} measurements")

    print("\nIndividual processing complete:")
    print(f"  Successful users: {successful_users:,}")
    print(f"  Failed users: {failed_users:,}")
    print(f"  Total measurements processed: {processed_measurements:,}")

    return results


def process_replay_with_outlier_detection(
    state_store: ProcessorStateDB,
    user_measurements: Dict[str, List[Measurement]],
    acceptance_tracker: AcceptanceTracker,
    config: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Process replay with outlier detection and selective replay.

    This implements the proper replay mechanism:
    1. Buffer measurements in windows
    2. Detect outliers by comparing to pre-window Kalman state
    3. Restore state to before window
    4. Replay only clean measurements chronologically

    Based on local_old.py:_process_replay_buffer() implementation.

    Args:
        state_store: State storage instance
        user_measurements: Dict mapping user_id to measurements
        acceptance_tracker: Tracker for accepted measurements
        config: Configuration dictionary

    Returns:
        Dict mapping user_id to replay results
    """
    replay_results = {}

    # Import replay components
    try:
        sys.path.insert(0, str(Path(__file__).parent / "weight_values"))
        from weight_values.src.core.processing.outlier_detection import OutlierDetector
        from weight_values.src.core.replay.replay_manager import ReplayManager
    except ImportError as e:
        print(f"Error: Could not import replay components: {e}")
        print("  Replay processing requires: OutlierDetector, ReplayManager")
        return replay_results

    # Initialize components
    replay_config = config.get("replay", {})
    outlier_config = config.get("outlier_detection", {})

    outlier_detector = OutlierDetector(outlier_config, db=state_store)
    replay_manager = ReplayManager(state_store, replay_config.get("safety", {}))

    # Filter users with enough data for meaningful replay
    min_measurements = replay_config.get("min_measurements", 10)
    eligible_users = {
        uid: measurements for uid, measurements in user_measurements.items()
        if len(measurements) >= min_measurements
    }

    if not eligible_users:
        print(f"No users have sufficient data for replay processing (need >= {min_measurements} measurements)")
        return replay_results

    print(f"\nProcessing replay with outlier detection for {len(eligible_users):,} eligible users...")
    print(f"  Buffer window: {replay_config.get('buffer_hours', 72)} hours")
    print(f"  Min measurements: {min_measurements}")

    successful_replays = 0
    failed_replays = 0
    total_outliers = 0
    total_corrections = 0

    for i, (user_id, measurements) in enumerate(eligible_users.items(), 1):
        print(f"[{i}/{len(eligible_users)}] Replay analysis for user {user_id[:12]}...")

        # Sort measurements by timestamp
        sorted_measurements = sorted(measurements, key=lambda m: m.measured_at)

        # Take middle point as buffer start (mimics 72-hour window)
        buffer_anchor_idx = len(sorted_measurements) // 2
        buffer_start_time = sorted_measurements[buffer_anchor_idx].measured_at

        # Get measurements from anchor onwards (the "buffer window")
        buffered_measurements = sorted_measurements[buffer_anchor_idx:]

        if len(buffered_measurements) < 5:
            print(f"  Skipping: insufficient buffer data ({len(buffered_measurements)} measurements)")
            continue

        print(f"  Analyzing {len(buffered_measurements)} measurements from {buffer_start_time}")

        # Save state snapshot before buffer processing
        state_store.save_state_snapshot(user_id, buffer_start_time)

        # Convert Measurement objects to dict format for outlier detector
        buffer_dicts = [{
            "weight": m.weight_value,
            "timestamp": m.measured_at,
            "source": m.source,
            "unit": m.weight_unit,
            "metadata": m.metadata or {}
        } for m in buffered_measurements]

        try:
            # Detect outliers
            clean_measurements, outlier_indices = outlier_detector.get_clean_measurements(
                buffer_dicts, user_id=user_id
            )

            outliers_found = len(outlier_indices)
            total_outliers += outliers_found

            if outliers_found > 0:
                print(f"  Found {outliers_found} outliers, replaying {len(clean_measurements)} clean measurements")

                # Replay clean measurements
                replay_result = replay_manager.replay_clean_measurements(
                    user_id=user_id,
                    clean_measurements=clean_measurements,
                    buffer_start_time=buffer_start_time
                )

                if replay_result["success"]:
                    successful_replays += 1
                    corrections = len(buffered_measurements) - len(clean_measurements)
                    total_corrections += corrections

                    result = {
                        "buffer_start": buffer_start_time.isoformat(),
                        "measurements_analyzed": len(buffered_measurements),
                        "outliers_found": outliers_found,
                        "clean_measurements": len(clean_measurements),
                        "replay_success": True,
                        "corrections_made": corrections,
                    }

                    print(f"  ✓ Replay successful: {corrections} measurements corrected")
                else:
                    failed_replays += 1
                    result = {
                        "buffer_start": buffer_start_time.isoformat(),
                        "measurements_analyzed": len(buffered_measurements),
                        "outliers_found": outliers_found,
                        "replay_success": False,
                        "error": replay_result.get("error", "Unknown error")
                    }
                    print(f"  ✗ Replay failed: {result['error']}")
            else:
                result = {
                    "buffer_start": buffer_start_time.isoformat(),
                    "measurements_analyzed": len(buffered_measurements),
                    "outliers_found": 0,
                    "clean_measurements": len(buffered_measurements),
                    "replay_success": False,
                    "skipped": "No outliers found"
                }
                print(f"  No outliers found, no replay needed")

            replay_results[user_id] = result

        except Exception as e:
            failed_replays += 1
            error_msg = str(e)
            print(f"  ✗ Analysis failed: {error_msg}")

            replay_results[user_id] = {
                "buffer_start": buffer_start_time.isoformat(),
                "measurements_analyzed": len(buffered_measurements),
                "replay_success": False,
                "error": error_msg
            }

    print("\nReplay processing complete:")
    print(f"  Successful replays: {successful_replays:,}/{len(eligible_users):,}")
    if failed_replays > 0:
        print(f"  Failed/skipped: {failed_replays:,}")
    print(f"  Total outliers found: {total_outliers:,}")
    print(f"  Total corrections made: {total_corrections:,}")

    # Note: Acceptance tracker is NOT cleared in proper replay
    # The replay updates Kalman state but doesn't change which measurements were "accepted"
    # The filtered CSV still reflects original acceptance decisions
    print("\nNote: Replay modifies Kalman state but does not change acceptance tracking")
    print("      Filtered CSV contains original acceptance results")

    return replay_results


def write_filtered_csv(
    original_rows: List[Dict[str, Any]],
    acceptance_tracker: AcceptanceTracker,
    output_path: str
) -> int:
    """
    Write filtered CSV with only accepted measurements.

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

    accepted_count = 0
    total_count = len(original_rows)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in original_rows:
            user_id = row.get("user_id")
            # Handle both old and new column names for timestamp
            timestamp = row.get("effective_date_time") or row.get("effectiveDateTime")

            # Convert timestamp to ISO format to match what's stored in AcceptanceTracker
            if timestamp:
                normalized_timestamp = parse_timestamp(timestamp).isoformat()
            else:
                normalized_timestamp = None

            if user_id and normalized_timestamp and acceptance_tracker.is_accepted(user_id, normalized_timestamp):
                # Write only the original CSV fields (exclude tracking fields)
                filtered_row = {k: v for k, v in row.items() if not k.startswith('_')}
                writer.writerow(filtered_row)
                accepted_count += 1

    print(f"Filtered CSV written: {accepted_count:,}/{total_count:,} measurements accepted ({accepted_count/total_count*100:.1f}%)")

    return accepted_count


def main():
    parser = argparse.ArgumentParser(description="Local Weight Stream Processor")
    parser.add_argument(
        "--csv-file",
        default="data/2025-09-29_weights_all.csv",
        help="CSV file to process (default: data/2025-09-29_weights_all.csv)"
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="Maximum users to process (0 for no limit)"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Maximum CSV rows to read (0 for no limit)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of measurements per processing call (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        default="output_local",
        help="Output directory for results"
    )
    parser.add_argument(
        "--filtered-csv",
        help="Output path for filtered CSV (default: output_dir/filtered_TIMESTAMP.csv)"
    )
    parser.add_argument(
        "--config",
        help="Path to config file (optional, will use defaults if not provided)"
    )
    parser.add_argument(
        "--enable-replay",
        action="store_true",
        default=True,
        help="Enable replay with outlier detection after individual processing (default: enabled)"
    )
    parser.add_argument(
        "--disable-replay",
        action="store_true",
        help="Disable replay processing (only do individual processing, matches reference dataset creation)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.csv_file).exists():
        print(f"Error: CSV file not found: {args.csv_file}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize in-memory storage
    print("Initializing in-memory storage...")
    state_store = ProcessorStateDB()

    # Load configuration
    print("Loading configuration...")
    if args.config:
        config = ConfigManager.load_config(source="file", config_path=args.config)
        print(f"  Using config from: {args.config}")
    else:
        # Use default config based on lambda.env.template
        config = get_default_config()
        print("  Using default configuration (based on lambda.env.template)")

    # Initialize service with in-memory storage
    print("Initializing weight processor service...")
    service = WeightProcessorService(state_store=state_store, config=config)

    # Load CSV data
    user_measurements, original_rows = load_csv_data(
        args.csv_file,
        max_users=args.max_users,
        max_rows=args.max_rows
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
        "csv_file": args.csv_file,
        "storage_type": "in-memory",
        "users_loaded": len(user_measurements),
        "total_measurements": sum(len(m) for m in user_measurements.values()),
        "individual_processing": None,
        "replay_processing": None,
        "replay_enabled": args.enable_replay,
    }

    # Phase 1: Individual measurement processing
    print("\n=== Phase 1: Individual Processing ===")
    individual_results = process_individual_measurements(
        service,
        user_measurements,
        acceptance_tracker,
        batch_size=args.batch_size
    )
    overall_results["individual_processing"] = individual_results

    # Phase 2: Replay with outlier detection (enabled by default)
    if args.disable_replay:
        print("\n=== Replay Disabled (--disable-replay specified) ===")
        print("NOTE: Matches reference dataset creation behavior")
    else:
        print("\n=== Phase 2: Replay with Outlier Detection ===")
        print("NOTE: This analyzes measurement windows, detects outliers, and replays clean data")
        print("      The Kalman state is corrected but acceptance tracker is NOT changed")

        replay_results = process_replay_with_outlier_detection(
            state_store,
            user_measurements,
            acceptance_tracker,
            config,
        )
        overall_results["replay_processing"] = replay_results

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
    print(f"Results saved to: {results_file}")
    print(f"Filtered CSV saved to: {filtered_csv_path}")

    # Print summary statistics
    print("\nProcessing Summary:")
    if overall_results["individual_processing"]:
        individual_stats = overall_results["individual_processing"]
        total_processed = sum(r["measurements_processed"] for r in individual_stats.values())
        total_accepted = sum(r["measurements_accepted"] for r in individual_stats.values())
        print(f"  Phase 1 (Individual): {total_processed:,} processed, {total_accepted:,} accepted")

    if overall_results["replay_processing"]:
        replay_stats = overall_results["replay_processing"]
        successful_replays = sum(1 for r in replay_stats.values() if r.get("replay_success", False))
        total_analyzed = sum(r.get("measurements_analyzed", 0) for r in replay_stats.values())
        total_outliers = sum(r.get("outliers_found", 0) for r in replay_stats.values())
        total_corrections = sum(r.get("corrections_made", 0) for r in replay_stats.values())
        print(f"  Phase 2 (Replay with Outlier Detection):")
        print(f"    Measurements analyzed: {total_analyzed:,}")
        print(f"    Outliers detected: {total_outliers:,}")
        print(f"    Corrections made: {total_corrections:,}")
        print(f"    Successful replays: {successful_replays:,}/{len(replay_stats):,} users")
        print(f"\n  NOTE: Replay corrects Kalman state but does NOT change acceptance decisions")
        print(f"        Filtered CSV still contains Phase 1 (individual) acceptance results")
    else:
        print(f"\n  NOTE: Filtered CSV contains INDIVIDUAL results (no replay)")

    print(f"\nFiltered CSV: {accepted_count:,} accepted measurements written")

    # Calculate acceptance rate
    if overall_results["total_original_measurements"] > 0:
        acceptance_rate = (accepted_count / overall_results["total_original_measurements"]) * 100
        print(f"Acceptance rate: {acceptance_rate:.1f}% ({accepted_count:,}/{overall_results['total_original_measurements']:,})")

    return 0


if __name__ == "__main__":
    exit(main())
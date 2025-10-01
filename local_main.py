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
# Add local to path for visualization imports
sys.path.insert(0, str(Path(__file__).parent))

from weight_values.src.aws.api.models import Measurement, ProcessResponseData
from weight_values.src.aws.services.weight_processor_service import WeightProcessorService
from weight_values.src.core.database.database import ProcessorStateDB
from weight_values.src.aws.config.config_manager import ConfigManager
from weight_values.src.core.constants import SUPPORTED_WEIGHT_UNITS

# Visualization imports (optional)
try:
    from local.viz.visualization import create_weight_timeline
    from local.viz.viz_index import create_index_from_results
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    # Store error for debugging
    _viz_import_error = str(e)


def get_default_config() -> Dict[str, Any]:
    """
    Load configuration from weight_values/config.toml.
    Override database backend for local processing.

    Returns:
        Configuration dictionary
    """
    # Load from unified config file
    config = ConfigManager.load_config(source="file")

    # Override database backend for local in-memory processing
    config["database"]["backend"] = "memory"

    return config


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


def load_csv_data(csv_path: str, max_users: int = 0, max_rows: int = 0, min_readings: int = 0) -> Tuple[Dict[str, List[Measurement]], List[Dict[str, Any]]]:
    """
    Load CSV data and group measurements by user_id.

    Args:
        csv_path: Path to CSV file
        max_users: Maximum number of users to process (0 for no limit)
        max_rows: Maximum number of rows to read (0 for no limit)
        min_readings: Minimum number of readings per user (0 for no filter)

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

    # Calculate initial totals
    initial_user_count = len(user_measurements)
    initial_measurement_count = sum(len(m) for m in user_measurements.values())

    # Filter users by minimum readings BEFORE applying max_users limit
    if min_readings > 0:
        users_before_filter = len(user_measurements)
        measurements_before_filter = sum(len(m) for m in user_measurements.values())

        # Filter out users with fewer than min_readings
        user_measurements = {
            uid: measurements
            for uid, measurements in user_measurements.items()
            if len(measurements) >= min_readings
        }

        # Filter original_rows to match remaining users
        remaining_user_set = set(user_measurements.keys())
        original_rows = [row for row in original_rows if row.get("user_id") in remaining_user_set]

        users_filtered = users_before_filter - len(user_measurements)
        measurements_filtered = measurements_before_filter - sum(len(m) for m in user_measurements.values())

        if users_filtered > 0:
            print(f"\nFiltered out {users_filtered:,} users with < {min_readings} readings ({measurements_filtered:,} measurements)")
            print(f"Remaining: {len(user_measurements):,} users with {sum(len(m) for m in user_measurements.values()):,} measurements")

    # Apply user limit AFTER min_readings filter
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
        self.user_detailed_results = {}     # user_id -> list of detailed results for viz

    def clear(self):
        """Clear all tracked acceptances (used for replay)."""
        self.accepted_measurements.clear()
        self.user_acceptance_details.clear()
        self.user_detailed_results.clear()

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

    def store_detailed_result(self, user_id: str, measurement: Measurement, result, was_reset: bool = False, reset_info: Dict = None):
        """Store detailed result for visualization."""
        if user_id not in self.user_detailed_results:
            self.user_detailed_results[user_id] = []

        # Get kalman estimate with fallback to raw weight
        kalman_est = getattr(result, 'kalman_estimate', None) or measurement.weight_value
        kalman_unc = getattr(result, 'kalman_uncertainty', getattr(result, 'kalman_variance', 1.0))
        if kalman_unc is None:
            kalman_unc = 1.0

        # Create detailed result dict for visualization
        # Ensure all numeric fields have non-None values for formatting
        detail = {
            "timestamp": measurement.measured_at.isoformat(),
            "raw_weight": measurement.weight_value,
            "source": measurement.source,
            "accepted": result.accepted,
            "filtered_weight": kalman_est if result.accepted else measurement.weight_value,
            "quality_score": getattr(result, 'quality_score', None) or 0.0,
            "kalman_estimate": kalman_est,
            "kalman_variance": kalman_unc,
            "innovation": getattr(result, 'innovation', 0.0),
            "normalized_innovation": getattr(result, 'normalized_innovation', 0.0),
            "confidence": getattr(result, 'confidence', 0.95),
            "trend": getattr(result, 'trend', 0.0),
            "trend_weekly": getattr(result, 'trend_weekly', 0.0),
            "kalman_confidence_upper": getattr(result, 'kalman_confidence_upper', None) or (kalman_est + 2 * kalman_unc),
            "kalman_confidence_lower": getattr(result, 'kalman_confidence_lower', None) or (kalman_est - 2 * kalman_unc),
            "quality_components": getattr(result, 'quality_components', None) or {},
            "was_reset": was_reset,
        }

        if not result.accepted:
            detail["reason"] = getattr(result, 'rejection_reason', getattr(result, 'reason', 'Unknown'))

        if reset_info:
            detail.update(reset_info)

        self.user_detailed_results[user_id].append(detail)

    def is_accepted(self, user_id: str, timestamp: str) -> bool:
        """Check if a measurement was accepted."""
        return (user_id, timestamp) in self.accepted_measurements

    def update_from_replay_results(self, user_id: str, replay_result):
        """
        Update acceptance tracking based on replay results.

        Args:
            user_id: User identifier
            replay_result: ReplayResultData from service.execute_replay()
        """
        # Clear existing acceptances for measurements in the replay window
        to_remove = [
            (uid, ts) for uid, ts in self.accepted_measurements
            if uid == user_id
        ]
        for item in to_remove:
            self.accepted_measurements.discard(item)

        # Re-add based on NEW replay results
        for result in replay_result.measurement_results:
            if result.accepted:
                # Extract timestamp from measurement result
                # This assumes the result has the measurement metadata
                timestamp = result.measured_at.isoformat() if hasattr(result, 'measured_at') else None
                if timestamp:
                    self.mark_measurement_accepted(user_id, timestamp, {
                        "quality_score": result.quality_score,
                        "kalman_estimate": result.kalman_estimate,
                        "from_replay": True
                    })


def process_measurements_with_continuous_replay(
    service: WeightProcessorService,
    user_measurements: Dict[str, List[Measurement]],
    acceptance_tracker: AcceptanceTracker,
    enable_replay: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Process measurements one at a time with external replay triggering.

    After each measurement, checks if replay should trigger and executes if needed.
    Caller maintains control over acceptance tracking.

    Args:
        service: Weight processor service instance
        user_measurements: Dict mapping user_id to measurements
        acceptance_tracker: Tracker for accepted measurements
        enable_replay: Whether to check for and execute replay after each measurement

    Returns:
        Dict mapping user_id to processing results
    """
    results = {}
    total_users = len(user_measurements)
    total_measurements = sum(len(measurements) for measurements in user_measurements.values())

    print(f"\nProcessing {total_users:,} users with continuous replay...")
    print(f"Total measurements: {total_measurements:,}")
    print(f"Replay: {'ENABLED' if enable_replay else 'DISABLED'}")

    processed_measurements = 0
    successful_users = 0
    failed_users = 0

    for i, (user_id, measurements) in enumerate(user_measurements.items(), 1):
        print(f"[{i}/{total_users}] Processing user {user_id[:12]}... ({len(measurements)} measurements)")

        user_results = {
            "measurements_processed": 0,
            "measurements_accepted": 0,
            "measurements_rejected": 0,
            "replays_triggered": 0,
            "total_corrections": 0,
            "errors": []
        }

        # Sort by timestamp
        sorted_measurements = sorted(measurements, key=lambda m: m.measured_at)

        # Process ONE AT A TIME
        for j, measurement in enumerate(sorted_measurements):
            try:
                # 1. Process measurement
                response: ProcessResponseData = service.process_batch(user_id, [measurement])
                user_results["measurements_processed"] += 1
                user_results["measurements_accepted"] += response.measurements_accepted
                user_results["measurements_rejected"] += response.measurements_rejected

                processed_measurements += 1

                # 2. Track initial acceptance
                acceptance_tracker.mark_batch_results(user_id, [measurement], response)

                # 3. Store detailed result for visualization
                if response.results:
                    acceptance_tracker.store_detailed_result(user_id, measurement, response.results[0])

                # 4. Check if replay should trigger
                if enable_replay:
                    trigger_check = service.should_trigger_replay(
                        user_id, measurement.measured_at
                    )

                    if trigger_check.should_trigger:
                        # 5. Execute replay (service handles outlier detection)
                        replay_result = service.execute_replay(
                            user_id, trigger_check.window_info
                        )

                        if replay_result.success:
                            user_results["replays_triggered"] += 1
                            user_results["total_corrections"] += replay_result.corrections_made

                            # 6. Update acceptance tracking based on NEW results
                            acceptance_tracker.update_from_replay_results(
                                user_id, replay_result
                            )


                        else:
                            user_results["errors"].append(f"Replay failed: {replay_result.error}")
                            print(f"  └─ Replay failed: {replay_result.error}")

            except Exception as e:
                error_msg = str(e)
                user_results["errors"].append(f"Measurement {j+1}: {error_msg}")
                print(f"  Error processing measurement {j+1}: {error_msg}")

        results[user_id] = user_results

        if user_results["errors"]:
            failed_users += 1
        else:
            successful_users += 1

        # Progress update
        if i % 10 == 0 or i == total_users:
            print(f"  Progress: {i}/{total_users} users, {processed_measurements:,}/{total_measurements:,} measurements")

    print("\nProcessing complete:")
    print(f"  Successful users: {successful_users:,}")
    print(f"  Failed users: {failed_users:,}")
    print(f"  Total measurements processed: {processed_measurements:,}")

    return results




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
        "--min-readings",
        type=int,
        default=20,
        help="Minimum number of readings per user (default: 20, users below this are filtered out)"
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
        help="Enable continuous replay checking after each measurement (default: enabled)"
    )
    parser.add_argument(
        "--disable-replay",
        action="store_true",
        help="Disable continuous replay (measurements processed sequentially without replay)"
    )
    parser.add_argument(
        "--enable-viz",
        action="store_true",
        help="Enable visualization generation (creates HTML dashboards for each user and index.html)"
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
        max_rows=args.max_rows,
        min_readings=args.min_readings
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
        "processing_results": None,
        "replay_mode": "continuous" if not args.disable_replay else "disabled",
    }

    # Single phase: Process with continuous replay
    print("\n=== Processing with Continuous Replay ===")
    print(f"Replay: {'ENABLED' if not args.disable_replay else 'DISABLED'}")

    processing_results = process_measurements_with_continuous_replay(
        service=service,
        user_measurements=user_measurements,
        acceptance_tracker=acceptance_tracker,
        enable_replay=not args.disable_replay
    )

    overall_results["processing_results"] = processing_results

    # Write filtered CSV
    timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
    filtered_csv_path = args.filtered_csv or str(output_dir / f"filtered_{timestamp_str}.csv")
    accepted_count = write_filtered_csv(original_rows, acceptance_tracker, filtered_csv_path)

    # Generate visualizations if enabled
    if args.enable_viz:
        if not VISUALIZATION_AVAILABLE:
            print("\n⚠️  Visualization libraries not available. Skipping visualization generation.")
            if '_viz_import_error' in globals():
                print(f"    Import error: {_viz_import_error}")
            print("    Install plotly to enable visualizations: pip install plotly")
        else:
            print("\n=== Generating Visualizations ===")
            viz_output_dir = output_dir / "visualizations"
            viz_output_dir.mkdir(exist_ok=True, parents=True)

            print(f"Output directory: {viz_output_dir}")

            # Generate dashboard for each user
            dashboard_count = 0
            failed_count = 0

            for user_id, results in acceptance_tracker.user_detailed_results.items():
                if not results:
                    continue

                try:
                    print(f"  Generating dashboard for {user_id[:12]}... ({len(results)} measurements)")
                    html_file = create_weight_timeline(
                        results=results,
                        user_id=user_id,
                        output_dir=str(viz_output_dir),
                        config=None
                    )
                    dashboard_count += 1
                except Exception as e:
                    print(f"  ⚠️  Failed to generate dashboard for {user_id[:12]}: {e}")
                    failed_count += 1

            print(f"\nGenerated {dashboard_count} user dashboards")
            if failed_count > 0:
                print(f"Failed to generate {failed_count} dashboards")

            # Generate index.html
            try:
                print("\nGenerating index.html...")
                index_path = create_index_from_results(
                    acceptance_tracker.user_detailed_results,
                    str(viz_output_dir),
                    "index.html"
                )
                print(f"✓ Index file created: {index_path}")
                print(f"\n📊 Open {index_path} in your browser to view all dashboards")
            except Exception as e:
                print(f"⚠️  Failed to generate index.html: {e}")

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
    if overall_results["processing_results"]:
        processing_stats = overall_results["processing_results"]
        total_processed = sum(r["measurements_processed"] for r in processing_stats.values())
        total_accepted = sum(r["measurements_accepted"] for r in processing_stats.values())
        total_replays = sum(r.get("replays_triggered", 0) for r in processing_stats.values())
        total_corrections = sum(r.get("total_corrections", 0) for r in processing_stats.values())

        print(f"  Measurements processed: {total_processed:,}")
        print(f"  Measurements accepted: {total_accepted:,}")

        if overall_results["replay_mode"] == "continuous":
            print(f"\n  Continuous Replay:")
            print(f"    Total replays triggered: {total_replays:,}")
            print(f"    Total corrections made: {total_corrections:,}")
            print(f"\n  NOTE: Replay results are reflected in the acceptance tracking")
            print(f"        Filtered CSV contains FINAL acceptance results after replay")
        else:
            print(f"\n  NOTE: Replay was DISABLED - filtered CSV contains results without replay")

    print(f"\nFiltered CSV: {accepted_count:,} accepted measurements written")

    # Calculate acceptance rate
    if overall_results["total_original_measurements"] > 0:
        acceptance_rate = (accepted_count / overall_results["total_original_measurements"]) * 100
        print(f"Acceptance rate: {acceptance_rate:.1f}% ({accepted_count:,}/{overall_results['total_original_measurements']:,})")

    return 0


if __name__ == "__main__":
    exit(main())

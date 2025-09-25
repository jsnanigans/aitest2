"""CSV batch processing for weight measurements."""

import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ..constants import SUPPORTED_WEIGHT_UNITS
from ..processing.processor import process_measurement
from ..processing.validation import DataQualityPreprocessor
from ..factories.component_factory import ComponentFactory
from ..services.weight_processor_service import WeightProcessorService
from ..api.models import Measurement
from ..utils import set_verbosity


class CSVBatchProcessor:
    """Handles CSV file processing for batch operations."""

    def __init__(self, service: WeightProcessorService = None):
        """
        Initialize CSV processor.

        Args:
            service: Optional weight processor service
        """
        self.service = service or ComponentFactory.get_weight_processor_service()
        self.state_store = self.service.state_store

    def process_file(
        self,
        csv_path: str,
        output_dir: str,
        config: Dict[str, Any],
        filtered_output: str = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Process CSV file containing weight measurements.

        Args:
            csv_path: Path to input CSV
            output_dir: Directory for outputs
            config: Configuration dict
            filtered_output: Optional path to write filtered CSV (accepted rows only)
            debug: Enable debug mode

        Returns:
            Processing statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Set verbosity
        self._set_verbosity(config)

        # Load height data once
        DataQualityPreprocessor.load_height_data()

        # Initialize replay processing if enabled
        replay_components = self._init_replay_processing(config)

        # Parse configuration
        processing_config = self._parse_config(config)

        # Initialize tracking
        stats = self._init_stats()
        user_results = {}

        # Setup filtered CSV writer if requested
        filtered_csv_file = None
        filtered_csv_writer = None

        print(f"Processing {csv_path}...")
        self._print_config(processing_config)

        # Determine eligible users
        eligible_users_set = self._determine_eligible_users(
            csv_path, processing_config, stats
        )

        # Main processing loop
        with open(csv_path) as f:
            reader = csv.DictReader(f)

            # Setup filtered CSV writer
            if filtered_output:
                filtered_csv_file, filtered_csv_writer = self._setup_filtered_csv(
                    filtered_output, reader.fieldnames
                )

            # Process rows
            for row in reader:
                result = self._process_row(
                    row,
                    processing_config,
                    eligible_users_set,
                    stats,
                    user_results,
                    config,
                    replay_components,
                )

                # Write to filtered CSV if accepted
                if result and result.get("accepted") and filtered_csv_writer:
                    self._write_filtered_row(row, result, filtered_csv_writer)

        # Close filtered CSV
        if filtered_csv_file:
            filtered_csv_file.close()

        # Generate visualizations if enabled
        if config.get("visualization", {}).get("enabled", True):
            self._generate_visualizations(user_results, output_path, config)

        # Write summary
        self._write_summary(stats, output_path, user_results)

        return stats

    def _set_verbosity(self, config: Dict[str, Any]):
        """Set verbosity level from configuration."""
        viz_config = config.get("visualization", {})
        verbosity_str = viz_config.get("verbosity", "normal")
        verbosity_map = {"silent": 0, "minimal": 1, "normal": 2, "verbose": 3}
        verbosity_level = verbosity_map.get(verbosity_str, 2)
        set_verbosity(verbosity_level)

    def _init_replay_processing(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize replay processing components if enabled."""
        replay_config = config.get("replay", {})
        replay_enabled = replay_config.get("enabled", False)

        if not replay_enabled:
            return {}

        try:
            from ..processing.buffer_factory import get_factory
            from ..processing.outlier_detection import OutlierDetector
            from ..replay.replay_manager import ReplayManager

            # Create buffer using factory
            buffer_factory = get_factory()
            buffer_factory.set_default_config(replay_config)
            replay_buffer = buffer_factory.create_buffer("default", replay_config)

            outlier_detector = OutlierDetector(
                replay_config.get("outlier_detection", {}), db=self.state_store
            )
            replay_manager = ReplayManager(
                self.state_store, replay_config.get("safety", {})
            )

            print("Replay processing enabled")
            print(f"  Buffer window: {replay_config.get('buffer_hours', 72)} hours")
            print(f"  Trigger mode: {replay_config.get('trigger_mode', 'time_based')}")

            return {
                "enabled": True,
                "buffer": replay_buffer,
                "outlier_detector": outlier_detector,
                "replay_manager": replay_manager,
            }
        except ImportError as e:
            print(f"Warning: Could not initialize replay processing: {e}")
            return {}

    def _parse_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate configuration."""
        data_config = config.get("data", {})

        # Load test users
        test_users = self._load_test_users(data_config)

        return {
            "max_users": data_config.get("max_users", 0),
            "user_offset": data_config.get("user_offset", 0),
            "min_readings": data_config.get("min_readings", 0),
            "test_users": test_users,
            "test_mode": bool(test_users),
            "min_date": self._parse_date(data_config.get("min_date", "")),
            "max_date": self._parse_date(data_config.get("max_date", "")),
        }

    def _load_test_users(self, data_config: Dict[str, Any]) -> List[str]:
        """Load test users from various sources."""
        # Priority: test_users from config > filtered_users_csv > test_users_file
        test_users_config = data_config.get("test_users", [])
        if test_users_config:
            return (
                test_users_config
                if isinstance(test_users_config, list)
                else [test_users_config]
            )

        filtered_users_csv = data_config.get("filtered_users_csv", "")
        if filtered_users_csv:
            users = self._load_filtered_users_csv(filtered_users_csv)
            if users:
                return users

        test_users_file = data_config.get("test_users_file", "")
        if test_users_file:
            return self._load_test_users_file(test_users_file)

        return []

    def _load_test_users_file(self, filepath: str) -> List[str]:
        """Load test user IDs from file."""
        if not Path(filepath).exists():
            return []

        users = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    users.append(line)
        return users

    def _load_filtered_users_csv(self, filepath: str) -> List[str]:
        """Load user IDs from a CSV file."""
        if not Path(filepath).exists():
            print(f"Warning: Filtered users CSV not found: {filepath}")
            return []

        users = []
        try:
            with open(filepath, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    user_id = row.get("user_id", "").strip()
                    if user_id:
                        users.append(user_id)
            print(f"Loaded {len(users)} users from filtered CSV: {filepath}")
        except Exception as e:
            print(f"Error reading filtered users CSV: {e}")
            return []

        return users

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various timestamp formats."""
        if not date_str:
            return None

        try:
            if "T" in date_str:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            elif " " in date_str:
                return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            else:
                return datetime.strptime(date_str, "%Y-%m-%d")
        except:
            return None

    def _init_stats(self) -> Dict[str, Any]:
        """Initialize statistics tracking."""
        return {
            "total_rows": 0,
            "accepted": 0,
            "rejected": 0,
            "date_filtered": 0,
            "unit_rejected": 0,
            "rejected_units": {},
            "start_time": datetime.now(),
            "processing_errors": 0,
            "parse_errors": 0,
            "invalid_weight": 0,
        }

    def _print_config(self, config: Dict[str, Any]):
        """Print processing configuration."""
        if config["test_mode"]:
            print(f"Test mode: Processing {len(config['test_users'])} specific users")
        else:
            if config["max_users"] > 0:
                print(f"  Processing up to {config['max_users']} users")
            if config["min_date"] or config["max_date"]:
                date_range = (
                    f"{config['min_date'] or 'start'} to {config['max_date'] or 'end'}"
                )
                print(f"  Date filter: {date_range}")
        print()

    def _determine_eligible_users(
        self, csv_path: str, config: Dict[str, Any], stats: Dict[str, Any]
    ) -> set:
        """Determine which users are eligible for processing."""
        if config["test_mode"]:
            return set(config["test_users"])

        # Count readings per user
        user_reading_counts = {}
        eligible_users = []

        print(
            f"Analyzing user data (min_readings={config['min_readings']}, max_users={config['max_users']})..."
        )

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_id = row.get("user_id")
                if not user_id:
                    continue

                # Basic validation
                weight_str = row.get("weight", "").strip()
                if not weight_str or weight_str.upper() == "NULL":
                    continue

                # Date check
                if config["min_date"] or config["max_date"]:
                    date_str = row.get("effectiveDateTime")
                    try:
                        timestamp = self._parse_date(date_str)
                        if config["min_date"] and timestamp < config["min_date"]:
                            continue
                        if config["max_date"] and timestamp > config["max_date"]:
                            continue
                    except:
                        continue

                user_reading_counts[user_id] = user_reading_counts.get(user_id, 0) + 1

        # Determine eligible users based on min_readings
        for user_id, count in sorted(user_reading_counts.items()):
            if count >= config["min_readings"]:
                eligible_users.append(user_id)

        # Apply user_offset and max_users
        if config["user_offset"] > 0:
            eligible_users = eligible_users[config["user_offset"] :]

        if config["max_users"] > 0 and len(eligible_users) > config["max_users"]:
            eligible_users = eligible_users[: config["max_users"]]

        print(f"  Found {len(user_reading_counts)} total users")
        print(
            f"  Processing {len(eligible_users)} users (after offset={config['user_offset']}, max={config['max_users']})"
        )

        return set(eligible_users)

    def _setup_filtered_csv(self, filepath: str, fieldnames: list) -> Tuple:
        """Setup filtered CSV writer."""
        print(f"Will write filtered data to: {filepath}")
        file = open(filepath, "w", newline="")
        extended_fieldnames = list(fieldnames) + ["quality_score"]
        writer = csv.DictWriter(file, fieldnames=extended_fieldnames)
        writer.writeheader()
        return file, writer

    def _process_row(
        self,
        row: Dict[str, Any],
        config: Dict[str, Any],
        eligible_users: set,
        stats: Dict[str, Any],
        user_results: Dict[str, Any],
        full_config: Dict[str, Any],
        replay_components: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Process a single CSV row."""
        stats["total_rows"] += 1

        # Progress update
        if (
            stats["total_rows"]
            % full_config.get("logging", {}).get("progress_interval", 10000)
            == 0
        ):
            self._print_progress(stats)

        # Parse and validate row
        user_id = row.get("user_id")
        if not user_id or user_id not in eligible_users:
            return None

        # Parse weight
        weight_str = row.get("weight", "").strip()
        if not weight_str or weight_str.upper() == "NULL":
            return None

        try:
            weight = float(weight_str)
            if weight <= 0 or weight > 1000 or math.isnan(weight) or math.isinf(weight):
                stats["invalid_weight"] += 1
                return None
        except (ValueError, TypeError):
            stats["parse_errors"] += 1
            return None

        # Parse metadata
        date_str = row.get("effectiveDateTime")
        source = row.get("source_type") or row.get("source", "unknown")
        unit = row.get("unit", "").strip()

        # Skip BSA measurements
        if "BSA" in source.upper() or "m2" in unit or "m²" in unit:
            return None

        # Validate unit
        if not unit or unit.lower().strip() not in SUPPORTED_WEIGHT_UNITS:
            stats["unit_rejected"] += 1
            stats["rejected_units"][unit or "<missing>"] = (
                stats["rejected_units"].get(unit or "<missing>", 0) + 1
            )
            return None

        # Parse timestamp
        try:
            timestamp = self._parse_date(date_str) or datetime.now()
        except:
            timestamp = datetime.now()

        # Apply date filters
        if config["min_date"] and timestamp < config["min_date"]:
            stats["date_filtered"] += 1
            return None
        if config["max_date"] and timestamp > config["max_date"]:
            stats["date_filtered"] += 1
            return None

        # Process measurement
        try:
            result = process_measurement(
                user_id=user_id,
                weight=weight,
                timestamp=timestamp,
                source=source,
                config=full_config,
                unit=unit,
                db=self.state_store,
            )

            # Track results
            if result:
                if result.get("accepted"):
                    stats["accepted"] += 1
                else:
                    stats["rejected"] += 1

                # Store for visualization
                if user_id not in user_results:
                    user_results[user_id] = []
                user_results[user_id].append(result)

            return result

        except Exception as e:
            stats["processing_errors"] += 1
            if full_config.get("logging", {}).get("verbose", False):
                print(f"Error processing {user_id}: {e}")
            return None

    def _write_filtered_row(
        self, row: Dict[str, Any], result: Dict[str, Any], writer: csv.DictWriter
    ):
        """Write accepted row to filtered CSV."""
        filtered_row = row.copy()
        filtered_row["quality_score"] = result.get("quality_score", "")
        writer.writerow(filtered_row)

    def _print_progress(self, stats: Dict[str, Any]):
        """Print processing progress."""
        elapsed = (datetime.now() - stats["start_time"]).total_seconds()
        rate = stats["total_rows"] / elapsed if elapsed > 0 else 0
        print(
            f"  Row {stats['total_rows']:,} | "
            f"Accepted: {stats['accepted']:,} | "
            f"Rate: {rate:.0f} rows/sec"
        )

    def _generate_visualizations(
        self, user_results: Dict[str, Any], output_dir: Path, config: Dict[str, Any]
    ):
        """Generate visualizations for processed users."""
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        print(f"\nGenerating visualizations for {len(user_results)} users...")

        # Get optimal thread count
        num_workers = self._get_optimal_thread_count(len(user_results), config)

        # Prepare arguments for parallel processing
        viz_args = [
            (idx, len(user_results), user_id, results, viz_dir, config)
            for idx, (user_id, results) in enumerate(user_results.items(), 1)
        ]

        # Process visualizations in parallel
        successful = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self._generate_single_visualization, args): args[2]
                for args in viz_args
            }

            for future in as_completed(futures):
                user_id, success, error_msg, dashboard_path = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                    if config.get("logging", {}).get("verbose", False):
                        print(
                            f"  Failed to generate visualization for {user_id}: {error_msg}"
                        )

        print(f"Visualizations complete: {successful} successful, {failed} failed")

    def _generate_single_visualization(self, args: Tuple) -> Tuple:
        """Generate visualization for a single user (process-safe)."""
        idx, total_users, user_id, results, viz_dir, config = args

        try:
            from ..viz.visualization import create_weight_timeline

            dashboard_path = create_weight_timeline(
                results, user_id, str(viz_dir), config=config
            )

            return (user_id, True, None, dashboard_path)
        except Exception as e:
            error_msg = str(e)[:100]
            return (user_id, False, error_msg, None)

    def _get_optimal_thread_count(self, num_users: int, config: Dict[str, Any]) -> int:
        """Calculate optimal number of threads for visualization."""
        viz_threading = config.get("visualization", {}).get("threading", {})
        if not viz_threading.get("enabled", True):
            return 1

        max_workers_config = viz_threading.get("max_workers", None)

        # Calculate based on CPU cores
        cpu_count = os.cpu_count() or 4
        default_workers = min(cpu_count, 8)

        # Use config value if specified
        max_workers = max_workers_config if max_workers_config else default_workers

        # Don't use more threads than users
        return max(1, min(max_workers, num_users))

    def _write_summary(
        self, stats: Dict[str, Any], output_dir: Path, user_results: Dict[str, Any]
    ):
        """Write processing summary."""
        elapsed = (datetime.now() - stats["start_time"]).total_seconds()

        summary = {
            "processing_time": elapsed,
            "total_rows": stats["total_rows"],
            "accepted": stats["accepted"],
            "rejected": stats["rejected"],
            "date_filtered": stats["date_filtered"],
            "unit_rejected": stats["unit_rejected"],
            "rejected_units": stats["rejected_units"],
            "users_processed": len(user_results),
            "processing_errors": stats.get("processing_errors", 0),
            "parse_errors": stats.get("parse_errors", 0),
            "invalid_weight": stats.get("invalid_weight", 0),
        }

        summary_path = output_dir / "processing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nProcessing complete in {elapsed:.1f} seconds")
        print(f"  Rows: {stats['total_rows']:,}")
        print(f"  Accepted: {stats['accepted']:,}")
        print(f"  Rejected: {stats['rejected']:,}")
        print(f"  Users: {len(user_results):,}")
        print(f"Summary written to: {summary_path}")

#!/usr/bin/env python3
"""
API-based Weight Stream Processor

Similar to local/main.py but uses the SAM API instead of direct method calls.
Processes weight measurements from CSV data via HTTP requests to the weight processor API.
Outputs a filtered CSV with only accepted (non-rejected) measurements.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class APIResponse:
    """Wrapper for API responses."""
    status_code: int
    data: Dict[str, Any]
    success: bool
    error: Optional[Dict[str, Any]] = None

    @property
    def is_success(self) -> bool:
        return self.success and 200 <= self.status_code < 300


class WeightProcessorAPIClient:
    """Client for the Weight Processor SAM API."""

    def __init__(self, base_url: str = "http://localhost:3080"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-API-Version": "v2"
        })

    def _parse_response(self, response: requests.Response) -> APIResponse:
        """Parse API response into standard format."""
        try:
            data = response.json()

            # V2 format
            if "success" in data:
                return APIResponse(
                    status_code=response.status_code,
                    data=data.get("data", {}),
                    success=data["success"],
                    error=data.get("error")
                )
            else:
                # Fallback for non-standard responses
                return APIResponse(
                    status_code=response.status_code,
                    data=data,
                    success=response.status_code < 400
                )
        except json.JSONDecodeError:
            return APIResponse(
                status_code=response.status_code,
                data={},
                success=False,
                error={"message": response.text}
            )

    def health_check(self) -> APIResponse:
        """Check API health."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/health", timeout=10)
            return self._parse_response(response)
        except requests.RequestException as e:
            return APIResponse(
                status_code=0,
                data={},
                success=False,
                error={"message": str(e)}
            )

    def process_measurements(
        self,
        user_id: str,
        measurements: List[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """Process weight measurements for a user."""
        payload = {
            "measurements": measurements,
            "options": options or {}
        }
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/process/{user_id}",
                json=payload,
                timeout=30
            )
            return self._parse_response(response)
        except requests.RequestException as e:
            return APIResponse(
                status_code=0,
                data={},
                success=False,
                error={"message": str(e)}
            )

    def replay_measurements(
        self,
        user_id: str,
        replay_from: datetime,
        measurements: List[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """Replay measurements from a specific timestamp (legacy endpoint)."""
        # Ensure timezone-aware timestamp
        if replay_from.tzinfo is None:
            replay_from_str = replay_from.isoformat() + "Z"
        else:
            replay_from_str = replay_from.isoformat()

        payload = {
            "replay_from_timestamp": replay_from_str,
            "measurements": measurements,
            "options": options or {}
        }
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/replay/{user_id}",
                json=payload,
                timeout=30
            )
            return self._parse_response(response)
        except requests.RequestException as e:
            return APIResponse(
                status_code=0,
                data={},
                success=False,
                error={"message": str(e)}
            )

    def check_replay(
        self,
        user_id: str,
        current_timestamp: datetime,
        buffer_hours: Optional[int] = None
    ) -> APIResponse:
        """
        Check if replay should trigger for a user.

        Args:
            user_id: User identifier
            current_timestamp: Timestamp of last processed measurement
            buffer_hours: Optional replay window size (default: 72)

        Returns:
            APIResponse with should_trigger and window_info
        """
        # Ensure timezone-aware timestamp
        if current_timestamp.tzinfo is None:
            timestamp_str = current_timestamp.isoformat() + "Z"
        else:
            timestamp_str = current_timestamp.isoformat()

        payload = {
            "user_id": user_id,
            "current_timestamp": timestamp_str
        }

        if buffer_hours is not None:
            payload["buffer_hours"] = buffer_hours

        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/replay/{user_id}/check",
                json=payload,
                timeout=10
            )
            return self._parse_response(response)
        except requests.RequestException as e:
            return APIResponse(
                status_code=0,
                data={},
                success=False,
                error={"message": str(e)}
            )

    def execute_replay(
        self,
        user_id: str,
        window_info: Dict[str, Any]
    ) -> APIResponse:
        """
        Execute replay for a measurement window.

        Args:
            user_id: User identifier
            window_info: Window information from check_replay()

        Returns:
            APIResponse with measurement_results containing NEW acceptance statuses
        """
        payload = {
            "user_id": user_id,
            "window_info": window_info
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/replay/{user_id}/execute",
                json=payload,
                timeout=30
            )
            return self._parse_response(response)
        except requests.RequestException as e:
            return APIResponse(
                status_code=0,
                data={},
                success=False,
                error={"message": str(e)}
            )

    def get_user_state(self, user_id: str) -> APIResponse:
        """Get user processing state."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/state/{user_id}",
                timeout=10
            )
            return self._parse_response(response)
        except requests.RequestException as e:
            return APIResponse(
                status_code=0,
                data={},
                success=False,
                error={"message": str(e)}
            )

    def cleanup_user(
        self,
        user_id: str,
        cleanup_type: str = "reset_adaptive"
    ) -> APIResponse:
        """Cleanup/reset user state."""
        payload = {
            "cleanup_type": cleanup_type,
            "options": {}
        }
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/cleanup/{user_id}",
                json=payload,
                timeout=30
            )
            return self._parse_response(response)
        except requests.RequestException as e:
            return APIResponse(
                status_code=0,
                data={},
                success=False,
                error={"message": str(e)}
            )


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

    print(f"Loading data from {csv_path}...")

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
            original_row["_accepted"] = False  # Will be updated during processing
            original_rows.append(original_row)

            # Convert to measurement format expected by API (Pydantic model format)
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

    # Apply user limit
    if max_users > 0 and len(user_measurements) > max_users:
        # Take first N users by sorted order for consistency
        sorted_users = sorted(user_measurements.keys())[:max_users]
        user_measurements = {uid: user_measurements[uid] for uid in sorted_users}

        # Filter original_rows to match selected users
        selected_user_set = set(sorted_users)
        original_rows = [row for row in original_rows if row.get("user_id") in selected_user_set]

    print(f"Loaded {len(user_measurements):,} users with {sum(len(m) for m in user_measurements.values()):,} total measurements")

    return user_measurements, original_rows


class AcceptanceTracker:
    """Tracks which measurements were accepted during processing."""

    def __init__(self):
        self.accepted_measurements = set()  # Track by (user_id, timestamp)
        self.user_acceptance_details = {}   # user_id -> list of acceptance info

    def mark_measurement_accepted(self, user_id: str, timestamp: str, additional_info: Dict[str, Any] = None):
        """Mark a measurement as accepted."""
        self.accepted_measurements.add((user_id, timestamp))
        if user_id not in self.user_acceptance_details:
            self.user_acceptance_details[user_id] = []

        info = {"timestamp": timestamp, "accepted": True}
        if additional_info:
            info.update(additional_info)
        self.user_acceptance_details[user_id].append(info)

    def mark_batch_results(self, user_id: str, measurements: List[Dict[str, Any]], response_data: Dict[str, Any]):
        """Mark batch measurement results based on API response."""
        # Extract results from API response
        results = response_data.get("results", [])

        if results:
            # API returned detailed results for each measurement
            for i, result in enumerate(results):
                if i < len(measurements):
                    timestamp = measurements[i]["effectiveDateTime"]
                    if result.get("accepted", False):
                        self.mark_measurement_accepted(user_id, timestamp, {
                            "quality_score": result.get("quality_score"),
                            "processing_result": result
                        })
        else:
            # Fallback: assume all measurements up to accepted count are accepted
            accepted_count = response_data.get("measurements_accepted", 0)
            for i in range(min(accepted_count, len(measurements))):
                timestamp = measurements[i]["effectiveDateTime"]
                self.mark_measurement_accepted(user_id, timestamp)

    def is_accepted(self, user_id: str, timestamp: str) -> bool:
        """Check if a measurement was accepted."""
        return (user_id, timestamp) in self.accepted_measurements

    def update_from_replay_results(self, user_id: str, replay_response: Dict[str, Any]):
        """
        Update acceptance tracking based on replay results.

        Args:
            user_id: User identifier
            replay_response: Replay result data from execute_replay() API call
        """
        # Clear existing acceptances for measurements in the replay window
        to_remove = [
            (uid, ts) for uid, ts in self.accepted_measurements
            if uid == user_id
        ]
        for item in to_remove:
            self.accepted_measurements.discard(item)

        # Re-add based on NEW replay results
        measurement_results = replay_response.get("measurement_results", [])
        for result in measurement_results:
            if result.get("accepted", False):
                # Extract timestamp from measurement result
                timestamp = result.get("effectiveDateTime") or result.get("timestamp")
                if timestamp:
                    self.mark_measurement_accepted(user_id, timestamp, {
                        "quality_score": result.get("quality_score"),
                        "from_replay": True
                    })


def process_individual_measurements(
    api_client: WeightProcessorAPIClient,
    user_measurements: Dict[str, List[Dict[str, Any]]],
    acceptance_tracker: AcceptanceTracker,
    batch_size: int = 1
) -> Dict[str, Dict[str, Any]]:
    """
    Process measurements individually (or in small batches) for each user.

    Args:
        api_client: API client instance
        user_measurements: Dict mapping user_id to measurements
        acceptance_tracker: Tracker for accepted measurements
        batch_size: Number of measurements to send per API call

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
        sorted_measurements = sorted(measurements, key=lambda m: parse_timestamp(m["effectiveDateTime"]))

        # Process in batches
        for batch_start in range(0, len(sorted_measurements), batch_size):
            batch = sorted_measurements[batch_start:batch_start + batch_size]

            response = api_client.process_measurements(user_id, batch)
            user_results["api_calls"] += 1

            if response.is_success:
                batch_processed = response.data.get("measurements_processed", 0)
                batch_accepted = response.data.get("measurements_accepted", 0)

                user_results["measurements_processed"] += batch_processed
                user_results["measurements_accepted"] += batch_accepted
                user_results["measurements_rejected"] += (batch_processed - batch_accepted)

                processed_measurements += batch_processed

                # Track accepted measurements
                acceptance_tracker.mark_batch_results(user_id, batch, response.data)

            else:
                error_msg = response.error.get("message", "Unknown error") if response.error else f"HTTP {response.status_code}"
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


def process_replay_batches(
    api_client: WeightProcessorAPIClient,
    user_measurements: Dict[str, List[Dict[str, Any]]],
    acceptance_tracker: AcceptanceTracker,
    replay_window_hours: int = 72
) -> Dict[str, Dict[str, Any]]:
    """
    Process replay batches for users with sufficient data.

    Args:
        api_client: API client instance
        user_measurements: Dict mapping user_id to measurements
        acceptance_tracker: Tracker for accepted measurements
        replay_window_hours: Hours of data to include in replay

    Returns:
        Dict mapping user_id to replay results
    """
    replay_results = {}

    # Filter users with enough data for meaningful replay
    eligible_users = {
        uid: measurements for uid, measurements in user_measurements.items()
        if len(measurements) >= 10  # Minimum measurements for replay
    }

    if not eligible_users:
        print("No users have sufficient data for replay processing")
        return replay_results

    print(f"\nProcessing replay batches for {len(eligible_users):,} eligible users...")
    print(f"Replay window: {replay_window_hours} hours")

    for i, (user_id, measurements) in enumerate(eligible_users.items(), 1):
        print(f"[{i}/{len(eligible_users)}] Replay for user {user_id[:12]}...")

        # Sort measurements by timestamp
        sorted_measurements = sorted(measurements, key=lambda m: parse_timestamp(m["effectiveDateTime"]))

        if len(sorted_measurements) < 10:
            continue

        # Take a point in the middle as replay anchor
        replay_anchor_idx = len(sorted_measurements) // 2
        replay_from = parse_timestamp(sorted_measurements[replay_anchor_idx]["effectiveDateTime"])

        # Get measurements from anchor onwards
        replay_measurements = sorted_measurements[replay_anchor_idx:]

        print(f"  Replaying {len(replay_measurements)} measurements from {replay_from}")

        response = api_client.replay_measurements(
            user_id,
            replay_from,
            replay_measurements
        )

        result = {
            "replay_from": replay_from.isoformat(),
            "measurements_replayed": len(replay_measurements),
            "success": response.is_success,
            "api_response": response.data if response.is_success else None,
            "error": response.error.get("message") if response.error else None
        }

        if response.is_success:
            result["measurements_processed"] = response.data.get("measurements_processed", 0)
            result["measurements_accepted"] = response.data.get("measurements_accepted", 0)
            print(f"  ✓ Replayed {result['measurements_processed']} measurements")

            # Track accepted measurements from replay
            acceptance_tracker.mark_batch_results(user_id, replay_measurements, response.data)
        else:
            error_msg = response.error.get("message", "Unknown error") if response.error else f"HTTP {response.status_code}"
            print(f"  ✗ Replay failed: {error_msg}")

        replay_results[user_id] = result

    successful_replays = sum(1 for r in replay_results.values() if r["success"])
    print("\nReplay processing complete:")
    print(f"  Successful replays: {successful_replays:,}/{len(eligible_users):,}")

    return replay_results


def process_measurements_with_continuous_replay(
    api_client: WeightProcessorAPIClient,
    user_measurements: Dict[str, List[Dict[str, Any]]],
    acceptance_tracker: AcceptanceTracker,
    enable_replay: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Process measurements one at a time with external replay triggering.

    After each measurement, checks if replay should trigger and executes if needed.
    Uses the new /replay/{userId}/check and /replay/{userId}/execute endpoints.

    Args:
        api_client: API client instance
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
        sorted_measurements = sorted(measurements, key=lambda m: parse_timestamp(m["effectiveDateTime"]))

        # Process ONE AT A TIME
        for j, measurement in enumerate(sorted_measurements):
            # 1. Process measurement
            response = api_client.process_measurements(user_id, [measurement])

            if response.is_success:
                user_results["measurements_processed"] += 1
                user_results["measurements_accepted"] += response.data.get("measurements_accepted", 0)
                user_results["measurements_rejected"] += (
                    response.data.get("measurements_processed", 0) -
                    response.data.get("measurements_accepted", 0)
                )

                processed_measurements += 1

                # 2. Track initial acceptance
                acceptance_tracker.mark_batch_results(user_id, [measurement], response.data)

                # 3. Check if replay should trigger
                if enable_replay:
                    measurement_timestamp = parse_timestamp(measurement["effectiveDateTime"])
                    trigger_check = api_client.check_replay(user_id, measurement_timestamp)

                    if trigger_check.is_success and trigger_check.data.get("should_trigger"):
                        # 4. Execute replay (service handles outlier detection)
                        window_info = trigger_check.data.get("window_info")
                        if window_info:
                            replay_result = api_client.execute_replay(user_id, window_info)

                            if replay_result.is_success and replay_result.data.get("success"):
                                user_results["replays_triggered"] += 1
                                user_results["total_corrections"] += replay_result.data.get("corrections_made", 0)

                                # 5. Update acceptance tracking based on NEW results
                                acceptance_tracker.update_from_replay_results(user_id, replay_result.data)

                                print(f"  └─ Replay: {replay_result.data.get('outliers_count', 0)} outliers, "
                                      f"{replay_result.data.get('corrections_made', 0)} corrections")
                            else:
                                error_msg = replay_result.error.get("message", "Unknown error") if replay_result.error else "Replay failed"
                                user_results["errors"].append(f"Replay failed: {error_msg}")
                                print(f"  └─ Replay failed: {error_msg}")
            else:
                error_msg = response.error.get("message", "Unknown error") if response.error else f"HTTP {response.status_code}"
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
    parser = argparse.ArgumentParser(description="API-based Weight Stream Processor")
    parser.add_argument(
        "--csv-file",
        default="data/2025-09-29_weights_all.csv",
        help="CSV file to process (default: data/2025-09-29_weights_all.csv)"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:3080",
        help="SAM API base URL (default: http://localhost:3080)"
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
        help="Number of measurements per API call (default: 1)"
    )
    parser.add_argument(
        "--skip-individual",
        action="store_true",
        help="Skip individual measurement processing"
    )
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="Skip replay batch processing (legacy mode)"
    )
    parser.add_argument(
        "--enable-continuous-replay",
        action="store_true",
        help="Use continuous replay mode (new check/execute endpoints)"
    )
    parser.add_argument(
        "--disable-replay",
        action="store_true",
        help="Disable all replay processing"
    )
    parser.add_argument(
        "--output-dir",
        default="output_api",
        help="Output directory for results"
    )
    parser.add_argument(
        "--filtered-csv",
        help="Output path for filtered CSV (default: output_dir/filtered_TIMESTAMP.csv)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.csv_file).exists():
        print(f"Error: CSV file not found: {args.csv_file}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize API client
    api_client = WeightProcessorAPIClient(args.api_url)

    # Test API connection
    print(f"Testing API connection to {args.api_url}...")
    health_response = api_client.health_check()
    if not health_response.is_success:
        print(f"Error: API health check failed: {health_response.error}")
        return 1

    print(f"✓ API is healthy: {health_response.data}")

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
        "api_url": args.api_url,
        "users_loaded": len(user_measurements),
        "total_measurements": sum(len(m) for m in user_measurements.values()),
        "processing_mode": None,
        "individual_processing": None,
        "replay_processing": None,
        "continuous_replay_processing": None
    }

    # Choose processing mode
    if args.enable_continuous_replay:
        # New continuous replay mode
        print("\n=== Processing with Continuous Replay (New Mode) ===")
        print(f"Replay: {'ENABLED' if not args.disable_replay else 'DISABLED'}")

        overall_results["processing_mode"] = "continuous_replay"
        continuous_results = process_measurements_with_continuous_replay(
            api_client,
            user_measurements,
            acceptance_tracker,
            enable_replay=not args.disable_replay
        )
        overall_results["continuous_replay_processing"] = continuous_results
    else:
        # Legacy two-phase mode
        print("\n=== Legacy Two-Phase Processing ===")
        overall_results["processing_mode"] = "two_phase_legacy"

        # Individual measurement processing
        if not args.skip_individual:
            individual_results = process_individual_measurements(
                api_client,
                user_measurements,
                acceptance_tracker,
                batch_size=args.batch_size
            )
            overall_results["individual_processing"] = individual_results

        # Replay batch processing
        if not args.skip_replay and not args.disable_replay:
            replay_results = process_replay_batches(api_client, user_measurements, acceptance_tracker)
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
    results_file = output_dir / f"api_processing_results_{timestamp_str}.json"

    with open(results_file, 'w') as f:
        json.dump(overall_results, f, indent=2, default=str)

    print("\n=== Processing Complete ===")
    print(f"Duration: {overall_results['duration_seconds']:.1f} seconds")
    print(f"Results saved to: {results_file}")
    print(f"Filtered CSV saved to: {filtered_csv_path}")

    # Print summary statistics
    print(f"\nProcessing Mode: {overall_results['processing_mode']}")

    if overall_results["continuous_replay_processing"]:
        # Continuous replay mode statistics
        continuous_stats = overall_results["continuous_replay_processing"]
        total_processed = sum(r["measurements_processed"] for r in continuous_stats.values())
        total_accepted = sum(r["measurements_accepted"] for r in continuous_stats.values())
        total_replays = sum(r.get("replays_triggered", 0) for r in continuous_stats.values())
        total_corrections = sum(r.get("total_corrections", 0) for r in continuous_stats.values())

        print(f"\nContinuous Replay Statistics:")
        print(f"  Measurements processed: {total_processed:,}")
        print(f"  Measurements accepted: {total_accepted:,}")
        print(f"  Replays triggered: {total_replays:,}")
        print(f"  Total corrections: {total_corrections:,}")
    else:
        # Legacy two-phase mode statistics
        if overall_results["individual_processing"]:
            individual_stats = overall_results["individual_processing"]
            total_processed = sum(r["measurements_processed"] for r in individual_stats.values())
            total_accepted = sum(r["measurements_accepted"] for r in individual_stats.values())
            print(f"\nIndividual processing: {total_processed:,} processed, {total_accepted:,} accepted")

        if overall_results["replay_processing"]:
            replay_stats = overall_results["replay_processing"]
            successful_replays = sum(1 for r in replay_stats.values() if r["success"])
            print(f"Replay processing: {successful_replays:,}/{len(replay_stats):,} successful")

    print(f"\nFiltered output: {accepted_count:,} accepted measurements written")

    return 0


if __name__ == "__main__":
    exit(main())

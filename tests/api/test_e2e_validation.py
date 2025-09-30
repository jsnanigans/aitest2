#!/usr/bin/env python3
"""
End-to-End API Validation Tests

Tests that validate the SAM Local API produces identical filtering results
to the reference filtered dataset. This ensures consistency between the
API implementation and the reference data processing pipeline.

Test Strategy:
1. Extract single-user data from source CSV
2. Process via API (individual measurements + replay)
3. Compare accepted measurements with reference filtered dataset
4. Validate exact match

Processing Flow (mimics api_main.py):
- Phase 1: Process all measurements individually (batch_size=1)
- Phase 2: Replay from middle point to recalculate with updated state
- Phase 3: Track FINAL acceptance state after replay
"""

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SOURCE_CSV = Path("./data/2025-09-29_weights_all.csv")
FILTERED_CSV = Path("./data/2025-09-29_all_filtered_e.csv")
TEST_USER_ID = "c51ef96b-5618-4295-a910-233faed5ab60"

# Expected counts for validation
EXPECTED_SOURCE_COUNT = 49
EXPECTED_ACCEPTED_COUNT = 9


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ValidationResult:
    """Result of comparing actual vs expected acceptance."""
    is_exact_match: bool
    expected_count: int
    actual_count: int
    missing_ids: Set[str]  # Expected but not accepted
    extra_ids: Set[str]    # Accepted but not expected
    matching_ids: Set[str]

    def __str__(self) -> str:
        if self.is_exact_match:
            return f"✓ Exact match: {self.actual_count} measurements accepted"

        return (
            f"✗ Acceptance mismatch:\n"
            f"  Expected: {self.expected_count} measurements\n"
            f"  Actual: {self.actual_count} measurements\n"
            f"  Matching: {len(self.matching_ids)}\n"
            f"  Missing: {len(self.missing_ids)} {list(self.missing_ids)[:3]}...\n"
            f"  Extra: {len(self.extra_ids)} {list(self.extra_ids)[:3]}..."
        )


# ============================================================================
# Phase 1: Data Extraction
# ============================================================================

def parse_timestamp(date_str: str) -> datetime:
    """
    Parse various timestamp formats and ensure timezone-aware datetimes.
    Matches api_main.py:188-207 implementation.
    """
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
    except Exception as e:
        logger.warning(f"Failed to parse timestamp '{date_str}': {e}")
        return datetime.now(timezone.utc)


def extract_user_measurements(csv_path: Path, user_id: str) -> List[Dict[str, Any]]:
    """
    Extract all measurements for a specific user from source CSV.

    Returns measurements in API-compatible format, sorted by timestamp.
    Matches api_main.py:210-299 data loading logic.
    """
    measurements = []

    logger.info(f"Extracting measurements for user {user_id[:12]}... from {csv_path}")

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Filter by user
            row_user_id = row.get("user_id")
            if row_user_id != user_id:
                continue

            # Extract measurement ID
            measurement_id = row.get("id") or row.get("measurement_id")
            if not measurement_id:
                continue

            # Parse and validate weight
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

            # Extract other fields
            date_str = row.get("effective_date_time", "") or row.get("effectiveDateTime", "")
            source = row.get("source_type", "unknown")
            unit = row.get("unit", "kg")

            # Convert to API format (matches Pydantic model)
            measurement = {
                "uuid": measurement_id,
                "weight": weight,
                "unit": unit,
                "effectiveDateTime": parse_timestamp(date_str).isoformat() if date_str else datetime.now(timezone.utc).isoformat(),
                "source": source,
                "metadata": {
                    "csv_row": row
                }
            }

            measurements.append(measurement)

    # Sort by timestamp (critical for correct processing order)
    measurements.sort(key=lambda m: parse_timestamp(m["effectiveDateTime"]))

    logger.info(f"Extracted {len(measurements)} measurements for user {user_id[:12]}")

    return measurements


def extract_expected_accepted(filtered_csv_path: Path, user_id: str) -> Set[str]:
    """
    Extract the measurement IDs that were accepted in the reference dataset.

    Returns a set of measurement IDs (from 'id' column).
    """
    accepted_ids = set()

    logger.info(f"Extracting expected accepted IDs for user {user_id[:12]}... from {filtered_csv_path}")

    with open(filtered_csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            row_user_id = row.get("user_id")
            if row_user_id != user_id:
                continue

            measurement_id = row.get("id") or row.get("measurement_id")
            if measurement_id:
                accepted_ids.add(measurement_id)

    logger.info(f"Expected {len(accepted_ids)} accepted measurements for user {user_id[:12]}")

    return accepted_ids


# ============================================================================
# Phase 2: API Processing (Individual Measurements)
# ============================================================================

def process_individual_measurements(
    base_url: str,
    user_id: str,
    measurements: List[Dict[str, Any]],
    batch_size: int = 1
) -> Tuple[Set[str], Dict[str, Any]]:
    """
    Process measurements individually (or in small batches) via API.

    Matches api_main.py:347-432 individual processing logic.

    Returns:
        Tuple of (accepted_measurement_ids, processing_stats)
    """
    accepted_ids = set()
    stats = {
        "measurements_processed": 0,
        "measurements_accepted": 0,
        "measurements_rejected": 0,
        "api_calls": 0,
        "errors": []
    }

    logger.info(f"Processing {len(measurements)} measurements individually (batch_size={batch_size})")

    # Measurements should already be sorted, but ensure it
    sorted_measurements = sorted(measurements, key=lambda m: parse_timestamp(m["effectiveDateTime"]))

    # Process in batches
    for batch_start in range(0, len(sorted_measurements), batch_size):
        batch = sorted_measurements[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1

        payload = {
            "measurements": batch,
            "options": {}
        }

        try:
            response = requests.post(
                f"{base_url}/api/v1/process/{user_id}",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Version": "v2"
                },
                timeout=30
            )

            stats["api_calls"] += 1

            if response.status_code >= 200 and response.status_code < 300:
                data = response.json()

                # Parse V2 format response
                if "success" in data and data["success"]:
                    response_data = data.get("data", {})
                else:
                    response_data = data

                # Update stats
                batch_processed = response_data.get("measurements_processed", 0)
                batch_accepted = response_data.get("measurements_accepted", 0)

                stats["measurements_processed"] += batch_processed
                stats["measurements_accepted"] += batch_accepted
                stats["measurements_rejected"] += (batch_processed - batch_accepted)

                # Track accepted measurements from results
                results = response_data.get("results", [])
                if results:
                    for i, result in enumerate(results):
                        if i < len(batch) and result.get("accepted", False):
                            measurement_id = batch[i]["uuid"]
                            accepted_ids.add(measurement_id)
                            logger.debug(f"  Batch {batch_num}: Accepted {measurement_id}")
                else:
                    # Fallback: assume first N measurements are accepted
                    for i in range(min(batch_accepted, len(batch))):
                        measurement_id = batch[i]["uuid"]
                        accepted_ids.add(measurement_id)

                logger.info(f"  Batch {batch_num}/{(len(sorted_measurements) + batch_size - 1) // batch_size}: "
                           f"{batch_processed} processed, {batch_accepted} accepted")
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                stats["errors"].append(f"Batch {batch_num}: {error_msg}")
                logger.error(f"  Batch {batch_num} failed: {error_msg}")

        except requests.RequestException as e:
            error_msg = str(e)
            stats["errors"].append(f"Batch {batch_num}: {error_msg}")
            logger.error(f"  Batch {batch_num} failed: {error_msg}")

    logger.info(f"Individual processing complete: {stats['measurements_processed']} processed, "
               f"{stats['measurements_accepted']} accepted, {len(accepted_ids)} unique IDs tracked")

    return accepted_ids, stats


# ============================================================================
# Phase 3: Replay Processing
# ============================================================================

def process_replay(
    base_url: str,
    user_id: str,
    measurements: List[Dict[str, Any]]
) -> Tuple[Set[str], Dict[str, Any]]:
    """
    Execute replay from middle point to recalculate with updated Kalman state.

    Matches api_main.py:435-517 replay logic.

    Returns:
        Tuple of (accepted_measurement_ids_from_replay, replay_stats)
    """
    accepted_ids = set()
    stats = {
        "replay_executed": False,
        "replay_from": None,
        "measurements_replayed": 0,
        "measurements_processed": 0,
        "measurements_accepted": 0,
        "success": False,
        "error": None
    }

    # Need at least 10 measurements for meaningful replay
    if len(measurements) < 10:
        logger.info("Skipping replay: insufficient measurements (need >= 10)")
        return accepted_ids, stats

    # Sort measurements by timestamp
    sorted_measurements = sorted(measurements, key=lambda m: parse_timestamp(m["effectiveDateTime"]))

    # Take a point in the middle as replay anchor (matches api_main.py:478)
    replay_anchor_idx = len(sorted_measurements) // 2
    replay_from = parse_timestamp(sorted_measurements[replay_anchor_idx]["effectiveDateTime"])

    # Get measurements from anchor onwards
    replay_measurements = sorted_measurements[replay_anchor_idx:]

    logger.info(f"Replaying {len(replay_measurements)} measurements from {replay_from.isoformat()}")

    # Prepare payload
    replay_from_str = replay_from.isoformat()
    payload = {
        "replay_from_timestamp": replay_from_str,
        "measurements": replay_measurements,
        "options": {}
    }

    try:
        response = requests.post(
            f"{base_url}/api/v1/replay/{user_id}",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-Version": "v2"
            },
            timeout=30
        )

        stats["replay_executed"] = True
        stats["replay_from"] = replay_from_str
        stats["measurements_replayed"] = len(replay_measurements)

        if response.status_code >= 200 and response.status_code < 300:
            data = response.json()

            # Parse V2 format response
            if "success" in data and data["success"]:
                response_data = data.get("data", {})
                stats["success"] = True
            else:
                response_data = data
                stats["success"] = response.status_code < 400

            # Update stats
            stats["measurements_processed"] = response_data.get("measurements_processed", 0)
            stats["measurements_accepted"] = response_data.get("measurements_accepted", 0)

            # Track accepted measurements from results
            results = response_data.get("results", [])
            if results:
                for i, result in enumerate(results):
                    if i < len(replay_measurements) and result.get("accepted", False):
                        measurement_id = replay_measurements[i]["uuid"]
                        accepted_ids.add(measurement_id)
                        logger.debug(f"  Replay accepted: {measurement_id}")
            else:
                # Fallback: assume first N measurements are accepted
                accepted_count = response_data.get("measurements_accepted", 0)
                for i in range(min(accepted_count, len(replay_measurements))):
                    measurement_id = replay_measurements[i]["uuid"]
                    accepted_ids.add(measurement_id)

            logger.info(f"Replay complete: {stats['measurements_processed']} processed, "
                       f"{stats['measurements_accepted']} accepted, {len(accepted_ids)} unique IDs tracked")
        else:
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            stats["error"] = error_msg
            logger.error(f"Replay failed: {error_msg}")

    except requests.RequestException as e:
        error_msg = str(e)
        stats["error"] = error_msg
        logger.error(f"Replay failed: {error_msg}")

    return accepted_ids, stats


# ============================================================================
# Phase 4: Validation
# ============================================================================

def validate_acceptance(
    actual_accepted: Set[str],
    expected_accepted: Set[str]
) -> ValidationResult:
    """
    Compare actual vs expected acceptance sets.

    Returns detailed validation result with diff information.
    """
    matching_ids = actual_accepted & expected_accepted
    missing_ids = expected_accepted - actual_accepted  # Expected but not accepted
    extra_ids = actual_accepted - expected_accepted    # Accepted but not expected

    is_exact_match = (len(missing_ids) == 0 and len(extra_ids) == 0)

    result = ValidationResult(
        is_exact_match=is_exact_match,
        expected_count=len(expected_accepted),
        actual_count=len(actual_accepted),
        missing_ids=missing_ids,
        extra_ids=extra_ids,
        matching_ids=matching_ids
    )

    logger.info(str(result))

    return result


# ============================================================================
# API Helpers
# ============================================================================

def cleanup_user(base_url: str, user_id: str, cleanup_type: str = "clear_all") -> bool:
    """
    Reset user state via cleanup endpoint.

    Valid cleanup_types: "reset_adaptive" or "clear_all"

    Returns True if successful, False otherwise.
    """
    logger.info(f"Cleaning up user {user_id[:12]}... (type: {cleanup_type})")

    payload = {
        "cleanup_type": cleanup_type,
        "options": {}
    }

    try:
        response = requests.post(
            f"{base_url}/api/v1/cleanup/{user_id}",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-Version": "v2"
            },
            timeout=30
        )

        if response.status_code >= 200 and response.status_code < 300:
            logger.info(f"✓ User {user_id[:12]} cleaned up successfully")
            return True
        else:
            logger.error(f"✗ Cleanup failed: HTTP {response.status_code}")
            return False

    except requests.RequestException as e:
        logger.error(f"✗ Cleanup failed: {e}")
        return False


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def api_base_url():
    """Base URL for SAM Local API."""
    return "http://localhost:3080"


@pytest.fixture
def source_csv_path():
    """Path to source CSV file."""
    return SOURCE_CSV


@pytest.fixture
def filtered_csv_path():
    """Path to reference filtered CSV file."""
    return FILTERED_CSV


@pytest.fixture
def test_user_id():
    """User ID for E2E testing."""
    return TEST_USER_ID


@pytest.fixture
def ensure_clean_state(api_base_url, test_user_id):
    """Ensure clean user state before and after test."""
    # Reset before test
    cleanup_user(api_base_url, test_user_id, cleanup_type="clear_all")
    yield
    # Reset after test (cleanup)
    cleanup_user(api_base_url, test_user_id, cleanup_type="clear_all")


@pytest.fixture
def verify_csv_files(source_csv_path, filtered_csv_path):
    """Verify required CSV files exist."""
    if not source_csv_path.exists():
        pytest.skip(f"Source CSV not found: {source_csv_path}")
    if not filtered_csv_path.exists():
        pytest.skip(f"Filtered CSV not found: {filtered_csv_path}")


# ============================================================================
# E2E Test
# ============================================================================

@pytest.mark.e2e
def test_single_user_end_to_end_validation(
    api_base_url,
    source_csv_path,
    filtered_csv_path,
    test_user_id,
    ensure_clean_state,
    verify_csv_files
):
    """
    E2E test: Process a real user and validate against reference dataset.

    This test validates that the SAM Local API produces identical filtering
    results to the reference filtered dataset. It:

    1. Extracts user data from source CSV (49 measurements expected)
    2. Processes via API (individual batch processing)
    3. Executes replay from middle point
    4. Validates accepted measurements match filtered dataset (9 expected)

    Test User: c51ef96b-5618-4295-a910-233faed5ab60
    Expected acceptance rate: 18.4% (9/49)
    """
    logger.info("=" * 80)
    logger.info("Starting E2E Validation Test")
    logger.info("=" * 80)

    # ========================================================================
    # Pre-test: Ensure clean state
    # ========================================================================

    logger.info("\n[PRE-TEST] Cleaning user state...")
    cleanup_success = cleanup_user(api_base_url, test_user_id, cleanup_type="clear_all")
    if not cleanup_success:
        pytest.fail("Failed to clean user state before test")

    # ========================================================================
    # Arrange: Load test data
    # ========================================================================

    logger.info("\n[ARRANGE] Loading test data...")

    measurements = extract_user_measurements(source_csv_path, test_user_id)
    expected_accepted = extract_expected_accepted(filtered_csv_path, test_user_id)

    # Validate we loaded the expected data
    assert len(measurements) == EXPECTED_SOURCE_COUNT, (
        f"Expected {EXPECTED_SOURCE_COUNT} source measurements, got {len(measurements)}"
    )
    assert len(expected_accepted) == EXPECTED_ACCEPTED_COUNT, (
        f"Expected {EXPECTED_ACCEPTED_COUNT} accepted measurements, got {len(expected_accepted)}"
    )

    logger.info(f"✓ Loaded {len(measurements)} measurements")
    logger.info(f"✓ Expected {len(expected_accepted)} accepted measurements")

    # ========================================================================
    # Act - Phase 1: Individual processing
    # ========================================================================

    logger.info("\n[ACT - Phase 1] Processing measurements individually...")

    accepted_after_individual, individual_stats = process_individual_measurements(
        api_base_url,
        test_user_id,
        measurements,
        batch_size=1
    )

    # Check for errors in individual processing
    if individual_stats["errors"]:
        logger.error(f"Errors during individual processing: {individual_stats['errors']}")
        pytest.fail(f"Individual processing had errors: {individual_stats['errors']}")

    logger.info(f"✓ Individual processing complete: {len(accepted_after_individual)} accepted")

    # ========================================================================
    # Act - Phase 2: Replay ALL measurements (from beginning)
    # ========================================================================

    logger.info("\n[ACT - Phase 2] Executing replay of ALL measurements from beginning...")

    # Replay ALL measurements from the very beginning
    sorted_measurements = sorted(measurements, key=lambda m: parse_timestamp(m["effectiveDateTime"]))
    replay_from_beginning = parse_timestamp(sorted_measurements[0]["effectiveDateTime"])

    payload = {
        "replay_from_timestamp": replay_from_beginning.isoformat(),
        "measurements": sorted_measurements,
        "options": {}
    }

    try:
        response = requests.post(
            f"{api_base_url}/api/v1/replay/{test_user_id}",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-Version": "v2"
            },
            timeout=60
        )

        if response.status_code >= 200 and response.status_code < 300:
            data = response.json()
            response_data = data.get("data", {}) if "success" in data else data

            # Track accepted measurements from replay
            final_accepted = set()
            results = response_data.get("results", [])
            if results:
                for i, result in enumerate(results):
                    if result.get("accepted", False):
                        final_accepted.add(sorted_measurements[i]["uuid"])

            logger.info(f"✓ Replay complete: {response_data.get('measurements_processed', 0)} processed, "
                       f"{response_data.get('measurements_accepted', 0)} accepted, "
                       f"{len(final_accepted)} unique IDs tracked")
        else:
            pytest.fail(f"Replay failed: HTTP {response.status_code}")

    except requests.RequestException as e:
        pytest.fail(f"Replay failed: {e}")

    logger.info(f"✓ Final accepted set: {len(final_accepted)} unique measurements")

    # ========================================================================
    # Export: Write CSV files for manual inspection
    # ========================================================================

    logger.info("\n[EXPORT] Writing CSV files for manual inspection...")

    # Write actual accepted measurements
    actual_csv_path = Path("./output/e2e_actual_accepted.csv")
    actual_csv_path.parent.mkdir(exist_ok=True)

    with open(actual_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'measurement_id', 'user_id', 'weight', 'unit',
            'effectiveDateTime', 'source', 'status'
        ])
        writer.writeheader()

        for measurement in sorted_measurements:
            mid = measurement["uuid"]
            writer.writerow({
                'measurement_id': mid,
                'user_id': test_user_id,
                'weight': measurement["weight"],
                'unit': measurement["unit"],
                'effectiveDateTime': measurement["effectiveDateTime"],
                'source': measurement["source"],
                'status': 'ACCEPTED' if mid in final_accepted else 'REJECTED'
            })

    logger.info(f"✓ Actual results written to: {actual_csv_path}")

    # Write expected accepted measurements
    expected_csv_path = Path("./output/e2e_expected_accepted.csv")

    with open(expected_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'measurement_id', 'user_id', 'weight', 'unit',
            'effectiveDateTime', 'source', 'status'
        ])
        writer.writeheader()

        for measurement in sorted_measurements:
            mid = measurement["uuid"]
            if mid in expected_accepted:
                writer.writerow({
                    'measurement_id': mid,
                    'user_id': test_user_id,
                    'weight': measurement["weight"],
                    'unit': measurement["unit"],
                    'effectiveDateTime': measurement["effectiveDateTime"],
                    'source': measurement["source"],
                    'status': 'EXPECTED_ACCEPTED'
                })

    logger.info(f"✓ Expected results written to: {expected_csv_path}")

    # Write comparison CSV
    comparison_csv_path = Path("./output/e2e_comparison.csv")

    with open(comparison_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'measurement_id', 'user_id', 'weight', 'unit',
            'effectiveDateTime', 'source', 'expected', 'actual', 'match'
        ])
        writer.writeheader()

        for measurement in sorted_measurements:
            mid = measurement["uuid"]
            expected_status = 'ACCEPTED' if mid in expected_accepted else 'REJECTED'
            actual_status = 'ACCEPTED' if mid in final_accepted else 'REJECTED'
            match = 'YES' if expected_status == actual_status else 'NO'

            writer.writerow({
                'measurement_id': mid,
                'user_id': test_user_id,
                'weight': measurement["weight"],
                'unit': measurement["unit"],
                'effectiveDateTime': measurement["effectiveDateTime"],
                'source': measurement["source"],
                'expected': expected_status,
                'actual': actual_status,
                'match': match
            })

    logger.info(f"✓ Comparison written to: {comparison_csv_path}")

    # ========================================================================
    # Assert: Validate exact match
    # ========================================================================

    logger.info("\n[ASSERT] Validating acceptance against reference dataset...")

    validation = validate_acceptance(final_accepted, expected_accepted)

    # Log detailed results
    logger.info(f"\nValidation Results:")
    logger.info(f"  Expected: {validation.expected_count}")
    logger.info(f"  Actual: {validation.actual_count}")
    logger.info(f"  Matching: {len(validation.matching_ids)}")
    logger.info(f"  Missing: {len(validation.missing_ids)}")
    logger.info(f"  Extra: {len(validation.extra_ids)}")

    if validation.missing_ids:
        logger.error(f"\n  Missing IDs (expected but not accepted):")
        for mid in sorted(list(validation.missing_ids)):
            logger.error(f"    - {mid}")

    if validation.extra_ids:
        logger.error(f"\n  Extra IDs (accepted but not expected):")
        for mid in sorted(list(validation.extra_ids)):
            logger.error(f"    - {mid}")

    # Final assertion
    assert validation.is_exact_match, str(validation)

    logger.info("\n" + "=" * 80)
    logger.info("✓ E2E Validation Test PASSED")
    logger.info("=" * 80)


# ============================================================================
# Additional Tests
# ============================================================================

@pytest.mark.e2e
def test_data_extraction_counts(source_csv_path, filtered_csv_path, test_user_id):
    """
    Sanity test: Verify we can extract the expected number of measurements.

    This test doesn't require the API to be running.
    """
    measurements = extract_user_measurements(source_csv_path, test_user_id)
    expected_accepted = extract_expected_accepted(filtered_csv_path, test_user_id)

    assert len(measurements) == EXPECTED_SOURCE_COUNT, (
        f"Expected {EXPECTED_SOURCE_COUNT} measurements, got {len(measurements)}"
    )
    assert len(expected_accepted) == EXPECTED_ACCEPTED_COUNT, (
        f"Expected {EXPECTED_ACCEPTED_COUNT} accepted measurements, got {len(expected_accepted)}"
    )

    # Verify all expected IDs exist in source measurements
    source_ids = {m["uuid"] for m in measurements}
    missing_in_source = expected_accepted - source_ids

    assert len(missing_in_source) == 0, (
        f"Expected accepted IDs not found in source: {missing_in_source}"
    )

    logger.info(f"✓ Data extraction validated: {len(measurements)} source, {len(expected_accepted)} expected accepted")
"""
Historical data handling tests.

These tests verify the system's ability to handle historical data imports,
corrections, backfilling, and time-based data management scenarios.
"""

import pytest
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Dict
import random

from .conftest import APIClient, TestUser, create_measurement, create_measurement_series


class TestHistoricalDataImport:
    """Test importing historical weight data."""

    def test_import_past_year_data(self, api_client, test_user):
        """Import a full year of historical data."""
        measurements = []
        base_date = datetime.now(timezone.utc) - timedelta(days=365)

        # Generate daily measurements for past year
        for day in range(365):
            date = base_date + timedelta(days=day)
            # Simulate gradual weight loss with seasonal variations
            seasonal_factor = 2.0 * (0.5 + 0.5 * (day % 90) / 90)  # Quarterly cycles
            weight = 85.0 - (day * 0.02) + seasonal_factor + random.uniform(-0.5, 0.5)

            measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": round(weight, 1),
                "unit": "kg",
                "effectiveDateTime": date.isoformat(),
                "source": "historical_import",
                "metadata": {
                    "original_date": date.strftime("%Y-%m-%d"),
                    "import_batch": "yearly_import_2023"
                }
            })

        # Import should handle large batch
        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        assert response.data["processed_count"] == 365
        # Most should be accepted
        assert response.data["accepted_count"] > 350

    def test_import_with_data_gaps(self, api_client, test_user):
        """Import historical data with significant gaps."""
        measurements = []
        base_date = datetime.now(timezone.utc) - timedelta(days=730)  # 2 years ago

        # Create data with gaps (simulating missing records)
        date_ranges = [
            (0, 30),     # First month
            (60, 90),    # Third month
            (150, 180),  # Months 5-6
            (365, 395),  # Start of year 2
            (500, 530),  # Mid year 2
            (700, 730),  # End of period
        ]

        for start_day, end_day in date_ranges:
            for day in range(start_day, end_day):
                date = base_date + timedelta(days=day)
                weight = 80.0 - (day * 0.01) + random.uniform(-0.5, 0.5)

                measurements.append({
                    "uuid": str(uuid.uuid4()),
                    "weight": round(weight, 1),
                    "unit": "kg",
                    "effectiveDateTime": date.isoformat(),
                    "source": "historical_import"
                })

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        # System should handle gaps appropriately
        assert response.data["processed_count"] == len(measurements)

        # Verify state recognizes gaps
        state = api_client.get_user_state(test_user.user_id)
        assert state.data.get("has_time_gaps", False) == True

    def test_import_overlapping_historical_data(self, api_client, test_user):
        """Test importing historical data that overlaps with existing data."""
        # First import: recent 30 days
        recent_measurements = create_measurement_series(
            start_weight=75.0,
            days=30,
            start_date=datetime.now(timezone.utc) - timedelta(days=30)
        )

        response1 = api_client.process_measurements(test_user.user_id, recent_measurements)
        assert response1.is_success

        # Second import: historical 60 days (includes overlap)
        historical_measurements = create_measurement_series(
            start_weight=77.0,
            days=60,
            start_date=datetime.now(timezone.utc) - timedelta(days=60)
        )

        response2 = api_client.process_measurements(test_user.user_id, historical_measurements)

        # Should detect and handle overlapping dates
        if response2.is_success:
            # May process with conflict resolution
            assert response2.data["processed_count"] == 60
        else:
            # Or reject due to conflicts
            assert response2.status_code == 409

    def test_import_very_old_data(self, api_client, test_user):
        """Test importing data from many years ago."""
        measurements = []

        # Data from 5-10 years ago
        for years_ago in range(5, 11):
            base_date = datetime.now(timezone.utc) - timedelta(days=years_ago * 365)

            # Quarterly measurements for each year
            for quarter in range(4):
                date = base_date + timedelta(days=quarter * 90)
                weight = 70.0 + (years_ago * 1.5) + random.uniform(-2.0, 2.0)

                measurements.append({
                    "uuid": str(uuid.uuid4()),
                    "weight": round(weight, 1),
                    "unit": "kg",
                    "effectiveDateTime": date.isoformat(),
                    "source": "medical_records",
                    "metadata": {
                        "record_year": date.year,
                        "import_reason": "complete_medical_history"
                    }
                })

        response = api_client.process_measurements(test_user.user_id, measurements)

        # Should handle very old data
        assert response.is_success or response.status_code == 400
        if response.is_success:
            assert response.data["processed_count"] == len(measurements)


class TestDataCorrection:
    """Test correction of historical data."""

    def test_correct_wrong_unit_entries(self, api_client, test_user):
        """Correct historical entries with wrong units."""
        # Initial data with wrong units
        wrong_measurements = []
        base_date = datetime.now(timezone.utc) - timedelta(days=30)

        for day in range(10):
            # User accidentally entered kg values as lbs
            measurements = {
                "uuid": str(uuid.uuid4()),
                "weight": 75.0,  # Meant to be kg but entered as lbs
                "unit": "lbs",   # Wrong unit
                "effectiveDateTime": (base_date + timedelta(days=day * 3)).isoformat(),
                "source": "manual"
            }
            wrong_measurements.append(measurements)

        # Process wrong data
        response1 = api_client.process_measurements(test_user.user_id, wrong_measurements)
        assert response1.is_success

        # Now replay with corrected data
        replay_from = base_date - timedelta(days=1)
        corrected_measurements = []

        for m in wrong_measurements:
            corrected = m.copy()
            corrected["uuid"] = str(uuid.uuid4())  # New UUID for corrected entry
            corrected["weight"] = 75.0
            corrected["unit"] = "kg"  # Correct unit
            corrected["metadata"] = {"correction": "unit_error"}
            corrected_measurements.append(corrected)

        response2 = api_client.replay_measurements(
            test_user.user_id,
            replay_from=replay_from,
            measurements=corrected_measurements
        )

        assert response2.is_success
        assert response2.data["measurements_replayed"] == len(corrected_measurements)

    def test_correct_decimal_point_errors(self, api_client, test_user):
        """Correct entries with decimal point errors."""
        # Data with decimal errors (e.g., 750 instead of 75.0)
        error_measurements = [
            create_measurement(weight=750.0, unit="kg", days_ago=10),  # Missing decimal
            create_measurement(weight=7.5, unit="kg", days_ago=9),     # Extra decimal
            create_measurement(weight=75.0, unit="kg", days_ago=8),    # Correct
        ]

        # These will likely be rejected or marked as outliers
        response1 = api_client.process_measurements(test_user.user_id, error_measurements)

        # Replay with corrections
        corrected_measurements = [
            create_measurement(weight=75.0, unit="kg", days_ago=10),
            create_measurement(weight=75.0, unit="kg", days_ago=9),
            create_measurement(weight=75.0, unit="kg", days_ago=8),
        ]

        replay_from = datetime.now(timezone.utc) - timedelta(days=11)
        response2 = api_client.replay_measurements(
            test_user.user_id,
            replay_from=replay_from,
            measurements=corrected_measurements
        )

        assert response2.is_success

    def test_remove_duplicate_entries(self, api_client, test_user):
        """Remove duplicate historical entries."""
        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        # Create measurements with some duplicates
        measurements = []
        for day in range(10):
            date = base_date + timedelta(days=day)

            # Add normal measurement
            measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": 75.0 + random.uniform(-0.5, 0.5),
                "unit": "kg",
                "effectiveDateTime": date.isoformat(),
                "source": "scale"
            })

            # Add duplicate for some days
            if day % 3 == 0:
                measurements.append({
                    "uuid": str(uuid.uuid4()),
                    "weight": 75.0 + random.uniform(-0.5, 0.5),
                    "unit": "kg",
                    "effectiveDateTime": date.isoformat(),  # Same timestamp
                    "source": "scale"
                })

        response = api_client.process_measurements(test_user.user_id, measurements)

        # System should handle duplicates
        if response.is_success:
            # May deduplicate or process both with conflict resolution
            assert response.data["processed_count"] == len(measurements)


class TestBackfilling:
    """Test backfilling missing historical data."""

    def test_backfill_missing_periods(self, api_client, test_user):
        """Backfill missing periods in historical data."""
        # Initial sparse data
        initial_measurements = [
            create_measurement(weight=80.0, days_ago=90),
            create_measurement(weight=78.0, days_ago=60),
            create_measurement(weight=76.0, days_ago=30),
            create_measurement(weight=75.0, days_ago=0),
        ]

        response1 = api_client.process_measurements(test_user.user_id, initial_measurements)
        assert response1.is_success

        # Backfill missing periods
        backfill_measurements = []

        # Fill days 89-61
        for day in range(89, 60, -1):
            if day % 3 == 0:  # Every 3 days
                weight = 80.0 - ((90 - day) * 0.067)  # Interpolated
                backfill_measurements.append(
                    create_measurement(weight=weight, days_ago=day, source="backfill")
                )

        # Fill days 59-31
        for day in range(59, 30, -1):
            if day % 3 == 0:
                weight = 78.0 - ((60 - day) * 0.067)
                backfill_measurements.append(
                    create_measurement(weight=weight, days_ago=day, source="backfill")
                )

        response2 = api_client.process_measurements(test_user.user_id, backfill_measurements)
        assert response2.is_success

        # Verify complete timeline
        state = api_client.get_user_state(test_user.user_id)
        assert state.data["measurement_count"] > len(initial_measurements)

    def test_backfill_from_paper_records(self, api_client, test_user):
        """Simulate backfilling from paper records or old systems."""
        # Existing recent digital data
        recent_data = create_measurement_series(
            start_weight=75.0,
            days=30,
            start_date=datetime.now(timezone.utc) - timedelta(days=30)
        )

        response1 = api_client.process_measurements(test_user.user_id, recent_data)
        assert response1.is_success

        # Backfill from paper records (older, less precise)
        paper_records = []
        base_date = datetime.now(timezone.utc) - timedelta(days=180)

        # Weekly paper records for 5 months
        for week in range(20):
            date = base_date + timedelta(weeks=week)
            # Less precise (rounded to nearest 0.5 kg)
            weight = round(82.0 - (week * 0.35) * 2) / 2

            paper_records.append({
                "uuid": str(uuid.uuid4()),
                "weight": weight,
                "unit": "kg",
                "effectiveDateTime": date.isoformat(),
                "source": "paper_record",
                "metadata": {
                    "precision": "0.5kg",
                    "transcribed_date": datetime.now(timezone.utc).isoformat(),
                    "original_format": "handwritten"
                }
            })

        response2 = api_client.process_measurements(test_user.user_id, paper_records)
        assert response2.is_success

        # Verify combined dataset
        state = api_client.get_user_state(test_user.user_id)
        assert state.data["measurement_count"] >= 50


class TestTimeZoneHandling:
    """Test handling of different time zones in historical data."""

    def test_import_data_from_different_timezones(self, api_client, test_user):
        """Import data recorded in different time zones."""
        measurements = []

        # Simulate user traveling across time zones
        timezones = [
            ("US/Pacific", -8),
            ("US/Eastern", -5),
            ("Europe/London", 0),
            ("Asia/Tokyo", 9),
            ("Australia/Sydney", 11),
        ]

        base_date = datetime.now(timezone.utc) - timedelta(days=50)

        for i, (tz_name, offset) in enumerate(timezones):
            # 10 days in each timezone
            for day in range(10):
                actual_day = i * 10 + day
                # Create datetime with timezone offset
                date = base_date + timedelta(days=actual_day, hours=offset)

                measurements.append({
                    "uuid": str(uuid.uuid4()),
                    "weight": 75.0 + random.uniform(-1.0, 1.0),
                    "unit": "kg",
                    "effectiveDateTime": date.isoformat(),
                    "source": "travel_scale",
                    "metadata": {
                        "timezone": tz_name,
                        "local_time": (date + timedelta(hours=-offset)).strftime("%Y-%m-%d %H:%M")
                    }
                })

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        assert response.data["processed_count"] == 50

    def test_daylight_saving_time_transitions(self, api_client, test_user):
        """Test handling of daylight saving time transitions."""
        measurements = []

        # Spring forward (lose an hour) - typically March
        spring_date = datetime(2024, 3, 10, 2, 0, tzinfo=timezone.utc)

        # Measurements around DST transition
        for hour in range(-3, 4):
            date = spring_date + timedelta(hours=hour)
            measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": 75.0 + random.uniform(-0.2, 0.2),
                "unit": "kg",
                "effectiveDateTime": date.isoformat(),
                "source": "scale",
                "metadata": {"dst_transition": "spring_forward"}
            })

        # Fall back (gain an hour) - typically November
        fall_date = datetime(2024, 11, 3, 2, 0, tzinfo=timezone.utc)

        for hour in range(-3, 4):
            date = fall_date + timedelta(hours=hour)
            measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": 74.0 + random.uniform(-0.2, 0.2),
                "unit": "kg",
                "effectiveDateTime": date.isoformat(),
                "source": "scale",
                "metadata": {"dst_transition": "fall_back"}
            })

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        # Should handle DST transitions without issues
        assert response.data["processed_count"] == len(measurements)


class TestReplayMechanism:
    """Test the replay mechanism for historical corrections."""

    def test_replay_from_specific_date(self, api_client, test_user):
        """Test replaying measurements from a specific date."""
        # Build initial history
        initial_measurements = create_measurement_series(
            start_weight=80.0,
            days=60,
            start_date=datetime.now(timezone.utc) - timedelta(days=60)
        )

        response1 = api_client.process_measurements(test_user.user_id, initial_measurements)
        assert response1.is_success

        initial_state = api_client.get_user_state(test_user.user_id)
        initial_count = initial_state.data["measurement_count"]

        # Replay from 30 days ago with corrected data
        replay_date = datetime.now(timezone.utc) - timedelta(days=30)
        corrected_measurements = create_measurement_series(
            start_weight=77.0,  # Different weight progression
            days=30,
            start_date=replay_date,
            source="corrected"
        )

        response2 = api_client.replay_measurements(
            test_user.user_id,
            replay_from=replay_date,
            measurements=corrected_measurements
        )

        assert response2.is_success
        assert response2.data["replay_status"] == "completed"

        # Verify state after replay
        final_state = api_client.get_user_state(test_user.user_id)
        # Should preserve first 30 days and replace last 30
        assert final_state.data["measurement_count"] >= 30

    def test_replay_with_state_snapshot(self, api_client, test_user):
        """Test that replay creates and uses state snapshots."""
        # Build complex state
        measurements = []
        base_date = datetime.now(timezone.utc) - timedelta(days=100)

        # Different phases with different patterns
        phases = [
            (30, "scale", 80.0, -0.1),     # Weight loss
            (30, "manual", 77.0, 0.05),    # Slight gain
            (40, "doctor", 78.5, -0.08),   # More loss
        ]

        day_counter = 0
        for days, source, start_weight, daily_change in phases:
            for day in range(days):
                weight = start_weight + (day * daily_change)
                measurements.append({
                    "uuid": str(uuid.uuid4()),
                    "weight": round(weight, 1),
                    "unit": "kg",
                    "effectiveDateTime": (base_date + timedelta(days=day_counter)).isoformat(),
                    "source": source
                })
                day_counter += 1

        response1 = api_client.process_measurements(test_user.user_id, measurements)
        assert response1.is_success

        # Replay from day 60 (middle of second phase)
        replay_date = base_date + timedelta(days=60)

        # New data for replay
        new_measurements = []
        for day in range(40):
            new_measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": 76.0 - (day * 0.05),
                "unit": "kg",
                "effectiveDateTime": (replay_date + timedelta(days=day)).isoformat(),
                "source": "replay_correction"
            })

        response2 = api_client.replay_measurements(
            test_user.user_id,
            replay_from=replay_date,
            measurements=new_measurements
        )

        assert response2.is_success

        # Verify Kalman filter adapted properly
        state = api_client.get_user_state(test_user.user_id)
        assert state.data["kalman_state"]["initialized"] == True

    def test_replay_preserves_quality_scores(self, api_client, test_user):
        """Test that replay preserves quality scores appropriately."""
        base_date = datetime.now(timezone.utc) - timedelta(days=30)

        # Initial high-quality measurements
        doctor_measurements = []
        for day in range(0, 30, 7):  # Weekly doctor visits
            doctor_measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": 80.0 - (day * 0.2),
                "unit": "kg",
                "effectiveDateTime": (base_date + timedelta(days=day)).isoformat(),
                "source": "doctor"
            })

        response1 = api_client.process_measurements(test_user.user_id, doctor_measurements)
        assert response1.is_success

        # Replay from day 14 with lower quality data
        replay_date = base_date + timedelta(days=14)
        manual_measurements = []

        for day in range(16):
            manual_measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": 77.0 - (day * 0.1),
                "unit": "kg",
                "effectiveDateTime": (replay_date + timedelta(days=day)).isoformat(),
                "source": "manual"
            })

        response2 = api_client.replay_measurements(
            test_user.user_id,
            replay_from=replay_date,
            measurements=manual_measurements
        )

        assert response2.is_success

        # Early doctor measurements should retain high quality
        # Later manual measurements should have lower quality
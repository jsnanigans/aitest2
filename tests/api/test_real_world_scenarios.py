"""
Real-world scenario tests simulating actual user behavior patterns.

These tests demonstrate how the API handles realistic use cases that
backend implementations will encounter in production environments.
"""

import pytest
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict

from .conftest import APIClient, TestUser, create_measurement


class TestWeightLossJourneys:
    """Test realistic weight loss journeys."""

    def test_successful_gradual_weight_loss(self, api_client, test_user):
        """Simulate a successful 12-week weight loss program."""
        start_weight = 90.0
        target_loss_per_week = 0.5  # kg per week (healthy rate)
        weeks = 12

        measurements = []
        current_date = datetime.utcnow() - timedelta(weeks=weeks)

        for week in range(weeks):
            for day in range(7):
                # Expected weight with daily variation
                expected_weight = start_weight - (week * target_loss_per_week)
                daily_variation = random.uniform(-0.5, 0.5)

                # Morning weight (usually lower)
                morning_weight = expected_weight + daily_variation - 0.3

                measurements.append({
                    "uuid": str(uuid.uuid4()),
                    "weight": round(morning_weight, 1),
                    "unit": "kg",
                    "effectiveDateTime": (current_date + timedelta(weeks=week, days=day, hours=6)).isoformat() + "Z",
                    "source": "scale"
                })

                # Some days have evening measurements too (usually higher)
                if day % 3 == 0:
                    evening_weight = expected_weight + daily_variation + 0.5
                    measurements.append({
                        "uuid": str(uuid.uuid4()),
                        "weight": round(evening_weight, 1),
                        "unit": "kg",
                        "effectiveDateTime": (current_date + timedelta(weeks=week, days=day, hours=20)).isoformat() + "Z",
                        "source": "scale"
                    })

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        assert response.data["accepted_count"] > len(measurements) * 0.9  # Most accepted

        # Verify weight trend
        state = api_client.get_user_state(test_user.user_id)
        assert state.data["statistics"]["weight_trend"] == "decreasing"
        assert state.data["statistics"]["average_weight"] < start_weight

    def test_weight_loss_with_plateaus(self, api_client, test_user):
        """Simulate weight loss with realistic plateaus."""
        measurements = []
        current_weight = 85.0
        base_date = datetime.utcnow() - timedelta(days=90)

        phases = [
            (14, -0.15),  # 2 weeks loss
            (14, 0.0),    # 2 weeks plateau
            (14, -0.12),  # 2 weeks loss
            (21, 0.02),   # 3 weeks slight gain/plateau
            (14, -0.18),  # 2 weeks loss
            (13, 0.0),    # ~2 weeks plateau
        ]

        day_counter = 0
        for days, daily_change in phases:
            for day in range(days):
                current_weight += daily_change
                # Add realistic daily variation
                measured_weight = current_weight + random.uniform(-0.4, 0.4)

                measurements.append({
                    "uuid": str(uuid.uuid4()),
                    "weight": round(measured_weight, 1),
                    "unit": "kg",
                    "effectiveDateTime": (base_date + timedelta(days=day_counter)).isoformat() + "Z",
                    "source": "scale" if day % 2 == 0 else "app"
                })
                day_counter += 1

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        # System should handle plateaus without marking as outliers
        assert response.data["accepted_count"] > len(measurements) * 0.85

    def test_yoyo_dieting_pattern(self, api_client, test_user):
        """Simulate yo-yo dieting pattern (weight cycling)."""
        measurements = []
        base_date = datetime.utcnow() - timedelta(days=180)  # 6 months

        # 3 cycles of loss and regain
        for cycle in range(3):
            cycle_start = base_date + timedelta(days=cycle * 60)

            # Rapid loss phase (4 weeks)
            for day in range(28):
                weight = 80.0 - (day * 0.3)  # Aggressive loss
                measurements.append({
                    "uuid": str(uuid.uuid4()),
                    "weight": round(weight + random.uniform(-0.3, 0.3), 1),
                    "unit": "kg",
                    "effectiveDateTime": (cycle_start + timedelta(days=day)).isoformat() + "Z",
                    "source": "scale"
                })

            # Regain phase (4 weeks)
            for day in range(28, 56):
                weight = 71.6 + ((day - 28) * 0.35)  # Faster regain
                measurements.append({
                    "uuid": str(uuid.uuid4()),
                    "weight": round(weight + random.uniform(-0.3, 0.3), 1),
                    "unit": "kg",
                    "effectiveDateTime": (cycle_start + timedelta(days=day)).isoformat() + "Z",
                    "source": "manual"  # Less reliable during regain
                })

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        # System should recognize pattern without excessive outlier marking
        assert response.data["processed_count"] == len(measurements)


class TestMultiSourceIntegration:
    """Test integration of measurements from multiple sources."""

    def test_healthcare_provider_integration(self, api_client, test_user):
        """Simulate integration with healthcare providers and devices."""
        measurements = []
        base_date = datetime.utcnow() - timedelta(days=90)

        # Weekly doctor visits (highly reliable)
        for week in range(13):
            measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": 75.0 - (week * 0.2) + random.uniform(-0.1, 0.1),
                "unit": "kg",
                "effectiveDateTime": (base_date + timedelta(weeks=week, hours=10)).isoformat() + "Z",
                "source": "doctor",
                "metadata": {
                    "provider": "Dr. Smith",
                    "location": "Primary Care Clinic"
                }
            })

        # Daily home scale (reliable)
        for day in range(90):
            if day % 7 != 0:  # Skip doctor visit days
                measurements.append({
                    "uuid": str(uuid.uuid4()),
                    "weight": 75.0 - (day * 0.03) + random.uniform(-0.5, 0.5),
                    "unit": "kg",
                    "effectiveDateTime": (base_date + timedelta(days=day, hours=7)).isoformat() + "Z",
                    "source": "scale"
                })

        # Occasional manual entries (less reliable)
        for i in range(15):
            day = random.randint(0, 89)
            measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": 75.0 - (day * 0.03) + random.uniform(-1.0, 1.0),
                "unit": "kg",
                "effectiveDateTime": (base_date + timedelta(days=day, hours=15)).isoformat() + "Z",
                "source": "manual"
            })

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success

        # Doctor measurements should have highest quality
        doctor_measurements = [m for m in response.data["measurements"] if m["source"] == "doctor"]
        assert all(m["quality_score"] > 0.9 for m in doctor_measurements)
        assert all(m["status"] == "accepted" for m in doctor_measurements)

    def test_fitness_tracker_integration(self, api_client, test_user):
        """Simulate integration with various fitness trackers."""
        measurements = []
        base_date = datetime.utcnow() - timedelta(days=30)

        # Different devices with different patterns
        devices = [
            ("fitbit", 0.8, 1),      # Daily
            ("garmin", 0.85, 1),     # Daily
            ("withings", 0.85, 2),   # Every other day
            ("apple_health", 0.75, 3), # Every 3 days
        ]

        for device, quality, interval in devices:
            for day in range(0, 30, interval):
                measurements.append({
                    "uuid": str(uuid.uuid4()),
                    "weight": 75.0 + random.uniform(-1.0, 1.0),
                    "unit": "kg",
                    "effectiveDateTime": (base_date + timedelta(days=day, hours=8)).isoformat() + "Z",
                    "source": device,
                    "metadata": {
                        "device_id": f"{device}_{uuid.uuid4().hex[:8]}",
                        "sync_time": datetime.utcnow().isoformat() + "Z"
                    }
                })

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        assert response.data["processed_count"] == len(measurements)

        # Verify quality scores align with device reliability
        for measurement in response.data["measurements"]:
            device_name = measurement["source"]
            expected_quality = next((q for d, q, _ in devices if d == device_name), 0.5)
            assert abs(measurement["quality_score"] - expected_quality) < 0.2

    def test_mixed_unit_sources(self, api_client, test_user):
        """Test handling of different sources using different units."""
        measurements = []
        base_date = datetime.utcnow() - timedelta(days=14)

        # US-based app (pounds)
        for day in range(14):
            measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": 165.0 - (day * 0.5),  # ~75kg losing weight
                "unit": "lbs",
                "effectiveDateTime": (base_date + timedelta(days=day, hours=6)).isoformat() + "Z",
                "source": "myfitnesspal"
            })

        # UK-based scale (stones)
        for day in range(0, 14, 2):
            measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": 11.8 - (day * 0.02),  # ~75kg in stones
                "unit": "st",
                "effectiveDateTime": (base_date + timedelta(days=day, hours=7)).isoformat() + "Z",
                "source": "uk_scale"
            })

        # European doctor (kilograms)
        for day in [0, 7, 14]:
            measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": 75.0 - (day * 0.15),
                "unit": "kg",
                "effectiveDateTime": (base_date + timedelta(days=day, hours=10)).isoformat() + "Z",
                "source": "doctor"
            })

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        # All units should be properly converted and processed
        assert response.data["accepted_count"] >= len(measurements) * 0.9


class TestRealWorldDataPatterns:
    """Test patterns seen in real-world data."""

    def test_sporadic_measurement_pattern(self, api_client, test_user):
        """Test users who measure weight sporadically."""
        measurements = []
        base_date = datetime.utcnow() - timedelta(days=365)

        # Sporadic measurement days over a year
        measurement_days = sorted(random.sample(range(365), 50))  # 50 random days

        current_weight = 80.0
        for day in measurement_days:
            # Weight changes slowly over time
            weight_change = (day / 365) * -5.0  # 5kg loss over year
            current_weight = 80.0 + weight_change + random.uniform(-1.0, 1.0)

            measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": round(current_weight, 1),
                "unit": "kg",
                "effectiveDateTime": (base_date + timedelta(days=day)).isoformat() + "Z",
                "source": "manual"
            })

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        # System should handle irregular intervals
        assert response.data["accepted_count"] >= len(measurements) * 0.8

    def test_morning_evening_pattern(self, api_client, test_user):
        """Test users who weigh themselves morning and evening."""
        measurements = []
        base_date = datetime.utcnow() - timedelta(days=7)

        for day in range(7):
            base_weight = 75.0 - (day * 0.05)  # Slight loss trend

            # Morning weight (typically lower)
            morning_weight = base_weight - 0.5 + random.uniform(-0.2, 0.2)
            measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": round(morning_weight, 1),
                "unit": "kg",
                "effectiveDateTime": (base_date + timedelta(days=day, hours=6, minutes=30)).isoformat() + "Z",
                "source": "scale",
                "metadata": {"time_of_day": "morning"}
            })

            # Evening weight (typically higher)
            evening_weight = base_weight + 0.8 + random.uniform(-0.2, 0.2)
            measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": round(evening_weight, 1),
                "unit": "kg",
                "effectiveDateTime": (base_date + timedelta(days=day, hours=21, minutes=0)).isoformat() + "Z",
                "source": "scale",
                "metadata": {"time_of_day": "evening"}
            })

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        # Both morning and evening should be accepted despite daily variation
        assert response.data["accepted_count"] == len(measurements)

    def test_post_surgery_recovery_pattern(self, api_client, test_user):
        """Test weight pattern during post-surgery recovery."""
        measurements = []
        base_date = datetime.utcnow() - timedelta(days=60)

        phases = [
            # Pre-surgery stable
            (7, 75.0, 0.0, "scale"),
            # Post-surgery rapid loss (fluid loss, reduced intake)
            (7, 75.0, -0.5, "hospital"),
            # Hospital recovery with IV fluids (rapid gain)
            (7, 71.5, 0.4, "hospital"),
            # Home recovery gradual normalization
            (39, 74.3, -0.05, "scale"),
        ]

        day_counter = 0
        for days, start_weight, daily_change, source in phases:
            for day in range(days):
                weight = start_weight + (day * daily_change) + random.uniform(-0.3, 0.3)
                measurements.append({
                    "uuid": str(uuid.uuid4()),
                    "weight": round(weight, 1),
                    "unit": "kg",
                    "effectiveDateTime": (base_date + timedelta(days=day_counter)).isoformat() + "Z",
                    "source": source
                })
                day_counter += 1

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        # Should handle medical weight fluctuations
        assert response.data["accepted_count"] >= len(measurements) * 0.85

    def test_pregnancy_weight_pattern(self, api_client, test_user):
        """Test weight pattern during pregnancy (significant expected gain)."""
        measurements = []
        base_date = datetime.utcnow() - timedelta(weeks=40)

        # Pregnancy weight gain pattern (simplified)
        for week in range(40):
            if week < 13:  # First trimester - minimal gain
                expected_gain = week * 0.1
            elif week < 27:  # Second trimester - steady gain
                expected_gain = 1.3 + (week - 13) * 0.35
            else:  # Third trimester - increased gain
                expected_gain = 6.2 + (week - 27) * 0.45

            weight = 65.0 + expected_gain + random.uniform(-0.5, 0.5)

            measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": round(weight, 1),
                "unit": "kg",
                "effectiveDateTime": (base_date + timedelta(weeks=week)).isoformat() + "Z",
                "source": "doctor",
                "metadata": {"pregnancy_week": week}
            })

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        # Should accept significant weight gain in pregnancy context
        assert response.data["accepted_count"] == len(measurements)

    def test_athlete_training_cycle(self, api_client, test_user):
        """Test weight patterns during athletic training cycles."""
        measurements = []
        base_date = datetime.utcnow() - timedelta(days=90)

        # Training phases
        phases = [
            # Bulking phase (muscle gain)
            (30, 75.0, 0.1, "scale"),
            # Maintenance phase
            (30, 78.0, 0.0, "gym_scale"),
            # Cutting phase (fat loss)
            (30, 78.0, -0.15, "scale"),
        ]

        day_counter = 0
        for days, start_weight, daily_change, source in phases:
            for day in range(days):
                # Athletes have less daily variation due to controlled diet
                weight = start_weight + (day * daily_change) + random.uniform(-0.2, 0.2)
                measurements.append({
                    "uuid": str(uuid.uuid4()),
                    "weight": round(weight, 1),
                    "unit": "kg",
                    "effectiveDateTime": (base_date + timedelta(days=day_counter)).isoformat() + "Z",
                    "source": source,
                    "metadata": {
                        "phase": phases[day_counter // 30][0] if day_counter < 90 else "cutting",
                        "body_fat_percentage": 12.0 + random.uniform(-1, 1)
                    }
                })
                day_counter += 1

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        assert response.data["accepted_count"] >= len(measurements) * 0.95


class TestEdgeCaseUserBehaviors:
    """Test edge case user behaviors."""

    def test_obsessive_weighing_pattern(self, api_client, test_user):
        """Test users who weigh themselves multiple times per day."""
        measurements = []
        base_date = datetime.utcnow() - timedelta(days=1)

        # User weighs themselves 10+ times in one day
        times = [6, 8, 10, 12, 14, 16, 18, 20, 22, 23]  # Hours of day
        base_weight = 75.0

        for hour in times:
            # Weight varies throughout day (food, hydration, etc.)
            if hour < 12:
                weight_variation = -0.5 + (hour - 6) * 0.1
            else:
                weight_variation = 0.3 + (hour - 12) * 0.05

            weight = base_weight + weight_variation + random.uniform(-0.1, 0.1)

            measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": round(weight, 1),
                "unit": "kg",
                "effectiveDateTime": (base_date + timedelta(hours=hour)).isoformat() + "Z",
                "source": "scale"
            })

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        # Should handle multiple daily measurements
        assert response.data["processed_count"] == len(measurements)

    def test_historical_data_import(self, api_client, test_user):
        """Test importing years of historical weight data."""
        measurements = []
        base_date = datetime.utcnow() - timedelta(days=1095)  # 3 years ago

        # Generate 3 years of weekly measurements
        for week in range(156):  # 3 years * 52 weeks
            weight = 85.0 - (week * 0.05) + random.uniform(-1.0, 1.0)  # Gradual loss
            weight = max(65.0, min(85.0, weight))  # Keep in realistic range

            measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": round(weight, 1),
                "unit": "kg",
                "effectiveDateTime": (base_date + timedelta(weeks=week)).isoformat() + "Z",
                "source": "historical_import",
                "metadata": {
                    "import_date": datetime.utcnow().isoformat() + "Z",
                    "original_source": "paper_records"
                }
            })

        # Process in chunks to simulate batch import
        chunk_size = 52  # 1 year at a time
        for i in range(0, len(measurements), chunk_size):
            chunk = measurements[i:i + chunk_size]
            response = api_client.process_measurements(test_user.user_id, chunk)
            assert response.is_success

        # Verify all historical data was processed
        state = api_client.get_user_state(test_user.user_id)
        assert state.data["measurement_count"] >= 150
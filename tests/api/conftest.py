"""Pytest configuration and fixtures for API tests."""

import os
import json
import uuid
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass
import pytest
import requests
from unittest.mock import Mock, patch


# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:3080")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))


@dataclass
class TestUser:
    """Test user data."""
    user_id: str
    height_cm: float = 175.0
    age: int = 35
    gender: str = "male"
    baseline_weight_kg: float = 75.0


@dataclass
class APIResponse:
    """Wrapper for API responses."""
    status_code: int
    data: Dict[str, Any]
    success: bool
    error: Optional[Dict[str, Any]] = None
    raw_response: Optional[requests.Response] = None

    @property
    def is_success(self) -> bool:
        return self.success and 200 <= self.status_code < 300

    @property
    def measurements_processed(self) -> int:
        if self.data:
            return self.data.get("measurements_processed", 0)
        return 0

    @property
    def measurements_accepted(self) -> int:
        if self.data:
            return self.data.get("measurements_accepted", 0)
        return 0


class APIClient:
    """API client for testing."""

    def __init__(self, base_url: str = API_BASE_URL):
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
                    error=data.get("error"),
                    raw_response=response
                )
            else:
                # Fallback for non-standard responses
                return APIResponse(
                    status_code=response.status_code,
                    data=data,
                    success=response.status_code < 400,
                    raw_response=response
                )
        except json.JSONDecodeError:
            return APIResponse(
                status_code=response.status_code,
                data={},
                success=False,
                error={"message": response.text},
                raw_response=response
            )

    def health_check(self) -> APIResponse:
        """Check API health."""
        response = self.session.get(f"{self.base_url}/api/v1/health")
        return self._parse_response(response)

    def process_measurements(
        self,
        user_id: str,
        measurements: List[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """Process weight measurements."""
        payload = {
            "measurements": measurements,
            "options": options or {}
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/process/{user_id}",
            json=payload,
            timeout=API_TIMEOUT
        )
        return self._parse_response(response)

    def get_user_state(self, user_id: str) -> APIResponse:
        """Get user state."""
        response = self.session.get(
            f"{self.base_url}/api/v1/state/{user_id}",
            timeout=API_TIMEOUT
        )
        return self._parse_response(response)

    def delete_user_state(self, user_id: str) -> APIResponse:
        """Delete user state."""
        response = self.session.delete(
            f"{self.base_url}/api/v1/state/{user_id}",
            timeout=API_TIMEOUT
        )
        return self._parse_response(response)

    def replay_measurements(
        self,
        user_id: str,
        replay_from: datetime,
        measurements: List[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """Replay measurements from a specific timestamp."""
        payload = {
            "replay_from_timestamp": replay_from.isoformat() + "Z",
            "measurements": measurements,
            "options": options or {}
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/replay/{user_id}",
            json=payload,
            timeout=API_TIMEOUT
        )
        return self._parse_response(response)

    def cleanup_user(
        self,
        user_id: str,
        cleanup_type: str = "reset_adaptive",
        options: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """Cleanup user data."""
        payload = {
            "cleanup_type": cleanup_type,
            "options": options or {}
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/cleanup/{user_id}",
            json=payload,
            timeout=API_TIMEOUT
        )
        return self._parse_response(response)


# Fixtures

@pytest.fixture
def api_client() -> APIClient:
    """Create API client."""
    return APIClient()


@pytest.fixture
def test_user() -> TestUser:
    """Create unique test user."""
    return TestUser(
        user_id=f"test-user-{uuid.uuid4().hex[:8]}-{int(time.time())}"
    )


@pytest.fixture
def test_users() -> List[TestUser]:
    """Create multiple test users."""
    return [
        TestUser(
            user_id=f"test-user-{i}-{uuid.uuid4().hex[:8]}",
            baseline_weight_kg=70.0 + i * 5,
            height_cm=165.0 + i * 5,
            age=25 + i * 5
        )
        for i in range(5)
    ]


@pytest.fixture
def cleanup_users(api_client: APIClient) -> Generator:
    """Cleanup fixture to delete user states after tests."""
    users_to_cleanup = []

    yield users_to_cleanup

    # Cleanup after test
    for user_id in users_to_cleanup:
        try:
            api_client.delete_user_state(user_id)
        except Exception:
            pass  # Ignore cleanup errors


@pytest.fixture
def create_measurement() -> callable:
    """Factory for creating measurements."""
    def _create(
        weight: float = 75.0,
        unit: str = "kg",
        source: str = "patient-device",
        effective_datetime: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {
            "uuid": str(uuid.uuid4()),
            "weight": weight,
            "unit": unit,
            "effectiveDateTime": (effective_datetime or datetime.utcnow()).isoformat() + "Z",
            "source": source,
            "metadata": metadata or {}
        }
    return _create


@pytest.fixture
def create_measurement_series() -> callable:
    """Factory for creating a series of measurements."""
    def _create_series(
        start_weight: float = 75.0,
        days: int = 30,
        daily_change: float = -0.1,  # Weight loss
        noise: float = 0.5,  # Daily variation
        source: str = "patient-device",
        start_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        measurements = []
        start_date = start_date or datetime.utcnow() - timedelta(days=days)

        for day in range(days):
            weight = start_weight + (day * daily_change) + random.uniform(-noise, noise)
            measurement_date = start_date + timedelta(days=day)

            measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": round(weight, 1),
                "unit": "kg",
                "effectiveDateTime": measurement_date.isoformat() + "Z",
                "source": source,
                "metadata": {"day": day}
            })

        return measurements
    return _create_series


@pytest.fixture
def weight_sources() -> List[str]:
    """List of realistic weight measurement sources."""
    return [
        "patient-device",
        "care-team-upload",
        "patient-upload",
        "questionnaire",
        "connectivehealth.io",
        "iglucose.com",
        "clinic-scale",
        "home-scale",
        "gym-scale"
    ]


@pytest.fixture
def realistic_weights() -> Dict[str, List[float]]:
    """Realistic weight ranges for different user profiles."""
    return {
        "underweight": [45.0, 48.0, 50.0, 52.0],
        "normal": [60.0, 65.0, 70.0, 75.0, 80.0],
        "overweight": [85.0, 90.0, 95.0, 100.0],
        "obese": [105.0, 110.0, 120.0, 130.0],
        "athlete": [70.0, 75.0, 80.0, 85.0],  # More muscle mass
        "elderly": [55.0, 60.0, 65.0, 70.0],  # Typically lighter
    }


@pytest.fixture
def mock_time() -> callable:
    """Mock time for testing time-sensitive operations."""
    def _mock_time(target_time: datetime):
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = target_time
            mock_datetime.utcnow.return_value = target_time
            yield mock_datetime
    return _mock_time


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed for reproducible tests."""
    random.seed(42)
    yield
    random.seed()  # Reset to default


# Markers
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "load: marks tests as load/performance tests"
    )
    config.addinivalue_line(
        "markers", "edge_case: marks tests for edge cases"
    )
    config.addinivalue_line(
        "markers", "real_world: marks tests simulating real-world scenarios"
    )
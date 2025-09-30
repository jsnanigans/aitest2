"""Abstract base class for state storage."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime


class StateStore(ABC):
    """Abstract interface for state storage backends."""

    @abstractmethod
    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve state for a user."""
        pass

    @abstractmethod
    def save_state(self, user_id: str, state: Dict[str, Any]) -> bool:
        """Save state for a user."""
        pass

    @abstractmethod
    def delete_state(self, user_id: str) -> bool:
        """Delete state for a user."""
        pass

    @abstractmethod
    def create_initial_state(self) -> Dict[str, Any]:
        """Create an empty initial state."""
        pass

    @abstractmethod
    def save_state_snapshot(self, user_id: str, timestamp: datetime) -> bool:
        """Save a snapshot of current state."""
        pass

    @abstractmethod
    def restore_state_snapshot(self, user_id: str) -> bool:
        """Restore state from snapshot."""
        pass

    @abstractmethod
    def get_snapshot(
        self, user_id: str, timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """Get the nearest snapshot before the given timestamp."""
        pass

    @abstractmethod
    def get_latest_snapshot(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent snapshot for a user."""
        pass

    @abstractmethod
    def get_measurements_in_window(
        self, user_id: str, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get measurements for a user within a time window."""
        pass

    @abstractmethod
    def check_and_restore_snapshot(
        self, user_id: str, buffer_start_time: datetime
    ) -> dict:
        """Check if a snapshot exists and restore it atomically."""
        pass

    @abstractmethod
    def export_to_csv(self, filepath: str) -> int:
        """Export all states to CSV."""
        pass

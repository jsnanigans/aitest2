"""Tests for InMemoryStore implementation."""

import pytest
from datetime import datetime, timezone, timedelta
from weight_processor_lib.core.database import InMemoryStore


class TestInMemoryStore:
    """Test InMemoryStore basic operations."""

    @pytest.fixture
    def store(self):
        """Create a fresh store for each test."""
        return InMemoryStore()

    @pytest.fixture
    def user_id(self):
        """Test user ID."""
        return "test_user_123"

    @pytest.fixture
    def sample_state(self, store):
        """Create a sample state."""
        return store.create_initial_state()

    def test_create_initial_state(self, store):
        """Test that create_initial_state returns expected structure."""
        state = store.create_initial_state()

        assert isinstance(state, dict)
        assert state["kalman_params"] is None
        assert state["last_state"] is None
        assert state["last_covariance"] is None
        assert state["last_timestamp"] is None
        assert state["measurement_history"] == []
        assert state["measurements_since_reset"] == 0
        assert state["version"] == 0

    def test_save_and_get_state(self, store, user_id, sample_state):
        """Test saving and retrieving state."""
        # Initially no state
        assert store.get_state(user_id) is None

        # Save state
        assert store.save_state(user_id, sample_state) is True

        # Retrieve state
        retrieved = store.get_state(user_id)
        assert retrieved is not None
        assert retrieved == sample_state

    def test_state_isolation(self, store, sample_state):
        """Test that states for different users are isolated."""
        user1 = "user1"
        user2 = "user2"

        state1 = sample_state.copy()
        state1["version"] = 1

        state2 = sample_state.copy()
        state2["version"] = 2

        store.save_state(user1, state1)
        store.save_state(user2, state2)

        # Each user should have their own state
        retrieved1 = store.get_state(user1)
        retrieved2 = store.get_state(user2)

        assert retrieved1["version"] == 1
        assert retrieved2["version"] == 2

    def test_delete_state(self, store, user_id, sample_state):
        """Test deleting state."""
        # Save state
        store.save_state(user_id, sample_state)
        assert store.get_state(user_id) is not None

        # Delete state
        assert store.delete_state(user_id) is True
        assert store.get_state(user_id) is None

        # Deleting non-existent state returns False
        assert store.delete_state(user_id) is False

    def test_state_deep_copy(self, store, user_id, sample_state):
        """Test that states are deep copied to prevent external modifications."""
        store.save_state(user_id, sample_state)

        # Get state and modify it
        retrieved = store.get_state(user_id)
        retrieved["version"] = 999

        # Original state should be unchanged
        retrieved_again = store.get_state(user_id)
        assert retrieved_again["version"] == 0

    def test_save_state_snapshot(self, store, user_id, sample_state):
        """Test saving snapshots."""
        # Save initial state
        store.save_state(user_id, sample_state)

        # Save snapshot
        timestamp = datetime.now(timezone.utc)
        assert store.save_state_snapshot(user_id, timestamp) is True

        # Snapshot should exist
        assert store.get_snapshot_count(user_id) == 1

    def test_save_snapshot_without_state_fails(self, store, user_id):
        """Test that saving snapshot fails if no current state exists."""
        timestamp = datetime.now(timezone.utc)
        assert store.save_state_snapshot(user_id, timestamp) is False

    def test_restore_latest_snapshot(self, store, user_id, sample_state):
        """Test restoring from latest snapshot."""
        # Save initial state and snapshot
        sample_state["version"] = 1
        store.save_state(user_id, sample_state)
        timestamp1 = datetime.now(timezone.utc)
        store.save_state_snapshot(user_id, timestamp1)

        # Modify state and save another snapshot
        sample_state["version"] = 2
        store.save_state(user_id, sample_state)
        timestamp2 = datetime.now(timezone.utc) + timedelta(seconds=1)
        store.save_state_snapshot(user_id, timestamp2)

        # Further modify current state
        sample_state["version"] = 3
        store.save_state(user_id, sample_state)

        # Current state should be version 3
        assert store.get_state(user_id)["version"] == 3

        # Restore latest snapshot (version 2)
        assert store.restore_state_snapshot(user_id) is True
        assert store.get_state(user_id)["version"] == 2

    def test_get_snapshot_by_timestamp(self, store, user_id, sample_state):
        """Test retrieving snapshot by timestamp."""
        # Create multiple snapshots
        store.save_state(user_id, sample_state)

        timestamp1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        sample_state["version"] = 1
        store.save_state(user_id, sample_state)
        store.save_state_snapshot(user_id, timestamp1)

        timestamp2 = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
        sample_state["version"] = 2
        store.save_state(user_id, sample_state)
        store.save_state_snapshot(user_id, timestamp2)

        timestamp3 = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
        sample_state["version"] = 3
        store.save_state(user_id, sample_state)
        store.save_state_snapshot(user_id, timestamp3)

        # Get snapshot before timestamp2 (should return version 1)
        query_time = datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc)
        snapshot = store.get_snapshot(user_id, query_time)
        assert snapshot is not None
        assert snapshot["version"] == 1

        # Get snapshot at exact timestamp2 (should return version 2)
        snapshot = store.get_snapshot(user_id, timestamp2)
        assert snapshot["version"] == 2

        # Get snapshot after all timestamps (should return version 3)
        query_time = datetime(2024, 1, 1, 15, 0, 0, tzinfo=timezone.utc)
        snapshot = store.get_snapshot(user_id, query_time)
        assert snapshot["version"] == 3

        # Get snapshot before all timestamps (should return None)
        query_time = datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
        snapshot = store.get_snapshot(user_id, query_time)
        assert snapshot is None

    def test_get_latest_snapshot(self, store, user_id, sample_state):
        """Test getting latest snapshot."""
        # No snapshots initially
        assert store.get_latest_snapshot(user_id) is None

        # Create snapshots
        store.save_state(user_id, sample_state)

        timestamp1 = datetime.now(timezone.utc)
        sample_state["version"] = 1
        store.save_state(user_id, sample_state)
        store.save_state_snapshot(user_id, timestamp1)

        timestamp2 = datetime.now(timezone.utc) + timedelta(seconds=1)
        sample_state["version"] = 2
        store.save_state(user_id, sample_state)
        store.save_state_snapshot(user_id, timestamp2)

        # Latest snapshot should be version 2
        latest = store.get_latest_snapshot(user_id)
        assert latest is not None
        assert latest["version"] == 2

    def test_check_and_restore_snapshot(self, store, user_id, sample_state):
        """Test atomic check and restore for replay functionality."""
        # Save state and create snapshot
        sample_state["version"] = 1
        store.save_state(user_id, sample_state)
        snapshot_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        store.save_state_snapshot(user_id, snapshot_time)

        # Update state
        sample_state["version"] = 2
        store.save_state(user_id, sample_state)

        # Check and restore snapshot
        buffer_start = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
        result = store.check_and_restore_snapshot(user_id, buffer_start)

        assert result["snapshot_found"] is True
        assert result["snapshot_restored"] is True
        assert result["snapshot_timestamp"] == snapshot_time

        # State should be restored to version 1
        assert store.get_state(user_id)["version"] == 1

    def test_check_and_restore_no_snapshot(self, store, user_id):
        """Test check_and_restore when no snapshot exists."""
        buffer_start = datetime.now(timezone.utc)
        result = store.check_and_restore_snapshot(user_id, buffer_start)

        assert result["snapshot_found"] is False
        assert result["snapshot_restored"] is False
        assert result["snapshot_timestamp"] is None

    def test_export_to_csv(self, store, sample_state, tmp_path):
        """Test exporting states to CSV."""
        # Save multiple states
        user1 = "user1"
        user2 = "user2"

        state1 = sample_state.copy()
        state1["last_raw_weight"] = 75.5
        state1["measurements_since_reset"] = 10

        state2 = sample_state.copy()
        state2["last_raw_weight"] = 80.2
        state2["measurements_since_reset"] = 5

        store.save_state(user1, state1)
        store.save_state(user2, state2)

        # Export to CSV
        csv_path = tmp_path / "export.csv"
        count = store.export_to_csv(str(csv_path))

        assert count == 2
        assert csv_path.exists()

        # Verify CSV content
        content = csv_path.read_text()
        assert "user1" in content
        assert "user2" in content
        assert "75.5" in content
        assert "80.2" in content

    def test_clear_all(self, store, user_id, sample_state):
        """Test clearing all data."""
        # Add some data
        store.save_state(user_id, sample_state)
        store.save_state_snapshot(user_id, datetime.now(timezone.utc))

        assert store.get_state(user_id) is not None
        assert store.get_snapshot_count(user_id) > 0

        # Clear all
        store.clear_all()

        assert store.get_state(user_id) is None
        assert store.get_snapshot_count(user_id) == 0

    def test_list_users(self, store, sample_state):
        """Test listing all user IDs."""
        # Initially empty
        assert store.list_users() == []

        # Add multiple users
        store.save_state("user1", sample_state)
        store.save_state("user2", sample_state)
        store.save_state("user3", sample_state)

        users = store.list_users()
        assert len(users) == 3
        assert "user1" in users
        assert "user2" in users
        assert "user3" in users

    def test_clear_snapshots(self, store, user_id, sample_state):
        """Test clearing snapshots for a user."""
        # Create multiple snapshots
        store.save_state(user_id, sample_state)
        store.save_state_snapshot(user_id, datetime.now(timezone.utc))
        store.save_state_snapshot(user_id, datetime.now(timezone.utc) + timedelta(seconds=1))
        store.save_state_snapshot(user_id, datetime.now(timezone.utc) + timedelta(seconds=2))

        assert store.get_snapshot_count(user_id) == 3

        # Clear snapshots
        count = store.clear_snapshots(user_id)
        assert count == 3
        assert store.get_snapshot_count(user_id) == 0

        # State should still exist
        assert store.get_state(user_id) is not None

    def test_repr(self, store, sample_state):
        """Test string representation."""
        store.save_state("user1", sample_state)
        store.save_state("user2", sample_state)
        store.save_state_snapshot("user1", datetime.now(timezone.utc))

        repr_str = repr(store)
        assert "InMemoryStore" in repr_str
        assert "states=2" in repr_str
        assert "users_with_snapshots=1" in repr_str

    def test_thread_safety_basic(self, store, sample_state):
        """Basic test that operations use locking (doesn't test actual concurrency)."""
        # This is a basic sanity check - true concurrent testing would require
        # threading.Thread and more complex setup

        # Just verify the lock exists and has the expected interface
        assert hasattr(store, "_lock")
        assert hasattr(store._lock, "acquire")
        assert hasattr(store._lock, "release")

        # Operations should work
        store.save_state("user1", sample_state)
        assert store.get_state("user1") is not None

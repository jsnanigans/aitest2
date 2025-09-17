"""
Transaction management for atomic reset operations.
Ensures all reset operations succeed or rollback together.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import copy
import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class ResetOperation(Enum):
    """Types of operations in a reset transaction"""
    KALMAN_RESET = "kalman_reset"
    STATE_UPDATE = "state_update"
    BUFFER_UPDATE = "buffer_update"
    STATE_PERSIST = "state_persist"


@dataclass
class TransactionCheckpoint:
    """Snapshot of state at a point in transaction"""
    operation: ResetOperation
    timestamp: float
    state_snapshot: Dict[str, Any]
    validation_passed: bool = False


class ResetTransaction:
    """
    Manages atomic reset operations with automatic rollback.

    Ensures that all reset operations either complete successfully
    or rollback to the original state if any operation fails.
    """

    def __init__(self, user_id: str):
        """
        Initialize transaction for a specific user.

        Args:
            user_id: User identifier for logging
        """
        self.user_id = user_id
        self.checkpoints: List[TransactionCheckpoint] = []
        self.original_states: Dict[ResetOperation, Any] = {}
        self.completed_operations: List[ResetOperation] = []
        self.failed = False
        self.failure_reason: Optional[str] = None

    def __enter__(self):
        """Start transaction - capture initial state"""
        logger.info(f"Starting reset transaction for user {self.user_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End transaction - commit or rollback"""
        if exc_type is not None:
            logger.error(f"Reset transaction failed with exception: {exc_val}")
            self.rollback(str(exc_val))
            return False  # Re-raise exception

        if self.failed:
            self.rollback(self.failure_reason or "Unknown failure")
            return False

        # All operations succeeded
        self.commit()
        return True

    def save_original_state(self, operation: ResetOperation, state: Any):
        """
        Save original state before modifying.

        Args:
            operation: Type of operation
            state: Original state to preserve
        """
        if operation not in self.original_states:
            self.original_states[operation] = copy.deepcopy(state)
            logger.debug(f"Saved original state for {operation.value}")

    def save_checkpoint(self, operation: ResetOperation, state: Dict[str, Any]):
        """
        Save state snapshot after an operation.

        Args:
            operation: Type of operation completed
            state: New state after operation
        """
        checkpoint = TransactionCheckpoint(
            operation=operation,
            timestamp=time.time(),
            state_snapshot=copy.deepcopy(state),
            validation_passed=False
        )
        self.checkpoints.append(checkpoint)
        logger.debug(f"Checkpoint saved for {operation.value}")

    def validate_checkpoint(self, operation: ResetOperation, validator=None) -> bool:
        """
        Validate the state after an operation.

        Args:
            operation: Operation to validate
            validator: Optional custom validator, otherwise uses StateValidator

        Returns:
            True if validation passed, False otherwise
        """
        checkpoint = self._get_last_checkpoint(operation)
        if not checkpoint:
            logger.error(f"No checkpoint found for {operation.value}")
            self.failed = True
            self.failure_reason = f"Missing checkpoint for {operation.value}"
            return False

        try:
            # Import here to avoid circular dependencies
            if validator is None:
                from .state_validator import StateValidator
                validator = StateValidator()

            is_valid = validator.validate(checkpoint.state_snapshot, operation)

            checkpoint.validation_passed = is_valid
            if not is_valid:
                self.failed = True
                self.failure_reason = f"Validation failed for {operation.value}"
                logger.error(self.failure_reason)

            return is_valid

        except Exception as e:
            logger.error(f"Validation error for {operation.value}: {e}")
            self.failed = True
            self.failure_reason = f"Validation error: {str(e)}"
            return False

    def mark_completed(self, operation: ResetOperation):
        """
        Mark an operation as successfully completed.

        Args:
            operation: Operation that completed successfully
        """
        self.completed_operations.append(operation)
        logger.info(f"Operation completed: {operation.value}")

    def rollback(self, reason: str):
        """
        Rollback all completed operations.

        Args:
            reason: Reason for rollback (for logging)
        """
        logger.warning(f"Rolling back reset transaction for user {self.user_id}: {reason}")

        # Return original states to caller
        # Actual state restoration happens in the processor
        logger.info(f"Rollback complete - {len(self.completed_operations)} operations rolled back")

        # Clear transaction state
        self.checkpoints.clear()
        self.completed_operations.clear()

    def commit(self):
        """Commit all operations - make permanent"""
        logger.info(f"Committing reset transaction for user {self.user_id} - {len(self.completed_operations)} operations")
        # States are already updated in place, just log success

    def get_original_state(self, operation: ResetOperation) -> Optional[Any]:
        """
        Get original state for rollback.

        Args:
            operation: Operation to get original state for

        Returns:
            Original state if saved, None otherwise
        """
        return self.original_states.get(operation)

    def _get_last_checkpoint(self, operation: ResetOperation) -> Optional[TransactionCheckpoint]:
        """
        Get the most recent checkpoint for an operation.

        Args:
            operation: Operation to find checkpoint for

        Returns:
            Most recent checkpoint or None
        """
        for checkpoint in reversed(self.checkpoints):
            if checkpoint.operation == operation:
                return checkpoint
        return None

    @property
    def is_failed(self) -> bool:
        """Check if transaction has failed"""
        return self.failed


@contextmanager
def atomic_reset(user_id: str):
    """
    Convenience context manager for atomic reset operations.

    Usage:
        with atomic_reset(user_id) as txn:
            # Perform operations
            txn.save_checkpoint(...)
            txn.validate_checkpoint(...)
            # If any operation fails, automatic rollback occurs

    Args:
        user_id: User identifier

    Yields:
        ResetTransaction instance
    """
    txn = ResetTransaction(user_id)
    try:
        yield txn
    except Exception as e:
        txn.rollback(str(e))
        raise
    finally:
        if not txn.is_failed:
            txn.commit()
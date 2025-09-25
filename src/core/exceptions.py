"""
Custom exceptions for the weight processing system.

These exceptions provide clear error signaling for data corruption
and validation failures in the Kalman state management system.
"""


class DataCorruptionError(Exception):
    """
    Raised when data corruption is detected in stored state.

    This indicates that the stored Kalman state has been corrupted
    beyond automatic recovery and requires manual intervention.
    """
    pass


class StateValidationError(Exception):
    """
    Raised when state validation fails.

    This indicates that the Kalman state does not meet expected
    validation criteria (shape, values, completeness).
    """
    pass


class RecoveryFailedError(Exception):
    """
    Raised when automatic recovery attempts fail.

    This indicates that the system attempted to recover from
    corrupted state but was unable to produce a valid result.
    """
    pass
"""
Circuit breaker pattern for preventing cascading failures.
"""

from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitOpenError(Exception):
    """Raised when circuit is open and rejecting calls"""
    pass


class CircuitBreaker:
    """
    Prevents cascading failures by breaking the circuit after repeated failures.

    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are rejected immediately
    - HALF_OPEN: Testing if the system has recovered
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        timeout: int = 60,
        success_threshold: int = 2,
        name: str = "circuit"
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting recovery
            success_threshold: Successes needed in HALF_OPEN to close circuit
            name: Name for logging
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout  # Seconds before attempting recovery
        self.success_threshold = success_threshold
        self.name = name

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_attempt_time: Optional[datetime] = None
        self.last_error: Optional[Exception] = None

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitOpenError: If circuit is open
            Exception: Any exception from func
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
            else:
                time_remaining = self._time_until_recovery()
                raise CircuitOpenError(
                    f"Circuit '{self.name}' open due to {self.failure_count} failures. "
                    f"Retry in {time_remaining:.0f} seconds"
                )

        try:
            # Attempt the call
            self.last_attempt_time = datetime.now()
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure(e)
            raise

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self.last_failure_time is None:
            return True

        elapsed = datetime.now() - self.last_failure_time
        return elapsed.total_seconds() >= self.timeout

    def _time_until_recovery(self) -> float:
        """Calculate seconds until recovery attempt"""
        if self.last_failure_time is None:
            return 0

        elapsed = datetime.now() - self.last_failure_time
        remaining = self.timeout - elapsed.total_seconds()
        return max(0, remaining)

    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.debug(f"Circuit '{self.name}' success in HALF_OPEN: {self.success_count}/{self.success_threshold}")

            if self.success_count >= self.success_threshold:
                logger.info(f"Circuit breaker '{self.name}' recovered, entering CLOSED state")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.last_error = None
        else:
            # Reset failure count on success in CLOSED state
            if self.failure_count > 0:
                logger.debug(f"Circuit '{self.name}' success in CLOSED, resetting failure count")
            self.failure_count = 0

    def _on_failure(self, error: Exception):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.last_error = error

        if self.state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit breaker '{self.name}' recovery failed, reopening circuit")
            self.state = CircuitState.OPEN
            self.success_count = 0

        elif self.failure_count >= self.failure_threshold:
            logger.error(f"Circuit breaker '{self.name}' opening after {self.failure_count} failures: {error}")
            self.state = CircuitState.OPEN

        else:
            logger.warning(f"Circuit '{self.name}' failure {self.failure_count}/{self.failure_threshold}: {error}")

    def reset(self):
        """Manually reset the circuit breaker to closed state"""
        logger.info(f"Circuit breaker '{self.name}' manually reset")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_error = None

    @property
    def is_open(self) -> bool:
        """Check if circuit is currently open"""
        return self.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is currently closed"""
        return self.state == CircuitState.CLOSED

    def get_status(self) -> Dict[str, Any]:
        """
        Get current circuit breaker status.

        Returns:
            Dictionary with status information
        """
        status = {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
        }

        if self.last_failure_time:
            status['last_failure'] = self.last_failure_time.isoformat()
            if self.state == CircuitState.OPEN:
                status['recovery_in'] = self._time_until_recovery()

        if self.last_error:
            status['last_error'] = str(self.last_error)

        return status


class MultiCircuitBreaker:
    """
    Manages multiple circuit breakers for different operations.
    """

    def __init__(self):
        """Initialize multi-circuit breaker manager"""
        self.breakers: Dict[str, CircuitBreaker] = {}

    def add_breaker(
        self,
        name: str,
        failure_threshold: int = 3,
        timeout: int = 60,
        success_threshold: int = 2
    ) -> CircuitBreaker:
        """
        Add a new circuit breaker.

        Args:
            name: Name of the breaker
            failure_threshold: Failures before opening
            timeout: Recovery timeout in seconds
            success_threshold: Successes needed to close

        Returns:
            The created circuit breaker
        """
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            timeout=timeout,
            success_threshold=success_threshold,
            name=name
        )
        self.breakers[name] = breaker
        return breaker

    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name"""
        return self.breakers.get(name)

    def call(self, breaker_name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through named circuit breaker.

        Args:
            breaker_name: Name of breaker to use
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            ValueError: If breaker doesn't exist
            CircuitOpenError: If circuit is open
        """
        breaker = self.breakers.get(breaker_name)
        if not breaker:
            raise ValueError(f"No circuit breaker named '{breaker_name}'")

        return breaker.call(func, *args, **kwargs)

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        return {
            name: breaker.get_status()
            for name, breaker in self.breakers.items()
        }

    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self.breakers.values():
            breaker.reset()
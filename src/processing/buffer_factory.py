"""
Buffer Factory for managing ReplayBuffer instances.

This module provides a factory pattern implementation for creating and managing
ReplayBuffer instances, replacing the previous singleton anti-pattern.
"""

import logging
import warnings
from threading import RLock
from typing import Dict, Optional, Any, Set
from contextlib import contextmanager

from .replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class BufferFactory:
    """
    Factory for creating and managing ReplayBuffer instances.

    Provides controlled instance creation with proper lifecycle management,
    replacing the global singleton pattern for improved testability and
    concurrent instance support.
    """

    def __init__(self):
        """Initialize the BufferFactory."""
        self._instances: Dict[str, ReplayBuffer] = {}
        self._lock = RLock()
        self._default_config: Optional[Dict[str, Any]] = None
        self._instance_refs: Dict[str, int] = {}  # Reference counting

    def create_buffer(
        self,
        name: str = 'default',
        config: Optional[Dict[str, Any]] = None
    ) -> ReplayBuffer:
        """
        Create or retrieve a named buffer instance.

        Args:
            name: Instance identifier (default: 'default')
            config: Configuration dict for the buffer

        Returns:
            ReplayBuffer instance

        Raises:
            ValueError: If instance limit exceeded (max 10 concurrent instances)
        """
        with self._lock:
            # Check instance limit
            if name not in self._instances and len(self._instances) >= 10:
                raise ValueError(
                    f"Maximum buffer instances (10) reached. "
                    f"Active instances: {list(self._instances.keys())}"
                )

            # Create or retrieve instance
            if name not in self._instances:
                effective_config = config or self._default_config
                logger.info(f"Creating new ReplayBuffer instance: {name}")
                self._instances[name] = ReplayBuffer(effective_config)
                self._instance_refs[name] = 0
            else:
                if config and config != self._default_config:
                    logger.warning(
                        f"Buffer '{name}' already exists with different config. "
                        "Using existing instance."
                    )

            # Increment reference count
            self._instance_refs[name] += 1
            return self._instances[name]

    def get_buffer(self, name: str = 'default') -> Optional[ReplayBuffer]:
        """
        Get an existing buffer instance without creating it.

        Args:
            name: Instance identifier

        Returns:
            ReplayBuffer instance if exists, None otherwise
        """
        with self._lock:
            return self._instances.get(name)

    def remove_buffer(self, name: str) -> bool:
        """
        Remove a buffer instance.

        Args:
            name: Instance identifier

        Returns:
            True if removed, False if not found or still referenced
        """
        with self._lock:
            if name not in self._instances:
                return False

            # Check reference count
            if self._instance_refs.get(name, 0) > 0:
                logger.warning(
                    f"Cannot remove buffer '{name}': still has "
                    f"{self._instance_refs[name]} references"
                )
                return False

            # Clean up the buffer
            buffer = self._instances[name]
            if hasattr(buffer, 'cleanup'):
                try:
                    buffer.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up buffer '{name}': {e}")

            # Remove from registry
            del self._instances[name]
            del self._instance_refs[name]
            logger.info(f"Removed buffer instance: {name}")
            return True

    @contextmanager
    def managed_buffer(
        self,
        name: str = 'default',
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for buffer lifecycle management.

        Ensures proper cleanup even on exceptions.

        Args:
            name: Instance identifier
            config: Configuration dict

        Yields:
            ReplayBuffer instance
        """
        buffer = self.create_buffer(name, config)
        try:
            yield buffer
        finally:
            with self._lock:
                # Decrement reference count
                if name in self._instance_refs:
                    self._instance_refs[name] -= 1

                    # Auto-remove if no references and not default
                    if self._instance_refs[name] == 0 and name != 'default':
                        self.remove_buffer(name)

    def list_buffers(self) -> Set[str]:
        """
        List all active buffer instance names.

        Returns:
            Set of instance names
        """
        with self._lock:
            return set(self._instances.keys())

    def clear_all(self, force: bool = False):
        """
        Clear all buffer instances.

        Args:
            force: Force removal even if references exist
        """
        with self._lock:
            if not force:
                # Check for active references
                active_refs = [
                    name for name, count in self._instance_refs.items()
                    if count > 0
                ]
                if active_refs:
                    raise RuntimeError(
                        f"Cannot clear: active references exist for {active_refs}"
                    )

            # Clean up all buffers
            for name, buffer in self._instances.items():
                if hasattr(buffer, 'cleanup'):
                    try:
                        buffer.cleanup()
                    except Exception as e:
                        logger.error(f"Error cleaning up buffer '{name}': {e}")

            self._instances.clear()
            self._instance_refs.clear()
            logger.info("Cleared all buffer instances")

    def set_default_config(self, config: Dict[str, Any]):
        """
        Set default configuration for new buffers.

        Args:
            config: Default configuration dict
        """
        self._default_config = config
        logger.debug(f"Set default buffer config: {config}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get factory statistics.

        Returns:
            Dict with instance counts and reference information
        """
        with self._lock:
            return {
                'total_instances': len(self._instances),
                'instances': list(self._instances.keys()),
                'references': dict(self._instance_refs),
                'has_default_config': self._default_config is not None
            }


# Global factory instance (replacing the buffer singleton)
_factory = BufferFactory()


# Public factory interface
def get_factory() -> BufferFactory:
    """Get the global BufferFactory instance."""
    return _factory


# Backward compatibility wrapper
_deprecation_shown = False

def get_replay_buffer(config: Optional[Dict[str, Any]] = None) -> ReplayBuffer:
    """
    DEPRECATED: Get or create the default replay buffer instance.

    This function maintains backward compatibility but should not be used
    in new code. Use BufferFactory.create_buffer() instead.

    Args:
        config: Configuration dict for the buffer

    Returns:
        Default ReplayBuffer instance
    """
    global _deprecation_shown

    if not _deprecation_shown:
        warnings.warn(
            "get_replay_buffer() is deprecated and will be removed in v2.0. "
            "Use BufferFactory.create_buffer() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        _deprecation_shown = True
        logger.warning(
            "Using deprecated get_replay_buffer(). "
            "Please migrate to BufferFactory."
        )

    factory = get_factory()
    if config:
        factory.set_default_config(config)
    return factory.create_buffer('default', config)


# Migration helper decorator
def with_buffer(name: str = 'default', config: Optional[Dict[str, Any]] = None):
    """
    Decorator for injecting buffer instances into functions.

    Usage:
        @with_buffer('test')
        def my_function(buffer, ...):
            buffer.add_measurement(...)

    Args:
        name: Buffer instance name
        config: Buffer configuration
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            factory = get_factory()
            with factory.managed_buffer(name, config) as buffer:
                return func(buffer, *args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator
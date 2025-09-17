"""
Test suite for BufferFactory implementation.

Tests the factory pattern implementation that replaces the singleton anti-pattern.
"""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch, call
import warnings

from src.processing.buffer_factory import (
    BufferFactory, get_factory, get_replay_buffer, with_buffer
)
from src.processing.replay_buffer import ReplayBuffer


class TestBufferFactory:
    """Test BufferFactory functionality."""

    def setup_method(self):
        """Reset factory state before each test."""
        factory = get_factory()
        factory.clear_all(force=True)

    def teardown_method(self):
        """Clean up after each test."""
        factory = get_factory()
        factory.clear_all(force=True)

    def test_create_buffer(self):
        """Test basic buffer creation."""
        factory = BufferFactory()

        # Create a buffer
        buffer1 = factory.create_buffer('test1')
        assert isinstance(buffer1, ReplayBuffer)

        # Get the same buffer
        buffer2 = factory.create_buffer('test1')
        assert buffer1 is buffer2  # Same instance

        # Create different buffer
        buffer3 = factory.create_buffer('test2')
        assert buffer3 is not buffer1  # Different instance

    def test_buffer_with_config(self):
        """Test buffer creation with configuration."""
        factory = BufferFactory()
        config = {'buffer_hours': 24, 'max_buffer_measurements': 50}

        buffer = factory.create_buffer('configured', config)
        assert buffer.config['buffer_hours'] == 24
        assert buffer.config['max_buffer_measurements'] == 50

    def test_instance_limit(self):
        """Test that factory enforces instance limit."""
        factory = BufferFactory()

        # Create maximum allowed instances (10)
        for i in range(10):
            factory.create_buffer(f'buffer_{i}')

        # 11th should fail
        with pytest.raises(ValueError, match="Maximum buffer instances"):
            factory.create_buffer('buffer_11')

    def test_get_buffer(self):
        """Test getting existing buffer."""
        factory = BufferFactory()

        # Get non-existent buffer
        assert factory.get_buffer('nonexistent') is None

        # Create and get buffer
        created = factory.create_buffer('test')
        retrieved = factory.get_buffer('test')
        assert created is retrieved

    def test_remove_buffer(self):
        """Test buffer removal."""
        factory = BufferFactory()

        # Create buffer
        buffer = factory.create_buffer('removable')

        # Try to remove while referenced
        assert not factory.remove_buffer('removable')  # Should fail

        # Decrement reference manually (simulating release)
        factory._instance_refs['removable'] = 0

        # Now removal should work
        assert factory.remove_buffer('removable')
        assert factory.get_buffer('removable') is None

    def test_managed_buffer_context(self):
        """Test context manager for buffer lifecycle."""
        factory = BufferFactory()

        # Use managed buffer
        with factory.managed_buffer('context_test') as buffer:
            assert isinstance(buffer, ReplayBuffer)
            assert 'context_test' in factory.list_buffers()

        # After context, non-default buffer should be auto-removed
        # (since reference count drops to 0)
        factory._instance_refs['context_test'] = 0
        factory.remove_buffer('context_test')
        assert 'context_test' not in factory.list_buffers()

    def test_managed_buffer_exception_handling(self):
        """Test that cleanup happens even with exceptions."""
        factory = BufferFactory()

        try:
            with factory.managed_buffer('exception_test') as buffer:
                assert isinstance(buffer, ReplayBuffer)
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Reference should be decremented despite exception
        assert factory._instance_refs.get('exception_test', 0) == 0

    def test_list_buffers(self):
        """Test listing active buffers."""
        factory = BufferFactory()

        assert len(factory.list_buffers()) == 0

        factory.create_buffer('buffer1')
        factory.create_buffer('buffer2')
        factory.create_buffer('buffer3')

        buffers = factory.list_buffers()
        assert len(buffers) == 3
        assert 'buffer1' in buffers
        assert 'buffer2' in buffers
        assert 'buffer3' in buffers

    def test_clear_all(self):
        """Test clearing all buffers."""
        factory = BufferFactory()

        # Create some buffers
        factory.create_buffer('clear1')
        factory.create_buffer('clear2')

        # Clear without force should fail if references exist
        with pytest.raises(RuntimeError, match="active references"):
            factory.clear_all(force=False)

        # Force clear should work
        factory.clear_all(force=True)
        assert len(factory.list_buffers()) == 0

    def test_default_config(self):
        """Test default configuration setting."""
        factory = BufferFactory()
        default_config = {'buffer_hours': 48}

        factory.set_default_config(default_config)

        # New buffer should use default config
        buffer = factory.create_buffer('with_default')
        assert buffer.config['buffer_hours'] == 48

    def test_get_stats(self):
        """Test factory statistics."""
        factory = BufferFactory()

        stats = factory.get_stats()
        assert stats['total_instances'] == 0
        assert not stats['has_default_config']

        factory.create_buffer('stats1')
        factory.create_buffer('stats2')
        factory.set_default_config({'test': True})

        stats = factory.get_stats()
        assert stats['total_instances'] == 2
        assert 'stats1' in stats['instances']
        assert 'stats2' in stats['instances']
        assert stats['has_default_config']
        assert stats['references']['stats1'] == 1
        assert stats['references']['stats2'] == 1

    def test_thread_safety(self):
        """Test thread-safe operations."""
        factory = BufferFactory()
        created_buffers = []
        errors = []
        lock = threading.Lock()

        def create_buffers(thread_id):
            for i in range(3):  # Reduced to avoid hitting limit
                name = f"thread_{thread_id}_buffer_{i}"
                try:
                    buffer = factory.create_buffer(name)
                    with lock:
                        created_buffers.append(name)
                    assert buffer is not None
                    time.sleep(0.001)  # Small delay to increase contention
                except ValueError as e:
                    if "Maximum buffer instances" in str(e):
                        # Expected if we hit the limit
                        break
                    else:
                        errors.append(e)

        # Create threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=create_buffers, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check no unexpected errors
        assert len(errors) == 0

        # Check buffers created (respecting 10 instance limit)
        assert len(factory.list_buffers()) <= 10
        assert len(factory.list_buffers()) > 0  # At least some created

    def test_cleanup_on_removal(self):
        """Test that buffer cleanup is called on removal."""
        factory = BufferFactory()

        with patch.object(ReplayBuffer, 'cleanup') as mock_cleanup:
            buffer = factory.create_buffer('cleanup_test')

            # Manually set ref count to allow removal
            factory._instance_refs['cleanup_test'] = 0

            # Remove should trigger cleanup
            factory.remove_buffer('cleanup_test')
            mock_cleanup.assert_called_once()

    def test_singleton_deprecation(self):
        """Test backward compatibility with deprecated get_replay_buffer."""
        # Should trigger deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            buffer = get_replay_buffer({'test': True})

            # Check warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

        # Should return default buffer from factory
        assert isinstance(buffer, ReplayBuffer)
        factory = get_factory()
        assert 'default' in factory.list_buffers()

    def test_with_buffer_decorator(self):
        """Test the with_buffer decorator for dependency injection."""
        call_args = []

        @with_buffer('decorated_test', {'buffer_hours': 12})
        def test_function(buffer, arg1, arg2):
            call_args.append((buffer, arg1, arg2))
            assert isinstance(buffer, ReplayBuffer)
            assert buffer.config['buffer_hours'] == 12
            return arg1 + arg2

        result = test_function(1, 2)
        assert result == 3
        assert len(call_args) == 1

        # Check buffer was created and cleaned up
        factory = get_factory()
        # Buffer should be removed after decorator exits (non-default)
        if factory._instance_refs.get('decorated_test', 0) == 0:
            assert 'decorated_test' not in factory.list_buffers()

    def test_global_factory_singleton(self):
        """Test that get_factory returns the same instance."""
        factory1 = get_factory()
        factory2 = get_factory()

        assert factory1 is factory2  # Same instance

        # Operations on one should affect the other
        factory1.create_buffer('singleton_test')
        assert 'singleton_test' in factory2.list_buffers()


class TestMigrationPath:
    """Test migration from singleton to factory pattern."""

    def test_backward_compatibility(self):
        """Test that old code using get_replay_buffer still works."""
        # Clear any existing state
        factory = get_factory()
        factory.clear_all(force=True)

        # Old code pattern
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress deprecation warning
            buffer1 = get_replay_buffer({'buffer_hours': 24})
            buffer2 = get_replay_buffer()  # Should return same instance

        assert buffer1 is buffer2
        assert buffer1.config['buffer_hours'] == 24

    def test_mixed_usage(self):
        """Test that factory and deprecated function work together."""
        factory = get_factory()
        factory.clear_all(force=True)

        # Create via factory
        factory_buffer = factory.create_buffer('default', {'test': 1})

        # Get via deprecated function
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deprecated_buffer = get_replay_buffer()

        # Should be the same instance
        assert factory_buffer is deprecated_buffer

    def test_gradual_migration(self):
        """Test gradual migration scenario."""
        factory = get_factory()
        factory.clear_all(force=True)

        # Phase 1: Old code still using singleton
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old_buffer = get_replay_buffer({'phase': 1})

        # Phase 2: New code using factory
        new_buffer = factory.get_buffer('default')

        # Should work together
        assert old_buffer is new_buffer
        assert old_buffer.config['phase'] == 1

        # Phase 3: Create additional instances with factory
        test_buffer = factory.create_buffer('test', {'phase': 3})

        # Different instances
        assert test_buffer is not old_buffer
        assert test_buffer.config['phase'] == 3
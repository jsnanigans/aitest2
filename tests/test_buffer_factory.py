"""
Test suite for BufferFactory implementation.

Tests the factory pattern implementation that replaces the singleton anti-pattern.
Follows pytest best practices with fixtures, parametrization, and markers.
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


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def clean_factory():
    """Provide a clean BufferFactory instance for each test."""
    factory = BufferFactory()
    # Clear any existing state
    factory.clear_all(force=True)
    yield factory
    # Cleanup after test
    factory.clear_all(force=True)


@pytest.fixture
def factory_with_buffers(clean_factory):
    """Factory pre-populated with test buffers."""
    clean_factory.create_buffer('buffer1')
    clean_factory.create_buffer('buffer2')
    clean_factory.create_buffer('buffer3')
    return clean_factory


@pytest.fixture
def buffer_config():
    """Standard buffer configuration for testing."""
    return {
        'buffer_hours': 24,
        'max_buffer_measurements': 50
    }


@pytest.fixture
def global_factory():
    """Get the global factory instance and clean it."""
    factory = get_factory()
    factory.clear_all(force=True)
    yield factory
    factory.clear_all(force=True)


# =============================================================================
# TEST CLASSES
# =============================================================================

@pytest.mark.unit
class TestBufferFactory:
    """Test BufferFactory core functionality."""

    def test_create_buffer_basic(self, clean_factory):
        """Test basic buffer creation.

        Given: A clean factory
        When: Creating buffers with different names
        Then: Should return correct ReplayBuffer instances
        """
        # Create a buffer
        buffer1 = clean_factory.create_buffer('test1')
        assert isinstance(buffer1, ReplayBuffer), "Should create ReplayBuffer instance"

        # Get the same buffer
        buffer2 = clean_factory.create_buffer('test1')
        assert buffer1 is buffer2, "Should return same instance for same name"

        # Create different buffer
        buffer3 = clean_factory.create_buffer('test2')
        assert buffer3 is not buffer1, "Should create different instance for different name"

    def test_buffer_with_config(self, clean_factory, buffer_config):
        """Test buffer creation with configuration.

        Given: A factory and configuration
        When: Creating a buffer with config
        Then: Buffer should have the specified configuration
        """
        buffer = clean_factory.create_buffer('configured', buffer_config)

        assert buffer.config['buffer_hours'] == buffer_config['buffer_hours'], \
            f"Expected buffer_hours {buffer_config['buffer_hours']}, got {buffer.config['buffer_hours']}"
        assert buffer.config['max_buffer_measurements'] == buffer_config['max_buffer_measurements'], \
            f"Expected max_buffer_measurements {buffer_config['max_buffer_measurements']}"

    @pytest.mark.parametrize("num_buffers,should_fail", [
        (5, False),   # Well under limit
        (9, False),   # Just under limit
        (10, False),  # At limit
        (11, True),   # Over limit
        (15, True),   # Well over limit
    ])
    def test_instance_limit(self, clean_factory, num_buffers, should_fail):
        """Test that factory enforces instance limit.

        Given: A factory with 10 instance limit
        When: Creating various numbers of buffers
        Then: Should enforce the limit appropriately
        """
        if should_fail:
            # Create max allowed first
            for i in range(10):
                clean_factory.create_buffer(f'buffer_{i}')

            # Next one should fail
            with pytest.raises(ValueError, match="Maximum buffer instances"):
                clean_factory.create_buffer(f'buffer_{num_buffers - 1}')
        else:
            # Should succeed
            for i in range(num_buffers):
                buffer = clean_factory.create_buffer(f'buffer_{i}')
                assert buffer is not None, f"Failed to create buffer {i}"

    @pytest.mark.parametrize("buffer_name,exists", [
        ('nonexistent', False),
        ('existing', True),
    ])
    def test_get_buffer(self, clean_factory, buffer_name, exists):
        """Test getting existing and non-existing buffers.

        Given: A factory with some buffers
        When: Getting buffers by name
        Then: Should return buffer or None appropriately
        """
        if exists:
            created = clean_factory.create_buffer('existing')
            retrieved = clean_factory.get_buffer('existing')
            assert created is retrieved, "Should retrieve same buffer instance"
        else:
            result = clean_factory.get_buffer(buffer_name)
            assert result is None, f"Should return None for non-existent buffer {buffer_name}"

    def test_remove_buffer_with_references(self, clean_factory):
        """Test buffer removal with reference counting.

        Given: A buffer with references
        When: Attempting to remove it
        Then: Should respect reference counting
        """
        # Create buffer
        buffer = clean_factory.create_buffer('removable')

        # Try to remove while referenced
        assert not clean_factory.remove_buffer('removable'), \
            "Should not remove buffer with active references"

        # Decrement reference manually (simulating release)
        clean_factory._instance_refs['removable'] = 0

        # Now removal should work
        assert clean_factory.remove_buffer('removable'), \
            "Should remove buffer with zero references"
        assert clean_factory.get_buffer('removable') is None, \
            "Removed buffer should not be retrievable"

    def test_managed_buffer_context(self, clean_factory):
        """Test context manager for buffer lifecycle.

        Given: A factory
        When: Using managed_buffer context manager
        Then: Should properly manage buffer lifecycle
        """
        # Use managed buffer
        with clean_factory.managed_buffer('context_test') as buffer:
            assert isinstance(buffer, ReplayBuffer), "Should provide ReplayBuffer in context"
            assert 'context_test' in clean_factory.list_buffers(), \
                "Buffer should exist in factory during context"

        # After context, check cleanup
        clean_factory._instance_refs['context_test'] = 0
        clean_factory.remove_buffer('context_test')
        assert 'context_test' not in clean_factory.list_buffers(), \
            "Buffer should be removable after context"

    def test_managed_buffer_exception_handling(self, clean_factory):
        """Test that cleanup happens even with exceptions.

        Given: A managed buffer context
        When: An exception occurs in the context
        Then: Should still properly cleanup
        """
        try:
            with clean_factory.managed_buffer('exception_test') as buffer:
                assert isinstance(buffer, ReplayBuffer)
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Reference should be decremented despite exception
        assert clean_factory._instance_refs.get('exception_test', 0) == 0, \
            "Reference count should be decremented after exception"

    def test_list_buffers(self, factory_with_buffers):
        """Test listing active buffers.

        Given: A factory with multiple buffers
        When: Listing buffers
        Then: Should return all buffer names
        """
        buffers = factory_with_buffers.list_buffers()

        assert len(buffers) == 3, f"Expected 3 buffers, got {len(buffers)}"
        assert 'buffer1' in buffers, "buffer1 should be in list"
        assert 'buffer2' in buffers, "buffer2 should be in list"
        assert 'buffer3' in buffers, "buffer3 should be in list"

    @pytest.mark.parametrize("force", [True, False])
    def test_clear_all(self, clean_factory, force):
        """Test clearing all buffers with and without force.

        Given: A factory with buffers
        When: Clearing with/without force
        Then: Should behave according to force flag
        """
        # Create some buffers
        clean_factory.create_buffer('clear1')
        clean_factory.create_buffer('clear2')

        if force:
            # Force clear should always work
            clean_factory.clear_all(force=True)
            assert len(clean_factory.list_buffers()) == 0, \
                "Force clear should remove all buffers"
        else:
            # Clear without force should fail if references exist
            with pytest.raises(RuntimeError, match="active references"):
                clean_factory.clear_all(force=False)

    def test_default_config(self, clean_factory, buffer_config):
        """Test default configuration setting.

        Given: A factory with default config
        When: Creating new buffers
        Then: Should use default config
        """
        clean_factory.set_default_config(buffer_config)

        # New buffer should use default config
        buffer = clean_factory.create_buffer('with_default')
        assert buffer.config['buffer_hours'] == buffer_config['buffer_hours'], \
            f"Buffer should use default config buffer_hours"

    def test_get_stats(self, factory_with_buffers):
        """Test factory statistics.

        Given: A factory with buffers and config
        When: Getting stats
        Then: Should return accurate statistics
        """
        factory_with_buffers.set_default_config({'test': True})

        stats = factory_with_buffers.get_stats()

        assert stats['total_instances'] == 3, f"Expected 3 instances, got {stats['total_instances']}"
        assert stats['has_default_config'], "Should have default config"
        assert 'buffer1' in stats['instances'], "buffer1 should be in instances"
        assert stats['references']['buffer1'] == 1, "buffer1 should have 1 reference"

    @pytest.mark.slow
    def test_thread_safety(self, clean_factory):
        """Test thread-safe operations.

        Given: Multiple threads
        When: Creating buffers concurrently
        Then: Should handle concurrent access safely
        """
        created_buffers = []
        errors = []
        lock = threading.Lock()

        def create_buffers(thread_id):
            for i in range(3):  # Reduced to avoid hitting limit
                name = f"thread_{thread_id}_buffer_{i}"
                try:
                    buffer = clean_factory.create_buffer(name)
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
        assert len(errors) == 0, f"Unexpected errors: {errors}"

        # Check buffers created (respecting 10 instance limit)
        num_buffers = len(clean_factory.list_buffers())
        assert 0 < num_buffers <= 10, f"Should have 1-10 buffers, got {num_buffers}"

    def test_cleanup_on_removal(self, clean_factory):
        """Test that buffer cleanup is called on removal.

        Given: A buffer with cleanup method
        When: Removing the buffer
        Then: Should call cleanup
        """
        with patch.object(ReplayBuffer, 'cleanup') as mock_cleanup:
            buffer = clean_factory.create_buffer('cleanup_test')

            # Manually set ref count to allow removal
            clean_factory._instance_refs['cleanup_test'] = 0

            # Remove should trigger cleanup
            clean_factory.remove_buffer('cleanup_test')
            mock_cleanup.assert_called_once()


@pytest.mark.integration
class TestFactoryIntegration:
    """Test factory integration with other components."""

    def test_singleton_deprecation(self):
        """Test backward compatibility with deprecated get_replay_buffer.

        Given: The deprecated get_replay_buffer function
        When: Using it
        Then: Should work but issue deprecation warning
        """
        # Should trigger deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            buffer = get_replay_buffer({'test': True})

            # Check warning was raised
            assert len(w) == 1, "Should raise one warning"
            assert issubclass(w[0].category, DeprecationWarning), \
                "Should be DeprecationWarning"
            assert "deprecated" in str(w[0].message).lower(), \
                "Warning should mention deprecation"

        # Should return default buffer from factory
        assert isinstance(buffer, ReplayBuffer), "Should return ReplayBuffer"
        factory = get_factory()
        assert 'default' in factory.list_buffers(), "Should create default buffer"

    def test_with_buffer_decorator(self, global_factory):
        """Test the with_buffer decorator for dependency injection.

        Given: A function decorated with @with_buffer
        When: Calling the function
        Then: Should inject buffer as first argument
        """
        call_args = []

        @with_buffer('decorated_test', {'buffer_hours': 12})
        def test_function(buffer, arg1, arg2):
            call_args.append((buffer, arg1, arg2))
            assert isinstance(buffer, ReplayBuffer)
            assert buffer.config['buffer_hours'] == 12
            return arg1 + arg2

        result = test_function(1, 2)

        assert result == 3, "Function should return correct result"
        assert len(call_args) == 1, "Function should be called once"

        # Check buffer handling
        if global_factory._instance_refs.get('decorated_test', 0) == 0:
            assert 'decorated_test' not in global_factory.list_buffers(), \
                "Buffer should be cleaned up after decorator"

    def test_global_factory_singleton(self):
        """Test that get_factory returns singleton.

        Given: Multiple calls to get_factory
        When: Comparing instances
        Then: Should be the same instance
        """
        factory1 = get_factory()
        factory2 = get_factory()

        assert factory1 is factory2, "Should return same factory instance"

        # Operations on one should affect the other
        factory1.create_buffer('singleton_test')
        assert 'singleton_test' in factory2.list_buffers(), \
            "Changes should be visible in both references"


@pytest.mark.integration
class TestMigrationPath:
    """Test migration from singleton to factory pattern."""

    def test_backward_compatibility(self, global_factory):
        """Test that old code using get_replay_buffer still works.

        Given: Legacy code pattern
        When: Using get_replay_buffer
        Then: Should work correctly
        """
        # Old code pattern
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress deprecation warning
            buffer1 = get_replay_buffer({'buffer_hours': 24})
            buffer2 = get_replay_buffer()  # Should return same instance

        assert buffer1 is buffer2, "Should return same default buffer"
        assert buffer1.config['buffer_hours'] == 24, \
            "Config should be applied to buffer"

    def test_mixed_usage(self, global_factory):
        """Test that factory and deprecated function work together.

        Given: Mixed usage of old and new patterns
        When: Creating and getting buffers
        Then: Should work seamlessly together
        """
        # Create via factory
        factory_buffer = global_factory.create_buffer('default', {'test': 1})

        # Get via deprecated function
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deprecated_buffer = get_replay_buffer()

        # Should be the same instance
        assert factory_buffer is deprecated_buffer, \
            "Factory and deprecated function should share default buffer"

    @pytest.mark.parametrize("phase", [1, 2, 3])
    def test_gradual_migration(self, global_factory, phase):
        """Test gradual migration scenario.

        Given: Code in various migration phases
        When: Using different patterns
        Then: Should all work together
        """
        if phase >= 1:
            # Phase 1: Old code still using singleton
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                old_buffer = get_replay_buffer({'phase': 1})

        if phase >= 2:
            # Phase 2: New code using factory
            new_buffer = global_factory.get_buffer('default')
            if phase >= 1:
                assert old_buffer is new_buffer, \
                    "Old and new patterns should share buffer"

        if phase >= 3:
            # Phase 3: Create additional instances with factory
            test_buffer = global_factory.create_buffer('test', {'phase': 3})

            # Different instances
            if phase >= 1:
                assert test_buffer is not old_buffer, \
                    "New buffers should be separate instances"
            assert test_buffer.config['phase'] == 3, \
                "New buffer should have its own config"
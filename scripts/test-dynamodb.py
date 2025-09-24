#!/usr/bin/env python3
"""
Test DynamoDB serialization with various data types.
"""

import os
import sys
import numpy as np
from datetime import datetime
from decimal import Decimal

# Set up environment for local testing
os.environ['DYNAMODB_ENDPOINT'] = 'http://localhost:8000'
os.environ['AWS_ACCESS_KEY_ID'] = 'local'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'local'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.database.dynamodb_store import DynamoDBStateStore

def test_serialization():
    """Test that various data types are properly serialized."""

    print("Testing DynamoDB serialization...")

    # Create store
    store = DynamoDBStateStore(table_name='test-weight-processor')

    # Test data with various types
    test_state = {
        'float_value': 123.456,
        'np_float32': np.float32(789.012),
        'np_float64': np.float64(345.678),
        'int_value': 42,
        'np_int32': np.int32(100),
        'np_int64': np.int64(200),
        'np_array': np.array([1.1, 2.2, 3.3]),
        'np_matrix': np.array([[1.0, 2.0], [3.0, 4.0]]),
        'nested_dict': {
            'inner_float': 99.99,
            'inner_array': np.array([5.5, 6.6])
        },
        'list_with_floats': [1.1, 2.2, 3.3],
        'datetime_value': datetime.now(),
        'none_value': None,
        'nan_value': np.nan,  # Should become None
        'inf_value': np.inf,  # Should become None
    }

    print("\nOriginal state:")
    for key, value in test_state.items():
        print(f"  {key}: {value} ({type(value).__name__})")

    # Serialize
    print("\nSerializing...")
    serialized = store._serialize_state(test_state)

    print("\nSerialized state:")
    for key, value in serialized.items():
        if value is not None:
            print(f"  {key}: {value} ({type(value).__name__})")
        else:
            print(f"  {key}: None")

    # Check that all floats are Decimals
    def check_for_floats(obj, path=""):
        """Recursively check for any remaining float types."""
        if isinstance(obj, float):
            return f"Float found at {path}: {obj}"
        elif isinstance(obj, dict):
            for key, value in obj.items():
                result = check_for_floats(value, f"{path}.{key}" if path else key)
                if result:
                    return result
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                result = check_for_floats(item, f"{path}[{i}]")
                if result:
                    return result
        return None

    float_check = check_for_floats(serialized)
    if float_check:
        print(f"\n❌ ERROR: {float_check}")
        return False
    else:
        print("\n✓ All floats properly converted to Decimal")

    # Test save and retrieve
    print("\nTesting save and retrieve...")
    user_id = 'test-user-123'

    try:
        # Save state
        success = store.save_state(user_id, test_state)
        if success:
            print("✓ State saved successfully")
        else:
            print("❌ Failed to save state")
            return False

        # Retrieve state
        retrieved = store.get_state(user_id)
        if retrieved:
            print("✓ State retrieved successfully")

            # Check deserialization
            print("\nRetrieved state types:")
            for key in ['float_value', 'np_float32', 'np_float64']:
                if key in retrieved:
                    value = retrieved[key]
                    print(f"  {key}: {value} ({type(value).__name__})")

            # Clean up
            store.delete_state(user_id)
            print("\n✓ Test data cleaned up")
        else:
            print("❌ Failed to retrieve state")
            return False

    except Exception as e:
        print(f"\n❌ Error during save/retrieve: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_serialization()
    sys.exit(0 if success else 1)
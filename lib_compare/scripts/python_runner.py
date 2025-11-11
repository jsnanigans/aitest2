#!/usr/bin/env python3
"""
Python runner script for cross-language comparison tests
Reads input from stdin, processes measurements, writes results to stdout
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Add python_lib to path
python_lib_path = Path(__file__).parent.parent.parent / "python_lib" / "src"
sys.path.insert(0, str(python_lib_path))

from weight_processor_lib.core.processing.processor import process_measurement
from weight_processor_lib.core.database.memory_store import InMemoryStore
from weight_processor_lib.core.config import ConfigManager

# Load configuration from config.toml
config_path = Path(__file__).parent.parent.parent / "python_lib" / "config.toml"
PROCESSING_CONFIG = ConfigManager.load_config(str(config_path))


def make_json_serializable(obj):
    """Recursively convert numpy/datetime objects to JSON-serializable types"""
    if obj is None:
        return None
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, datetime):
        return obj.timestamp() * 1000  # milliseconds
    elif isinstance(obj, str):
        # Check if it's an ISO timestamp string and convert to milliseconds
        # Format: "2025-11-10T10:24:32.710000" or "2025-11-10T10:24:32.710Z"
        if len(obj) >= 19 and 'T' in obj and (':' in obj):
            try:
                # Try to parse as ISO format
                if obj.endswith('Z'):
                    dt = datetime.fromisoformat(obj.replace('Z', '+00:00'))
                else:
                    dt = datetime.fromisoformat(obj)
                return dt.timestamp() * 1000  # Convert to milliseconds
            except ValueError:
                # Not a timestamp, return as is
                return obj
        return obj
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


def serialize_result(result):
    """Serialize processing result to JSON-compatible dict"""
    if result is None:
        return None

    # process_measurement already returns a dict, just ensure it's JSON-compatible
    # Convert any datetime objects to ISO format strings
    serialized = dict(result)

    # Convert timestamp if it's a datetime
    if 'timestamp' in serialized and hasattr(serialized['timestamp'], 'isoformat'):
        serialized['timestamp'] = serialized['timestamp'].timestamp() * 1000  # Convert to milliseconds

    return serialized


def serialize_state(state):
    """Serialize Kalman state to JSON-compatible dict"""
    if state is None:
        return None

    # State is already a dict from InMemoryStore
    serialized = dict(state)

    # Convert numpy arrays to lists if present
    if 'kalman_state' in serialized and serialized['kalman_state'] is not None:
        if hasattr(serialized['kalman_state'], 'tolist'):
            serialized['kalman_state'] = serialized['kalman_state'].tolist()

    if 'kalman_covariance' in serialized and serialized['kalman_covariance'] is not None:
        if hasattr(serialized['kalman_covariance'], 'tolist'):
            serialized['kalman_covariance'] = serialized['kalman_covariance'].tolist()

    # Convert datetime objects to timestamps in milliseconds
    for key in ['last_timestamp', 'created_at', 'updated_at']:
        if key in serialized and serialized[key] is not None:
            if hasattr(serialized[key], 'timestamp'):
                serialized[key] = serialized[key].timestamp() * 1000

    return serialized


def main():
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())

    device_id = input_data["deviceId"]
    user_id = input_data["userId"]
    measurements = input_data["measurements"]

    # Create state store
    state_store = InMemoryStore()

    # Combine device_id and user_id for the user_id parameter
    combined_user_id = f"{device_id}:{user_id}"

    # Process measurements
    results = []
    for measurement in measurements:
        # Convert timestamp from milliseconds to datetime
        # JavaScript Date.now() returns milliseconds, Python expects seconds
        # Use UTC to match JavaScript Date behavior
        timestamp = datetime.utcfromtimestamp(measurement["timestamp"] / 1000)

        result = process_measurement(
            user_id=combined_user_id,
            weight=measurement["weight_kg"],
            timestamp=timestamp,
            source=measurement["source"],
            config=PROCESSING_CONFIG,
            db=state_store,
        )
        results.append(serialize_result(result))

    # Get final state
    final_state = state_store.get_state(combined_user_id)

    # Output results as JSON
    output = {
        "results": results,
        "finalState": serialize_state(final_state),
    }

    # Make everything JSON-serializable
    output = make_json_serializable(output)

    print(json.dumps(output))


if __name__ == "__main__":
    main()

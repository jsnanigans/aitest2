#!/usr/bin/env python3
"""
Persistent Python server for cross-language comparison tests
Keeps Python process running and processes requests via stdin/stdout
This eliminates subprocess spawning overhead for fair comparison
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

# Load configuration once at startup (like a real service would)
config_path = Path(__file__).parent.parent.parent / "python_lib" / "config.toml"
PROCESSING_CONFIG = ConfigManager.load_config(str(config_path))

# Keep a single state store instance (simulating a persistent service)
state_stores = {}


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
        if len(obj) >= 19 and 'T' in obj and (':' in obj):
            try:
                if obj.endswith('Z'):
                    dt = datetime.fromisoformat(obj.replace('Z', '+00:00'))
                else:
                    dt = datetime.fromisoformat(obj)
                return dt.timestamp() * 1000
            except ValueError:
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

    serialized = dict(result)
    if 'timestamp' in serialized and hasattr(serialized['timestamp'], 'isoformat'):
        serialized['timestamp'] = serialized['timestamp'].timestamp() * 1000

    return serialized


def serialize_state(state):
    """Serialize Kalman state to JSON-compatible dict"""
    if state is None:
        return None

    serialized = dict(state)

    if 'kalman_state' in serialized and serialized['kalman_state'] is not None:
        if hasattr(serialized['kalman_state'], 'tolist'):
            serialized['kalman_state'] = serialized['kalman_state'].tolist()

    if 'kalman_covariance' in serialized and serialized['kalman_covariance'] is not None:
        if hasattr(serialized['kalman_covariance'], 'tolist'):
            serialized['kalman_covariance'] = serialized['kalman_covariance'].tolist()

    for key in ['last_timestamp', 'created_at', 'updated_at']:
        if key in serialized and serialized[key] is not None:
            if hasattr(serialized[key], 'timestamp'):
                serialized[key] = serialized[key].timestamp() * 1000

    return serialized


def process_request(request_data):
    """Process a single request"""
    device_id = request_data["deviceId"]
    user_id = request_data["userId"]
    measurements = request_data["measurements"]

    # Get or create state store for this device/user
    # Each test gets its own store (like separate Lambda invocations)
    combined_user_id = f"{device_id}:{user_id}"

    # Create fresh store for this request (simulating stateless Lambda)
    state_store = InMemoryStore()

    # Process measurements
    results = []
    for measurement in measurements:
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

    # Output results
    output = {
        "results": results,
        "finalState": serialize_state(final_state),
    }

    return make_json_serializable(output)


def main():
    """Main server loop - keep processing requests until EOF"""
    sys.stderr.write("Python server ready\n")
    sys.stderr.flush()

    while True:
        try:
            # Read request length first (newline-delimited JSON)
            line = sys.stdin.readline()
            if not line:
                # EOF - exit gracefully
                break

            request_data = json.loads(line)

            # Process request
            response = process_request(request_data)

            # Write response as newline-delimited JSON
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        except Exception as e:
            # Send error response
            error_response = {
                "error": str(e),
                "results": [],
                "finalState": None,
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

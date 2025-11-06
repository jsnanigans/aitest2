"""
Type conversion utilities for handling Decimal/float conversions.
"""

from decimal import Decimal
from typing import Any, Dict, List, Union
import numpy as np


def ensure_float(value: Any) -> float:
    """
    Convert a value to float, handling Decimal and other numeric types.

    Args:
        value: Value to convert

    Returns:
        Float value
    """
    if value is None:
        return 0.0

    # Check if it's a Decimal
    if isinstance(value, Decimal):
        return float(value)

    # Check if it's already a float or int
    if isinstance(value, (float, int)):
        return float(value)

    # Check for numpy types
    if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
        return float(value)

    # Try to convert anything else
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def ensure_numeric_types(data: Union[Dict, List, Any]) -> Union[Dict, List, Any]:
    """
    Recursively ensure all numeric values in a data structure are proper Python types.
    Converts Decimal to float, numpy types to Python types.

    Args:
        data: Data structure to process

    Returns:
        Data with proper numeric types
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key in [
                "weight",
                "filtered_weight",
                "raw_weight",
                "quality_score",
                "kalman_deviation",
                "temporal_consistency",
                "source_reliability",
            ]:
                # These are numeric fields that should be floats
                result[key] = ensure_float(value)
            elif isinstance(value, (dict, list)):
                result[key] = ensure_numeric_types(value)
            elif isinstance(value, Decimal):
                result[key] = float(value)
            elif isinstance(value, (np.float32, np.float64)):
                result[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                result[key] = int(value)
            else:
                result[key] = value
        return result

    elif isinstance(data, list):
        result = []
        for item in data:
            if isinstance(item, (dict, list)):
                result.append(ensure_numeric_types(item))
            elif isinstance(item, Decimal):
                result.append(float(item))
            elif isinstance(item, (np.float32, np.float64)):
                result.append(float(item))
            elif isinstance(item, (np.int32, np.int64)):
                result.append(int(item))
            else:
                result.append(item)
        return result

    elif isinstance(data, Decimal):
        return float(data)
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)

    return data


def prepare_measurement_for_processing(measurement: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a measurement for processing by ensuring proper types.

    Args:
        measurement: Measurement dictionary

    Returns:
        Measurement with proper types
    """
    clean_measurement = measurement.copy()

    # Ensure weight is float
    if "weight" in clean_measurement:
        clean_measurement["weight"] = ensure_float(clean_measurement["weight"])

    if "raw_weight" in clean_measurement:
        clean_measurement["raw_weight"] = ensure_float(clean_measurement["raw_weight"])

    if "filtered_weight" in clean_measurement:
        clean_measurement["filtered_weight"] = ensure_float(
            clean_measurement["filtered_weight"]
        )

    if "quality_score" in clean_measurement:
        clean_measurement["quality_score"] = ensure_float(
            clean_measurement["quality_score"]
        )

    # Handle nested metadata
    if "metadata" in clean_measurement and isinstance(
        clean_measurement["metadata"], dict
    ):
        clean_measurement["metadata"] = ensure_numeric_types(
            clean_measurement["metadata"]
        )

    return clean_measurement

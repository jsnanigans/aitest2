#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from src.filters.custom_kalman_filter import CustomKalmanFilter
from datetime import datetime, timedelta

# Initialize filter
kf = CustomKalmanFilter(
    initial_weight=79.3,
    initial_trend=0.0,
    measurement_noise=0.5,
    enable_validation=True
)

# Start time
start_time = datetime(2025, 5, 1)

# Process some normal readings
weights = [79.3, 79.2, 79.1, 79.0, 78.9]
for i, w in enumerate(weights):
    result = kf.process_measurement(w, start_time + timedelta(days=i))
    print(f"Day {i}: Weight={w:.1f}, Filtered={result['filtered_weight']:.1f}, Accepted={result['measurement_accepted']}")

# Now add repeated suspicious value
print("\n--- Adding repeated 77.564 values ---")
for i in range(5, 10):
    result = kf.process_measurement(77.564, start_time + timedelta(days=i))
    print(f"Day {i}: Weight=77.564, Filtered={result['filtered_weight']:.1f}, Accepted={result['measurement_accepted']}")
    if hasattr(kf, 'recent_exact_values'):
        count = sum(1 for v in kf.recent_exact_values if abs(v - 77.564) < 0.001)
        print(f"  Exact value count: {count}, Recent values: {kf.recent_exact_values[-5:]}")

# Now add normal values again
print("\n--- Back to normal values around 71kg ---")
normal_values = [71.5, 71.6, 71.4]
for i, w in enumerate(normal_values, start=10):
    result = kf.process_measurement(w, start_time + timedelta(days=i))
    print(f"Day {i}: Weight={w:.1f}, Filtered={result['filtered_weight']:.1f}, Accepted={result['measurement_accepted']}")
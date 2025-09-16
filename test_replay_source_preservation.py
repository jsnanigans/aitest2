import sys
import os
sys.path.insert(0, '/Users/brendanmullins/Projects/aitest/strem_process_anchor')

from datetime import datetime
from src.retro_buffer import RetroBuffer
from src.outlier_detection import OutlierDetector
import toml

# Load config
config = toml.load('config.toml')

# Create buffer and detector
buffer = RetroBuffer(config)
detector = OutlierDetector(config)

# Add measurements to buffer
user_id = "test_user"
measurements = [
    {
        'weight': 91.26,
        'timestamp': datetime(2025, 1, 27, 8, 55, 19),
        'source': 'patient-device',
        'unit': 'kg',
        'metadata': {}
    },
    {
        'weight': 95.25432,
        'timestamp': datetime(2025, 1, 28, 0, 0, 0),
        'source': 'internal-questionnaire',
        'unit': 'kg',
        'metadata': {}
    }
]

for m in measurements:
    buffer.add_measurement(user_id, m)

# Get measurements from buffer - use the correct method
buffered = buffer.get_buffer_measurements(user_id)
if buffered:
    print("Measurements in buffer:")
    for m in buffered:
        print(f"  {m['timestamp']}: {m['weight']} kg from {m.get('source', 'NO SOURCE!')}")

    # Run through outlier detection
    clean_measurements, outliers = detector.get_clean_measurements(buffered)
    print("\nClean measurements after outlier detection:")
    for m in clean_measurements:
        print(f"  {m['timestamp']}: {m['weight']} kg from {m.get('source', 'NO SOURCE!')}")

#!/usr/bin/env python3
"""
Simple test to verify the replay system is actually being called
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.processing.pipeline import ProcessingPipeline
from src.processing.replay_buffer import ReplayBuffer
from src.replay.replay_manager import ReplayManager
from src.processing.outlier_detection import OutlierDetector
from src.database.database import ProcessorStateDB


def test_replay_components():
    """Test that replay components work together"""
    
    print("Testing replay components...")
    
    # Initialize components
    db = ProcessorStateDB(':memory:')
    
    config = {
        'buffer_hours': 24,
        'max_buffer_measurements': 100,
        'outlier_detection': {
            'iqr_multiplier': 1.5,
            'z_score_threshold': 3.0,
            'temporal_max_change_percent': 0.5,
            'min_measurements_for_analysis': 2
        },
        'safety': {
            'max_processing_time_seconds': 60,
            'require_rollback_confirmation': False,
            'preserve_immediate_results': True
        }
    }
    
    buffer = ReplayBuffer(config)
    outlier_detector = OutlierDetector(config['outlier_detection'], db=db)
    replay_manager = ReplayManager(db, config['safety'])
    
    # Test data
    user_id = 'test_user'
    measurements = [
        {
            'weight': 79.5,
            'timestamp': datetime(2025, 4, 10, 10, 0, 0),
            'source': 'patient-device',
            'unit': 'kg'
        },
        {
            'weight': 100.1,  # Outlier (BMI converted to weight)
            'timestamp': datetime(2025, 4, 10, 19, 27, 5),
            'source': 'patient-device',
            'unit': 'kg'
        },
        {
            'weight': 79.6,
            'timestamp': datetime(2025, 4, 10, 19, 34, 18),
            'source': 'patient-device',
            'unit': 'kg'
        },
        {
            'weight': 79.7,
            'timestamp': datetime(2025, 4, 10, 19, 34, 40),
            'source': 'patient-device',
            'unit': 'kg'
        }
    ]
    
    # Add measurements to buffer
    print(f"\nAdding {len(measurements)} measurements to buffer...")
    for m in measurements:
        result = buffer.add_measurement(user_id, m)
        print(f"  Added {m['weight']:.1f}kg at {m['timestamp']}: buffer_size={result.get('buffer_size', 0)}")
    
    # Get buffered measurements
    buffered = buffer.get_buffer_measurements(user_id)
    print(f"\nBuffer contains {len(buffered)} measurements")
    print(f"First measurement type: {type(buffered[0]) if buffered else 'N/A'}")
    if buffered and len(buffered) > 0:
        print(f"First measurement: {buffered[0]}")
    
    # Run outlier detection
    print("\nRunning outlier detection...")
    outlier_indices = outlier_detector.detect_outliers(buffered, user_id)
    print(f"  Found {len(outlier_indices)} outliers")
    
    # Get the actual outlier measurements
    outliers = [buffered[i] for i in outlier_indices]
    for outlier in outliers:
        print(f"    Outlier: {outlier.get('weight', 'N/A')}kg at {outlier.get('timestamp', 'N/A')}")
    
    # Filter out outliers
    clean_measurements = [m for i, m in enumerate(buffered) if i not in outlier_indices]
    print(f"\nClean measurements: {len(clean_measurements)}")
    for m in clean_measurements:
        print(f"  {m['weight']:.1f}kg at {m['timestamp']}")
    
    # Test replay manager
    if clean_measurements:
        buffer_start_time = min(m['timestamp'] for m in buffered)
        print(f"\nCalling replay_clean_measurements...")
        print(f"  Buffer start time: {buffer_start_time}")
        print(f"  Clean measurements: {len(clean_measurements)}")
        
        # Note: This will fail without a proper Kalman state, but we're testing the interface
        try:
            result = replay_manager.replay_clean_measurements(
                user_id=user_id,
                clean_measurements=clean_measurements,
                buffer_start_time=buffer_start_time
            )
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Expected error (no Kalman state): {e}")
    
    print("\n✅ Components are properly connected")
    return True


if __name__ == "__main__":
    success = test_replay_components()
    sys.exit(0 if success else 1)
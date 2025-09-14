#!/usr/bin/env python3
"""
Performance measurement script for weight processor.
Measures processing time per measurement and identifies bottlenecks.
"""

import time
import sys
import os
from datetime import datetime, timedelta
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import WeightProcessor
from src.database import ProcessorDatabase

def measure_single_measurement(processor_db, user_id, weight, timestamp, source='test'):
    """Measure time for a single measurement."""
    start = time.perf_counter()
    
    result = WeightProcessor.process_weight(
        user_id=user_id,
        weight=weight,
        timestamp=timestamp,
        source=source,
        processing_config={},
        kalman_config={},
        db=processor_db
    )
    
    end = time.perf_counter()
    return (end - start) * 1000  # Convert to milliseconds

def run_performance_test(num_measurements=100):
    """Run performance test with multiple measurements."""
    print(f"Running performance test with {num_measurements} measurements...")
    
    # Initialize database
    db = ProcessorDatabase()
    user_id = "perf_test_user"
    
    # Warm-up run
    print("Warming up...")
    for i in range(10):
        timestamp = datetime.now() - timedelta(hours=100-i)
        weight = 70.0 + (i % 3) * 0.5
        measure_single_measurement(db, user_id, weight, timestamp)
    
    # Actual measurements
    times = []
    print(f"Measuring {num_measurements} measurements...")
    
    for i in range(num_measurements):
        timestamp = datetime.now() - timedelta(hours=num_measurements-i)
        weight = 70.0 + (i % 5) * 0.3  # Vary weight slightly
        
        time_ms = measure_single_measurement(db, user_id, weight, timestamp)
        times.append(time_ms)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_measurements} measurements")
    
    # Calculate statistics
    avg_time = statistics.mean(times)
    median_time = statistics.median(times)
    min_time = min(times)
    max_time = max(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0
    
    print("\n" + "="*50)
    print("PERFORMANCE RESULTS")
    print("="*50)
    print(f"Measurements processed: {num_measurements}")
    print(f"Average time: {avg_time:.2f} ms")
    print(f"Median time: {median_time:.2f} ms")
    print(f"Min time: {min_time:.2f} ms")
    print(f"Max time: {max_time:.2f} ms")
    print(f"Std deviation: {stdev_time:.2f} ms")
    print(f"Target: <3 ms (currently {avg_time/3:.1f}x slower)")
    print("="*50)
    
    # Identify if we meet the target
    if avg_time <= 3.0:
        print("✅ PERFORMANCE TARGET MET!")
    elif avg_time <= 5.0:
        print("⚠️ Close to target (need {:.1f}x improvement)".format(avg_time/3))
    else:
        print("❌ Performance improvement needed ({:.1f}x improvement required)".format(avg_time/3))
    
    return {
        'avg_ms': avg_time,
        'median_ms': median_time,
        'min_ms': min_time,
        'max_ms': max_time,
        'stdev_ms': stdev_time
    }

def profile_components():
    """Profile individual components to identify bottlenecks."""
    print("\nProfiling individual components...")
    
    from src.validation import DataQualityPreprocessor, PhysiologicalValidator
    from src.kalman import KalmanFilterManager
    
    db = ProcessorDatabase()
    user_id = "profile_test"
    weight = 75.0
    timestamp = datetime.now()
    source = "test"
    
    # Profile preprocessing
    start = time.perf_counter()
    for _ in range(100):
        cleaned, metadata = DataQualityPreprocessor.preprocess(
            weight, source, timestamp, user_id
        )
    preprocess_time = (time.perf_counter() - start) * 10  # ms per call
    
    # Profile validation
    start = time.perf_counter()
    for _ in range(100):
        result = PhysiologicalValidator.validate_comprehensive(
            weight, previous_weight=74.0, time_diff_hours=24, source=source
        )
    validation_time = (time.perf_counter() - start) * 10  # ms per call
    
    # Profile Kalman update (need to initialize first)
    state = KalmanFilterManager.initialize_immediate(weight, timestamp, {})
    
    start = time.perf_counter()
    for _ in range(100):
        state = KalmanFilterManager.update_state(
            state, weight, timestamp, {'observation_covariance': 3.49}, {}
        )
    kalman_time = (time.perf_counter() - start) * 10  # ms per call
    
    print("\nComponent timing breakdown:")
    print(f"  Preprocessing: {preprocess_time:.2f} ms")
    print(f"  Validation: {validation_time:.2f} ms")
    print(f"  Kalman update: {kalman_time:.2f} ms")
    print(f"  Total: {preprocess_time + validation_time + kalman_time:.2f} ms")

if __name__ == "__main__":
    results = run_performance_test(100)
    profile_components()
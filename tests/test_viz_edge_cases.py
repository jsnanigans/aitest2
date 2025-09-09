#!/usr/bin/env python3
"""
Test visualization edge cases including users with only raw data, 
no accepted results, or partially processed data.
"""

import sys
sys.path.insert(0, '..')

from datetime import datetime, timedelta
from src.processor import WeightProcessor
from src.processor_database import ProcessorStateDB
from src.visualization import create_dashboard

def test_various_scenarios():
    """Test different edge case scenarios for visualization."""
    
    config = {
        "processing": {
            "min_weight": 30,
            "max_weight": 400,
            "max_daily_change": 0.05,
            "extreme_threshold": 0.20,
            "kalman_cleanup_threshold": 4.0
        },
        "kalman": {
            "initial_variance": 0.5,
            "transition_covariance_weight": 0.05,
            "transition_covariance_trend": 0.0005,
            "observation_covariance": 1.5,
            "reset_gap_days": 30
        }
    }
    
    viz_config = {
        "dashboard_dpi": 100,
        "enabled": True
    }
    
    # Scenario 1: User with only raw data (less than initialization threshold)
    print("\n=== Scenario 1: User with only raw data (< init threshold) ===")
    user_id_1 = "test-user-raw-only"
    db1 = ProcessorStateDB()
    raw_data_1 = []
    results_1 = []
    
    base_date = datetime(2025, 2, 1)
    for i in range(5):  # Only 5 measurements (less than typical init)
        weight = 75.0 + i * 0.2
        timestamp = base_date + timedelta(days=i*2)
        raw_data_1.append({
            'weight': weight,
            'timestamp': timestamp,
            'source': 'scale'
        })
        
        result = WeightProcessor.process_weight(
            user_id=user_id_1,
            weight=weight,
            timestamp=timestamp,
            source='scale',
            processing_config=config["processing"],
            kalman_config=config["kalman"],
            db=db1
        )
        
        if result:
            results_1.append(result)
    
    print(f"User {user_id_1}: {len(raw_data_1)} raw, {len(results_1)} processed")
    
    output_file = create_dashboard(
        user_id=user_id_1,
        results=results_1,
        output_dir=".",
        viz_config=viz_config
    )
    print(f"Visualization: {output_file}")
    
    # Scenario 2: User with all rejected measurements
    print("\n=== Scenario 2: User with all rejected measurements ===")
    user_id_2 = "test-user-all-rejected"
    db2 = ProcessorStateDB()
    raw_data_2 = []
    results_2 = []
    
    # First initialize normally
    for i in range(3):
        weight = 75.0
        timestamp = base_date + timedelta(days=i)
        raw_data_2.append({
            'weight': weight,
            'timestamp': timestamp,
            'source': 'scale'
        })
        
        result = WeightProcessor.process_weight(
            user_id=user_id_2,
            weight=weight,
            timestamp=timestamp,
            source='scale',
            processing_config=config["processing"],
            kalman_config=config["kalman"],
            db=db2
        )
        
        if result:
            results_2.append(result)
    
    # Then add extreme outliers that will be rejected
    for i in range(3, 8):
        weight = 150.0  # Extreme jump from 75kg
        timestamp = base_date + timedelta(days=i)
        raw_data_2.append({
            'weight': weight,
            'timestamp': timestamp,
            'source': 'scale'
        })
        
        result = WeightProcessor.process_weight(
            user_id=user_id_2,
            weight=weight,
            timestamp=timestamp,
            source='scale',
            processing_config=config["processing"],
            kalman_config=config["kalman"],
            db=db2
        )
        
        if result:
            results_2.append(result)
    
    accepted_2 = sum(1 for r in results_2 if r['accepted'])
    rejected_2 = sum(1 for r in results_2 if not r['accepted'])
    print(f"User {user_id_2}: {len(raw_data_2)} raw, {accepted_2} accepted, {rejected_2} rejected")
    
    output_file = create_dashboard(
        user_id=user_id_2,
        results=results_2,
        output_dir=".",
        viz_config=viz_config
    )
    print(f"Visualization: {output_file}")
    
    # Scenario 3: User with mixed results (some accepted, some rejected, some initialization)
    print("\n=== Scenario 3: User with mixed results ===")
    user_id_3 = "test-user-mixed"
    db3 = ProcessorStateDB()
    raw_data_3 = []
    results_3 = []
    
    # Generate mixed data
    weights_pattern = [75, 75.2, 75.1, 90, 75.3, 75.4, 100, 75.5, 75.6, 75.7, 
                      75.8, 40, 75.9, 76.0, 76.1]
    
    for i, weight in enumerate(weights_pattern):
        timestamp = base_date + timedelta(days=i*3)
        raw_data_3.append({
            'weight': weight,
            'timestamp': timestamp,
            'source': 'scale'
        })
        
        result = WeightProcessor.process_weight(
            user_id=user_id_3,
            weight=weight,
            timestamp=timestamp,
            source='scale',
            processing_config=config["processing"],
            kalman_config=config["kalman"],
            db=db3
        )
        
        if result:
            results_3.append(result)
    
    accepted_3 = sum(1 for r in results_3 if r['accepted'])
    rejected_3 = sum(1 for r in results_3 if not r['accepted'])
    print(f"User {user_id_3}: {len(raw_data_3)} raw, {accepted_3} accepted, {rejected_3} rejected")
    
    output_file = create_dashboard(
        user_id=user_id_3,
        results=results_3,
        output_dir=".",
        viz_config=viz_config
    )
    print(f"Visualization: {output_file}")
    
    print("\n=== All test visualizations created successfully ===")

if __name__ == "__main__":
    test_various_scenarios()
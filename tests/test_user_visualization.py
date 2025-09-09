#!/usr/bin/env python3
"""
Test visualization for specific user 00e96965-7bc0-43e4-bafa-fb7a0b573cf3
"""

import csv
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '..')

from src.processor import WeightProcessor
from src.processor_database import ProcessorStateDB
from src.visualization import create_dashboard

def test_user_visualization():
    user_id = "00e96965-7bc0-43e4-bafa-fb7a0b573cf3"
    db = ProcessorStateDB()
    
    # Create test data for this user
    base_date = datetime(2024, 10, 1)
    test_data = []
    
    # Generate 30 measurements over 3 months
    base_weight = 75.0
    for i in range(30):
        date = base_date + timedelta(days=i*3)
        # Add some variation
        weight = base_weight + (i * 0.05) + (0.5 if i % 3 == 0 else -0.3)
        test_data.append({
            'user_id': user_id,
            'weight': weight,
            'timestamp': date,
            'source': 'scale'
        })
    
    print(f"Processing {len(test_data)} measurements for user {user_id[:8]}...")
    
    # Process all measurements
    results = []
    raw_data = []
    
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
    
    for measurement in test_data:
        raw_data.append(measurement)
        
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=measurement['weight'],
            timestamp=measurement['timestamp'],
            source=measurement['source'],
            processing_config=config["processing"],
            kalman_config=config["kalman"],
            db=db
        )
        
        if result:
            results.append(result)
            print(f"  [{len(results):2d}] {measurement['timestamp'].strftime('%Y-%m-%d')}: "
                  f"Raw={measurement['weight']:.1f}kg → "
                  f"Filtered={result.get('filtered_weight', 0):.1f}kg "
                  f"({'✓' if result['accepted'] else '✗'})")
        else:
            print(f"  [--] {measurement['timestamp'].strftime('%Y-%m-%d')}: "
                  f"Raw={measurement['weight']:.1f}kg → Buffering...")
    
    print(f"\nResults: {len(results)} processed, {sum(1 for r in results if r['accepted'])} accepted")
    
    # Create visualization
    viz_config = {
        "dashboard_dpi": 100,
        "enabled": True
    }
    
    print("\nCreating visualization...")
    output_file = create_dashboard(
        user_id=user_id,
        results=results,
        output_dir=".",
        viz_config=viz_config
    )
    
    if output_file:
        print(f"Visualization saved to: {output_file}")
    else:
        print("No visualization created")
    
    # Check what's in the results
    if results:
        print(f"\nFirst result keys: {list(results[0].keys())}")
        print(f"Sample result: {results[0]}")

if __name__ == "__main__":
    test_user_visualization()
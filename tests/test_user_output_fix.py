#!/usr/bin/env python3
"""
Test that the visualization shows all processed values (accepted and rejected).
Previously, if a user had only one value shown, it meant others were stuck in initialization.
Now all processed values should be visible.
"""

import sys
sys.path.insert(0, '..')

from datetime import datetime, timedelta
from src.processor import WeightProcessor
from src.database import ProcessorStateDB
from src.visualization import create_dashboard

def test_output_fix():
    """Test that all processed measurements appear in visualization."""
    
    user_id = "00e96965-7bc0-43e4-bafa-fb7a0b573cf3"
    db = ProcessorStateDB()
    
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
    
    # Create test measurements
    base_date = datetime(2024, 10, 1)
    measurements = []
    
    # Generate 15 measurements
    weights = [75, 75.2, 75.1, 74.9, 75.3, 75.5, 75.4, 75.6, 75.8, 76.0,
              76.2, 76.1, 75.9, 76.3, 76.5]
    
    for i, weight in enumerate(weights):
        measurements.append({
            'weight': weight,
            'timestamp': base_date + timedelta(days=i*5),
            'source': 'scale'
        })
    
    print(f"Testing user {user_id[:8]} with {len(measurements)} measurements")
    
    results = []
    for i, m in enumerate(measurements):
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=m['weight'],
            timestamp=m['timestamp'],
            source=m['source'],
            processing_config=config["processing"],
            kalman_config=config["kalman"],
            db=db
        )
        
        if result:
            status = "✓" if result['accepted'] else "✗"
            print(f"  Measurement {i+1:2d}: {m['timestamp'].strftime('%Y-%m-%d')} "
                  f"Weight={m['weight']:.1f}kg → {status}")
            results.append(result)
        else:
            print(f"  Measurement {i+1:2d}: {m['timestamp'].strftime('%Y-%m-%d')} "
                  f"Weight={m['weight']:.1f}kg → NONE (shouldn't happen!)")
    
    accepted = sum(1 for r in results if r['accepted'])
    rejected = sum(1 for r in results if not r['accepted'])
    
    print(f"\nSummary:")
    print(f"  Total measurements: {len(measurements)}")
    print(f"  Processed results: {len(results)}")
    print(f"  Accepted: {accepted}")
    print(f"  Rejected: {rejected}")
    
    if len(results) != len(measurements):
        print("\n❌ ERROR: Not all measurements were processed!")
        print(f"   Expected {len(measurements)} results, got {len(results)}")
        return False
    
    print(f"\n✓ All {len(measurements)} measurements were processed and will appear in visualization")
    
    # Create visualization
    print("\nCreating visualization...")
    output_file = create_dashboard(
        user_id=user_id,
        results=results,
        output_dir=".",
        viz_config=viz_config
    )
    
    if output_file:
        print(f"Visualization saved to: {output_file}")
    
    return True

if __name__ == "__main__":
    success = test_output_fix()
    sys.exit(0 if success else 1)
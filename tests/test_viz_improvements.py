"""
Test improved visualization with synthetic data including rejections
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from processor import WeightProcessor
from visualization import create_dashboard


def generate_test_data():
    """Generate synthetic weight data with various scenarios."""
    user_id = "test-user-001"
    base_weight = 75.0
    timestamps = []
    weights = []
    sources = []
    
    start_date = datetime.now() - timedelta(days=90)
    
    for i in range(90):
        timestamp = start_date + timedelta(days=i)
        timestamps.append(timestamp)
        
        daily_variation = np.random.normal(0, 0.3)
        trend = -0.01 * i
        weight = base_weight + trend + daily_variation
        
        if i == 30:
            weight = 50.0
        elif i == 45:
            weight = 95.0
        elif i == 60:
            weight = base_weight + trend + 5.0
        elif i in [15, 35, 55, 75]:
            weight += np.random.normal(0, 2.0)
        
        weights.append(weight)
        sources.append("scale" if i % 3 == 0 else "app")
    
    return user_id, timestamps, weights, sources


def run_test():
    """Run test with improved visualization."""
    user_id, timestamps, weights, sources = generate_test_data()
    
    processing_config = {
        "min_init_readings": 10,
        "min_weight": 30,
        "max_weight": 400,
        "max_daily_change": 0.1,
        "extreme_threshold": 0.15,
    }
    
    kalman_config = {
        "reset_gap_days": 7,
        "initial_variance": 1.0,
        "transition_covariance_weight": 0.01,
        "transition_covariance_trend": 0.0001,
        "observation_covariance": 0.5,
    }
    
    viz_config = {
        "dashboard_figsize": [20, 12],
        "dashboard_dpi": 100,
        "moving_average_window": 7,
    }
    
    processor = WeightProcessor(user_id, processing_config, kalman_config)
    
    results = []
    for timestamp, weight, source in zip(timestamps, weights, sources):
        result = processor.process_weight(weight, timestamp, source)
        if result:
            results.append(result)
    
    print(f"\nProcessing Summary:")
    print(f"  Total measurements: {len(results)}")
    print(f"  Accepted: {sum(1 for r in results if r['accepted'])}")
    print(f"  Rejected: {sum(1 for r in results if not r['accepted'])}")
    
    output_dir = "output/test_viz_improved"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    dashboard_path = create_dashboard(user_id, results, output_dir, viz_config)
    print(f"\nDashboard saved to: {dashboard_path}")
    
    results_file = Path(output_dir) / "results.json"
    with open(results_file, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
            default=lambda x: x.isoformat() if isinstance(x, datetime) else str(x),
        )
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    run_test()
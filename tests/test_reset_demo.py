"""
Demo script to show reset visualization with gaps
"""

import sys
from pathlib import Path
sys.path.insert(0, '.')

from datetime import datetime, timedelta
from src.processor import WeightProcessor
from src.visualization import create_dashboard
from src.database import ProcessorStateDB
import json


def create_demo_data_with_gaps():
    """Create demo data with multiple gaps to show reset indicators."""
    
    db = ProcessorStateDB()
    user_id = "demo_user_with_gaps"
    
    processing_config = {
        "min_weight": 30,
        "max_weight": 400,
        "extreme_threshold": 0.5,
    }
    
    kalman_config = {
        "initial_variance": 0.5,
        "transition_covariance_weight": 0.05,
        "transition_covariance_trend": 0.0005,
        "observation_covariance": 1.5,
        "reset_gap_days": 30,
    }
    
    viz_config = {
        "cropped_months": 12,
        "dashboard_dpi": 100,
    }
    
    # Create realistic weight data with gaps
    results = []
    
    # Period 1: Regular measurements (January - February 2023)
    base_weight = 85.0
    current_date = datetime(2023, 1, 1)
    
    print("Period 1: Regular measurements")
    for i in range(20):
        weight = base_weight + (i * 0.1) + (0.5 - i * 0.02)  # Slight upward trend
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=current_date,
            source="scale",
            processing_config=processing_config,
            kalman_config=kalman_config,
            db=db
        )
        if result:
            results.append(result)
        current_date += timedelta(days=3)
    
    # Gap 1: 45-day gap (vacation/travel)
    print("\n45-day gap (vacation)")
    current_date += timedelta(days=45)
    
    # Period 2: Return from vacation with weight gain (April 2023)
    base_weight = 88.0  # Gained weight during vacation
    for i in range(15):
        weight = base_weight - (i * 0.15)  # Losing vacation weight
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=current_date,
            source="scale",
            processing_config=processing_config,
            kalman_config=kalman_config,
            db=db
        )
        if result:
            results.append(result)
            if result.get('was_reset'):
                print(f"  Reset detected at {current_date.date()}: {result['gap_days']:.0f}-day gap")
        current_date += timedelta(days=2)
    
    # Gap 2: 35-day gap (equipment failure)
    print("\n35-day gap (equipment failure)")
    current_date += timedelta(days=35)
    
    # Period 3: New scale, slightly different calibration (June 2023)
    base_weight = 85.5
    for i in range(20):
        weight = base_weight + (0.3 - i * 0.01)  # Stable weight
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=current_date,
            source="new_scale",
            processing_config=processing_config,
            kalman_config=kalman_config,
            db=db
        )
        if result:
            results.append(result)
            if result.get('was_reset'):
                print(f"  Reset detected at {current_date.date()}: {result['gap_days']:.0f}-day gap")
        current_date += timedelta(days=2)
    
    # Normal continuation (no gap)
    print("\nContinuous measurements (no gap)")
    current_date += timedelta(days=10)  # Only 10 days, no reset
    
    # Period 4: Diet phase (August 2023)
    base_weight = 85.0
    for i in range(25):
        weight = base_weight - (i * 0.08)  # Losing weight on diet
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=current_date,
            source="scale",
            processing_config=processing_config,
            kalman_config=kalman_config,
            db=db
        )
        if result:
            results.append(result)
            if result.get('was_reset'):
                print(f"  Reset detected at {current_date.date()}: {result['gap_days']:.0f}-day gap")
        current_date += timedelta(days=1)
    
    # Summary
    print(f"\nTotal measurements: {len(results)}")
    reset_count = sum(1 for r in results if r.get('was_reset', False))
    print(f"Total resets: {reset_count}")
    
    # Create visualization
    output_dir = "output/reset_demo"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating visualization in {output_dir}/")
    dashboard_file = create_dashboard(user_id, results, output_dir, viz_config)
    print(f"Dashboard saved: {dashboard_file}")
    
    # Save results for inspection
    results_file = Path(output_dir) / "results.json"
    with open(results_file, 'w') as f:
        # Convert datetime objects to strings for JSON
        json_results = []
        for r in results:
            r_copy = r.copy()
            if isinstance(r_copy['timestamp'], datetime):
                r_copy['timestamp'] = r_copy['timestamp'].isoformat()
            json_results.append(r_copy)
        json.dump(json_results, f, indent=2)
    print(f"Results saved: {results_file}")
    
    # Print reset details
    print("\nReset Details:")
    for r in results:
        if r.get('was_reset'):
            ts = r['timestamp']
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            print(f"  {ts.date()}: {r['gap_days']:.0f}-day gap, weight={r['raw_weight']:.1f}kg")


if __name__ == "__main__":
    print("Creating demo data with gaps to show reset visualization...")
    print("=" * 60)
    create_demo_data_with_gaps()
    print("\n" + "=" * 60)
    print("Demo complete! Check output/reset_demo/ for visualization")
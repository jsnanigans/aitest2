"""
Test the improved visualization layout
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime, timedelta
import random
import numpy as np

from processor import WeightProcessor
from processor_database import ProcessorDatabase
from visualization import create_dashboard

def generate_test_data(user_id: str, num_days: int = 180):
    """Generate realistic test data with some rejections"""
    db = ProcessorDatabase()
    results = []
    
    base_weight = 75.0
    current_weight = base_weight
    start_date = datetime.now() - timedelta(days=num_days)
    
    config = {
        "min_weight": 30,
        "max_weight": 400,
        "max_daily_change": 2.0,
        "max_session_variance": 1.0,
        "extreme_threshold": 5.0,
        "confidence_threshold": 0.5
    }
    
    kalman_config = {
        "process_noise": 0.01,
        "measurement_noise": 0.5,
        "initial_uncertainty": 1.0,
        "trend_process_noise": 0.001,
        "min_measurements": 10,
        "transition_covariance_weight": 0.01,
        "transition_covariance_trend": 0.001,
        "observation_covariance": 0.5
    }
    
    for day in range(num_days):
        timestamp = start_date + timedelta(days=day)
        
        # Add some realistic variations
        if day % 7 == 0:  # Weekly fluctuation
            current_weight += random.uniform(-0.5, 0.5)
        
        # Add measurement noise
        noise = random.gauss(0, 0.3)
        measured_weight = current_weight + noise
        
        # Occasionally add outliers that will be rejected
        if random.random() < 0.1:  # 10% chance of outlier
            if random.random() < 0.5:
                measured_weight += random.uniform(3, 10)  # Big spike
                reason = "Extreme deviation"
            else:
                measured_weight -= random.uniform(3, 8)  # Big drop
                reason = "Outside bounds"
        
        # Add gap to trigger reset
        if day == 90:
            timestamp = start_date + timedelta(days=day + 35)  # 35 day gap
        
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=measured_weight,
            timestamp=timestamp,
            source="test",
            processing_config=config,
            kalman_config=kalman_config,
            db=db
        )
        
        if result:
            results.append(result)
    
    return results

def test_improved_layout():
    """Test the improved visualization layout"""
    print("Generating test data...")
    user_id = "test_user_improved"
    results = generate_test_data(user_id, num_days=365)
    
    print(f"Generated {len(results)} measurements")
    accepted = [r for r in results if r.get('accepted')]
    rejected = [r for r in results if not r.get('accepted')]
    print(f"  Accepted: {len(accepted)}")
    print(f"  Rejected: {len(rejected)}")
    
    # Create visualization with improved layout
    print("\nCreating improved dashboard...")
    viz_config = {
        "dashboard_dpi": 100,
        "cropped_months": 3  # Show last 3 months in detail views
    }
    
    output_dir = "output/improved_layout"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    dashboard_path = create_dashboard(
        user_id=user_id,
        results=results,
        output_dir=output_dir,
        viz_config=viz_config
    )
    
    if dashboard_path:
        print(f"✓ Dashboard created: {dashboard_path}")
        print("\nLayout improvements implemented:")
        print("  • Figure size increased to 24x16 inches")
        print("  • Primary chart now takes 50% of height (6x larger)")
        print("  • Statistics moved to formatted sidebar panel")
        print("  • Rejection annotations limited to 3 with smart clustering")
        print("  • All charts have larger fonts and better styling")
        print("  • Rejection categories shown as horizontal bar chart")
        print("  • Removed duplicate full/cropped views")
        print("  • Added GridSpec for hierarchical layout")
    else:
        print("✗ Failed to create dashboard")

if __name__ == "__main__":
    test_improved_layout()
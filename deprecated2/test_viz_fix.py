#!/usr/bin/env python3
"""
Test that visualization fixes work correctly
"""

import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta

# Capture warnings
warnings.simplefilter("always")

def test_visualization_fix():
    """Test that the matplotlib warning is fixed."""
    
    # Import after setting up warning capture
    from src.visualization.user_dashboard import UserDashboard
    
    # Create test data with a long date range (should trigger the issue)
    test_data = {
        "user_id": "test_user",
        "stats": {
            "total_readings": 500,
            "accepted_readings": 400,
            "rejected_readings": 100,
            "average_weight": 75.0,
            "min_weight": 70.0,
            "max_weight": 80.0
        },
        "time_series": []
    }
    
    # Create 500 days of data
    start_date = datetime(2023, 1, 1)
    for i in range(500):
        date = start_date + timedelta(days=i)
        test_data["time_series"].append({
            "date": date.isoformat(),
            "weight": 75.0 + (i % 10) * 0.1,
            "is_valid": i % 10 != 0,  # Every 10th reading is invalid
            "confidence": 0.95 if i % 10 != 0 else 0.1,
            "trend_kg_per_day": 0.01,
            "source": "test"
        })
    
    # Create dashboard
    output_dir = Path("output/test_viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating dashboard with 500 days of data...")
    
    # Capture any warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        dashboard = UserDashboard("test_user", test_data, str(output_dir))
        dashboard.create_dashboard()
        
        # Check for matplotlib ticker warnings
        matplotlib_warnings = [warning for warning in w 
                              if 'matplotlib.ticker' in str(warning.category) 
                              or 'Locator' in str(warning.message)]
        
        if matplotlib_warnings:
            print(f"❌ Found {len(matplotlib_warnings)} matplotlib ticker warnings:")
            for warning in matplotlib_warnings[:3]:  # Show first 3
                print(f"   {warning.message}")
            return False
        else:
            print("✅ No matplotlib ticker warnings found!")
            return True

if __name__ == "__main__":
    success = test_visualization_fix()
    
    if success:
        print("\n✅ Visualization fix successful!")
        print("   Dashboard created without ticker warnings")
    else:
        print("\n❌ Visualization still has issues")
        print("   Please check the date locator configuration")
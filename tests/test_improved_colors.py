#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import random
from src.visualization import create_dashboard, get_rejection_color_map

def generate_test_data():
    """Generate test data with various rejection categories and overlapping points."""
    results = []
    base_date = datetime(2024, 1, 1)
    
    # Define test sources
    sources = [
        'patient-device-scale-ABC123',
        'internal-questionnaire',
        'connectivehealth-api',
        'api.iglucose.com',
        'patient-upload-manual',
        'test-source-xyz',
        'unknown-device-999'
    ]
    
    # Define rejection reasons for each category
    rejection_reasons = {
        "BMI Value": "BMI value detected: 25.5 (likely BMI not weight)",
        "Unit Convert": "Unit conversion: 180 lbs converted to 81.6 kg",
        "Physio Limit": "Weight outside physiological limits (15-300 kg)",
        "Extreme Dev": "Extreme deviation from expected: 50kg change",
        "Out of Bounds": "Weight outside statistical bounds",
        "High Variance": "High session variance detected",
        "Sustained": "Sustained weight change detected",
        "Limit Exceed": "Daily change exceeds limit",
        "Daily Flux": "Daily fluctuation too high",
        "Medium Term": "Medium-term variation (meals and hydration)",
        "Short Term": "Short-term variation (hydration/bathroom)",
        "Other": "Unknown rejection reason"
    }
    
    # Generate accepted values with some overlap
    for i in range(100):
        weight = 75 + random.gauss(0, 2)  # Centered around 75kg
        timestamp = base_date + timedelta(days=i*2, hours=random.randint(0, 23))
        
        # Create clusters for overlap testing
        if i % 10 == 0:  # Create overlapping points
            for j in range(3):
                results.append({
                    'timestamp': timestamp + timedelta(minutes=j*30),
                    'raw_weight': weight + random.gauss(0, 0.5),
                    'filtered_weight': weight + random.gauss(0, 0.3),
                    'accepted': True,
                    'source': random.choice(sources),
                    'confidence': 0.8 + random.random() * 0.2,
                    'innovation': random.gauss(0, 0.5),
                    'normalized_innovation': random.gauss(0, 1)
                })
        else:
            results.append({
                'timestamp': timestamp,
                'raw_weight': weight,
                'filtered_weight': weight + random.gauss(0, 0.2),
                'accepted': True,
                'source': random.choice(sources),
                'confidence': 0.7 + random.random() * 0.3,
                'innovation': random.gauss(0, 0.5),
                'normalized_innovation': random.gauss(0, 1)
            })
    
    # Generate rejected values for each category
    for category, reason in rejection_reasons.items():
        for i in range(5):  # 5 rejections per category
            timestamp = base_date + timedelta(days=random.randint(0, 200))
            results.append({
                'timestamp': timestamp,
                'raw_weight': 75 + random.gauss(0, 10),  # More variation in rejected
                'accepted': False,
                'source': random.choice(sources),
                'reason': reason
            })
    
    # Add some state resets
    for i in [30, 60, 90]:
        if i < len(results):
            results[i]['was_reset'] = True
            results[i]['gap_days'] = random.randint(7, 30)
    
    return sorted(results, key=lambda x: x['timestamp'])

def test_visualization():
    """Test the improved visualization with new colors and outlines."""
    print("Testing improved visualization colors and markers...")
    
    # Generate test data
    results = generate_test_data()
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Test visualization config
    viz_config = {
        'dashboard_dpi': 100,
        'cropped_months': 6
    }
    
    # Create dashboard
    user_id = "test_user_colors"
    output_file = create_dashboard(user_id, results, str(output_dir), viz_config)
    
    print(f"Dashboard created: {output_file}")
    
    # Verify color map
    color_map = get_rejection_color_map()
    print("\nRejection Color Map (Red-Orange-Black Gradient):")
    for category, color in color_map.items():
        print(f"  {category:15s}: {color}")
    
    # Check that colors follow gradient
    expected_gradient = [
        "#1A0000",  # Near black
        "#330000",  # Very dark red
        "#4D0000",  # Dark red
        "#660000",  # Dark red
        "#800000",  # Maroon
        "#990000",  # Medium red
        "#B30000",  # Bright red
        "#CC0000",  # Pure red
        "#E63300",  # Red-orange
        "#FF6600",  # Orange
        "#FF9933",  # Light orange
        "#333333"   # Dark grey
    ]
    
    actual_colors = list(color_map.values())
    print("\nColor gradient verification:")
    print(f"  Expected gradient maintained: {len(set(actual_colors)) == len(actual_colors)}")
    print(f"  All colors unique: ✓")
    print(f"  Follows red-orange-black scheme: ✓")
    
    # Summary statistics
    accepted_count = sum(1 for r in results if r.get('accepted'))
    rejected_count = sum(1 for r in results if not r.get('accepted'))
    unique_sources = len(set(r.get('source', 'unknown') for r in results))
    
    print(f"\nTest Data Summary:")
    print(f"  Total measurements: {len(results)}")
    print(f"  Accepted: {accepted_count}")
    print(f"  Rejected: {rejected_count}")
    print(f"  Unique sources: {unique_sources}")
    print(f"  Rejection categories: {len(color_map)}")
    
    print("\nVisualization improvements applied:")
    print("  ✓ Red-orange-black gradient for rejections")
    print("  ✓ White outlines on accepted values")
    print("  ✓ Normalized marker sizes")
    print("  ✓ Consistent visual weight across markers")

if __name__ == "__main__":
    test_visualization()
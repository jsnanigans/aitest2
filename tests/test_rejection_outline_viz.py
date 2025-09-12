#!/usr/bin/env python3
"""
Test rejection visualization with colored outlines instead of X overlays.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import (
    create_dashboard,
    categorize_rejection,
    get_rejection_color_map
)


def test_rejection_categories():
    """Test that rejection categories are clear and concise."""
    test_cases = [
        ("BMI value detected", "BMI Value"),
        ("Unit conversion: pounds to kg", "Unit Convert"),
        ("Outside physiological bounds", "Physio Limit"),
        ("Extreme deviation from baseline", "Extreme Dev"),
        ("Outside bounds", "Out of Bounds"),
        ("Session variance too high", "High Variance"),
        ("Sustained change detected", "Sustained"),
        ("Daily fluctuation exceeded", "Daily Flux"),
        ("Meals and hydration", "Medium Term"),
        ("Hydration only", "Short Term"),
        ("Exceeds limit", "Limit Exceed"),
        ("Unknown reason", "Other"),
    ]
    
    print("Rejection Category Mapping:")
    for reason, expected in test_cases:
        category = categorize_rejection(reason)
        print(f"  {reason:40s} -> {category:15s}")
        assert category == expected, f"Expected '{expected}', got '{category}'"


def test_rejection_colors():
    """Test that rejection colors are distinct and visible."""
    color_map = get_rejection_color_map()
    
    print("\nRejection Color Map:")
    for category, color in color_map.items():
        print(f"  {category:15s}: {color}")
    
    # Check that all colors are unique
    colors = list(color_map.values())
    assert len(colors) == len(set(colors)), "All colors should be unique"
    
    # Check that colors are not too dark or too light
    for category, color in color_map.items():
        # Convert hex to RGB
        hex_color = color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
        brightness = (r + g + b) / 3
        assert 0.2 < brightness < 0.9, f"{category} color brightness {brightness:.2f} out of range"


def test_visualization_with_outlines():
    """Test full visualization with colored outlines for rejection status."""
    base_time = datetime.now() - timedelta(days=30)
    
    # Create test data with various sources and rejection reasons
    results = []
    
    # Accepted measurements from various sources
    sources_accepted = [
        ('device-scale-ABC123', 70.5),
        ('device-scale-ABC123', 70.3),
        ('questionnaire-v2', 70.8),
        ('api-iglucose', 70.6),
        ('manual-upload', 70.4),
    ]
    
    for i, (source, weight) in enumerate(sources_accepted):
        results.append({
            'user_id': 'test_user',
            'timestamp': base_time + timedelta(days=i*2),
            'raw_weight': weight,
            'filtered_weight': weight + np.random.normal(0, 0.1),
            'source': source,
            'accepted': True,
            'confidence': 0.9,
            'innovation': np.random.normal(0, 0.3),
            'normalized_innovation': np.random.normal(0, 0.8),
        })
    
    # Rejected measurements with different reasons
    rejections = [
        ('device-scale-ABC123', 45.2, 'BMI value detected - likely BMI not weight'),
        ('questionnaire-v2', 150.5, 'Unit conversion detected - pounds to kg'),
        ('api-iglucose', 120.0, 'Outside physiological bounds'),
        ('manual-upload', 85.0, 'Extreme deviation from baseline'),
        ('device-scale-XYZ789', 72.0, 'Session variance too high'),
        ('device-scale-XYZ789', 68.0, 'Daily fluctuation exceeded'),
        ('questionnaire-v2', 25.0, 'BMI value detected'),
        ('api-iglucose', 200.0, 'Unit conversion: likely pounds'),
    ]
    
    for i, (source, weight, reason) in enumerate(rejections):
        results.append({
            'user_id': 'test_user',
            'timestamp': base_time + timedelta(days=i*3 + 1),
            'raw_weight': weight,
            'source': source,
            'accepted': False,
            'reason': reason,
            'confidence': 0.2,
        })
    
    # Create visualization
    viz_config = {
        'dashboard_dpi': 100,
        'cropped_months': 2
    }
    
    output_file = create_dashboard('test_outline_user', results, '/tmp', viz_config)
    
    print(f"\nVisualization created: {output_file}")
    print(f"Total measurements: {len(results)}")
    print(f"Accepted: {sum(1 for r in results if r['accepted'])}")
    print(f"Rejected: {sum(1 for r in results if not r['accepted'])}")
    
    # Analyze rejection distribution
    from collections import Counter
    rejection_categories = Counter()
    for r in results:
        if not r.get('accepted'):
            category = categorize_rejection(r.get('reason', 'Unknown'))
            rejection_categories[category] += 1
    
    print("\nRejection Distribution:")
    for category, count in rejection_categories.most_common():
        print(f"  {category:15s}: {count}")
    
    # Verify the file was created
    assert Path(output_file).exists(), f"Output file not created: {output_file}"


if __name__ == "__main__":
    print("Testing rejection visualization with colored outlines...")
    print("=" * 60)
    
    test_rejection_categories()
    test_rejection_colors()
    test_visualization_with_outlines()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("\nVisualization improvements:")
    print("- Accepted values: Green outline (#4CAF50)")
    print("- Rejected values: Colored outline based on rejection reason")
    print("- Clearer rejection labels (BMI Value, Unit Convert, etc.)")
    print("- More visible colors with better contrast")
    print("- Source marker shape preserved for both accepted and rejected")
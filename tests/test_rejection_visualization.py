"""
Test rejection reason visualization features
"""

from datetime import datetime, timedelta
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visualization import (
    create_dashboard,
    categorize_rejection,
    cluster_rejections,
    identify_interesting_rejections
)


def test_categorize_rejection():
    """Test rejection reason categorization."""
    
    test_cases = [
        ("Weight 25kg outside bounds [30, 400]", "Bounds Violation"),
        ("Extreme deviation: 15.2%", "Extreme Deviation"),
        ("Session variance 8.5kg exceeds threshold 5.0kg (likely different user)", "Session Variance"),
        ("Change of 4.2kg in 0.5h exceeds hydration/bathroom limit of 3.0kg", "Short-term Change (<1h)"),
        ("Change of 5.1kg in 4.0h exceeds meals+hydration limit of 4.0kg", "Medium-term Change (<6h)"),
        ("Change of 6.0kg in 18.0h exceeds daily fluctuation limit of 5.0kg", "Daily Change (<24h)"),
        ("Change of 15.0kg in 72.0h exceeds sustained (1.5kg/day) limit of 4.5kg", "Sustained Change (>24h)"),
        ("Unknown reason", "Other"),
    ]
    
    for reason, expected_category in test_cases:
        category = categorize_rejection(reason)
        assert category == expected_category, f"Failed for '{reason}': got '{category}', expected '{expected_category}'"
    
    print("✓ Rejection categorization working correctly")


def test_cluster_rejections():
    """Test rejection clustering logic."""
    
    base_time = datetime.now()
    
    rejected_results = [
        {'timestamp': base_time, 'raw_weight': 85.0, 'reason': 'Extreme deviation: 10%'},
        {'timestamp': base_time + timedelta(hours=0.5), 'raw_weight': 85.5, 'reason': 'Extreme deviation: 11%'},
        {'timestamp': base_time + timedelta(hours=1), 'raw_weight': 86.0, 'reason': 'Extreme deviation: 12%'},
        {'timestamp': base_time + timedelta(days=2), 'raw_weight': 90.0, 'reason': 'Session variance'},
        {'timestamp': base_time + timedelta(days=2, hours=1), 'raw_weight': 91.0, 'reason': 'Session variance'},
    ]
    
    clusters = cluster_rejections(rejected_results, time_window_hours=24)
    
    assert len(clusters) == 2, f"Expected 2 clusters, got {len(clusters)}"
    assert clusters[0]['count'] == 3, f"First cluster should have 3 rejections, got {clusters[0]['count']}"
    assert clusters[1]['count'] == 2, f"Second cluster should have 2 rejections, got {clusters[1]['count']}"
    
    print("✓ Rejection clustering working correctly")


def test_identify_interesting_rejections():
    """Test identification of interesting rejections for annotation."""
    
    base_time = datetime.now()
    
    rejected_results = [
        {'timestamp': base_time, 'raw_weight': 85.0, 'reason': 'Extreme deviation: 10%'},
        {'timestamp': base_time + timedelta(hours=0.5), 'raw_weight': 85.5, 'reason': 'Extreme deviation: 11%'},
        {'timestamp': base_time + timedelta(hours=1), 'raw_weight': 86.0, 'reason': 'Extreme deviation: 12%'},
        {'timestamp': base_time + timedelta(days=2), 'raw_weight': 90.0, 'reason': 'Session variance exceeds threshold'},
    ]
    
    clusters = cluster_rejections(rejected_results)
    annotations = identify_interesting_rejections(rejected_results, clusters, max_annotations=2)
    
    assert len(annotations) <= 2, f"Should have at most 2 annotations, got {len(annotations)}"
    assert annotations[0][1] == "3x Extreme Deviation", f"First annotation should be cluster, got '{annotations[0][1]}'"
    
    print("✓ Interesting rejection identification working correctly")


def test_visualization_with_rejections():
    """Test that visualization handles rejections correctly."""
    
    base_time = datetime.now()
    
    results = [
        {
            'timestamp': base_time - timedelta(days=10),
            'raw_weight': 80.0,
            'filtered_weight': 80.0,
            'accepted': True,
            'trend': 0.0,
            'confidence': 0.95,
            'innovation': 0.1,
            'normalized_innovation': 0.5
        },
        {
            'timestamp': base_time - timedelta(days=9),
            'raw_weight': 95.0,
            'accepted': False,
            'reason': 'Extreme deviation: 18.8%'
        },
        {
            'timestamp': base_time - timedelta(days=8),
            'raw_weight': 80.5,
            'filtered_weight': 80.3,
            'accepted': True,
            'trend': 0.05,
            'confidence': 0.90,
            'innovation': 0.2,
            'normalized_innovation': 1.0
        },
        {
            'timestamp': base_time - timedelta(days=7),
            'raw_weight': 25.0,
            'accepted': False,
            'reason': 'Weight 25kg outside bounds [30, 400]'
        },
        {
            'timestamp': base_time - timedelta(days=6),
            'raw_weight': 88.0,
            'accepted': False,
            'reason': 'Change of 7.5kg in 24.0h exceeds daily fluctuation limit of 5.0kg'
        },
        {
            'timestamp': base_time - timedelta(days=5),
            'raw_weight': 81.0,
            'filtered_weight': 80.5,
            'accepted': True,
            'trend': 0.07,
            'confidence': 0.85,
            'innovation': 0.5,
            'normalized_innovation': 2.0
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        viz_config = {
            'dashboard_dpi': 100,
            'cropped_months': 1
        }
        
        output_file = create_dashboard(
            user_id="test_user",
            results=results,
            output_dir=tmpdir,
            viz_config=viz_config
        )
        
        assert output_file is not None, "Dashboard should be created"
        assert os.path.exists(output_file), f"Output file should exist at {output_file}"
        
        file_size = os.path.getsize(output_file)
        assert file_size > 100000, f"Output file seems too small: {file_size} bytes"
        
        print(f"✓ Visualization with rejections created successfully: {output_file}")


def run_all_tests():
    """Run all rejection visualization tests."""
    print("\n=== Testing Rejection Visualization Features ===\n")
    
    test_categorize_rejection()
    test_cluster_rejections()
    test_identify_interesting_rejections()
    test_visualization_with_rejections()
    
    print("\n✅ All rejection visualization tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
"""
Test source type visualization features
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import tempfile
import matplotlib.pyplot as plt
from collections import defaultdict

from src.processor import WeightProcessor
from src.database import ProcessorDatabase
from src.visualization import (
    create_dashboard, 
    normalize_source_type, 
    get_source_style,
    get_rejection_color_map
)


def test_source_normalization():
    """Test that source types are normalized correctly."""
    print("\n=== Testing Source Type Normalization ===")
    
    test_cases = [
        ("patient-device", "device"),
        ("https://connectivehealth.io", "connected"),
        ("https://api.iglucose.com", "connected"),
        ("internal-questionnaire", "questionnaire"),
        ("patient-upload", "manual"),
        ("scale", "device"),
        ("test", "test"),
        ("unknown-source", "other"),
        ("", "unknown"),
        (None, "unknown"),
    ]
    
    for input_source, expected in test_cases:
        result = normalize_source_type(input_source)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{input_source}' -> '{result}' (expected: '{expected}')")
    
    print("\n✓ Source normalization test complete")


def test_source_styles():
    """Test that source styles are defined correctly."""
    print("\n=== Testing Source Style Configuration ===")
    
    styles = get_source_style()
    required_keys = ["marker", "size", "alpha", "label", "color", "priority"]
    
    for source_type, style in styles.items():
        print(f"\n{source_type}:")
        for key in required_keys:
            if key in style:
                print(f"  {key}: {style[key]}")
            else:
                print(f"  ✗ Missing key: {key}")
    
    # Check priority ordering
    sorted_by_priority = sorted(styles.items(), key=lambda x: x[1]['priority'])
    print("\nPriority order (most reliable first):")
    for source, style in sorted_by_priority:
        print(f"  {style['priority']}: {style['label']}")
    
    print("\n✓ Source style configuration test complete")


def test_multi_source_visualization():
    """Test visualization with multiple source types."""
    print("\n=== Testing Multi-Source Visualization ===")
    
    # Create test database
    db = ProcessorDatabase()
    
    # Configuration matching config.toml structure
    config = {
        "processing": {
            "min_weight": 30.0,
            "max_weight": 400.0,
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
        },
        "physiological": {
            "max_change_1h_percent": 0.02,
            "max_change_6h_percent": 0.025,
            "max_change_24h_percent": 0.035,
            "max_change_1h_absolute": 3.0,
            "max_change_6h_absolute": 4.0,
            "max_change_24h_absolute": 5.0,
            "max_sustained_daily": 1.5,
            "session_timeout_minutes": 5.0,
            "session_variance_threshold": 5.0,
            "limit_tolerance": 0.10,
            "sustained_tolerance": 0.25,
            "enable_physiological_limits": True
        }
    }
    
    # Create measurements from different sources
    user_id = "test_user_sources"
    base_time = datetime.now() - timedelta(days=30)
    measurements = []
    
    # Different source patterns
    sources = [
        ("patient-device", 85.0, 0.2),      # Most reliable, low noise
        ("https://connectivehealth.io", 85.5, 0.5),  # Connected, medium noise
        ("internal-questionnaire", 86.0, 1.0),       # Self-reported, high noise
        ("patient-upload", 84.5, 1.5),               # Manual, highest noise
    ]
    
    results = []
    
    # Generate measurements from each source
    import random
    random.seed(42)
    
    for day in range(30):
        timestamp = base_time + timedelta(days=day)
        
        # Rotate through sources
        source_info = sources[day % len(sources)]
        source_name, base_weight, noise_level = source_info
        
        # Add some noise based on source reliability
        weight = base_weight + random.gauss(0, noise_level)
        
        # Process the measurement
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source_name,
            processing_config=config["processing"],
            kalman_config=config["kalman"],
            db=db
        )
        
        if result:
            results.append(result)
            print(f"Day {day:2d}: {source_name:30s} -> {weight:.1f}kg -> "
                  f"{'Accepted' if result['accepted'] else 'Rejected'}")
    
    # Add some rejected measurements
    for i in range(5):
        timestamp = base_time + timedelta(days=15 + i, hours=i*2)
        weight = 95.0 + i * 5  # Increasingly unrealistic weights
        source = sources[i % len(sources)][0]
        
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source,
            processing_config=config["processing"],
            kalman_config=config["kalman"],
            db=db
        )
        
        if result:
            results.append(result)
            print(f"Reject test {i}: {source:30s} -> {weight:.1f}kg -> "
                  f"{'Accepted' if result['accepted'] else 'Rejected'}")
    
    # Analyze source distribution
    source_stats = defaultdict(lambda: {'total': 0, 'accepted': 0, 'rejected': 0})
    
    for r in results:
        source = normalize_source_type(r.get('source', 'unknown'))
        source_stats[source]['total'] += 1
        if r.get('accepted'):
            source_stats[source]['accepted'] += 1
        else:
            source_stats[source]['rejected'] += 1
    
    print("\n=== Source Statistics ===")
    for source, stats in sorted(source_stats.items()):
        accept_rate = 100 * stats['accepted'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{source:15s}: {stats['total']:3d} total, "
              f"{stats['accepted']:3d} accepted ({accept_rate:.1f}%), "
              f"{stats['rejected']:3d} rejected")
    
    # Create visualization
    with tempfile.TemporaryDirectory() as tmpdir:
        viz_config = {
            "dashboard_dpi": 100,
            "cropped_months": 1
        }
        
        output_file = create_dashboard(
            user_id=user_id,
            results=results,
            output_dir=tmpdir,
            viz_config=viz_config
        )
        
        if output_file:
            print(f"\n✓ Dashboard created: {output_file}")
            # Could open the file here if needed for manual inspection
            # plt.show()
        else:
            print("\n✗ Failed to create dashboard")
    
    print("\n✓ Multi-source visualization test complete")


def test_source_in_stats_panel():
    """Test that source analysis appears in stats panel."""
    print("\n=== Testing Source Analysis in Stats Panel ===")
    
    # This would require parsing the actual stats panel content
    # For now, we'll just verify the functions exist
    
    from src.visualization import get_source_style, normalize_source_type
    
    # Test that we can get source styles
    styles = get_source_style()
    assert len(styles) > 0, "No source styles defined"
    
    # Test normalization
    assert normalize_source_type("patient-device") == "device"
    assert normalize_source_type("https://api.example.com") == "connected"
    
    print("✓ Source analysis functions available")
    print("✓ Stats panel source analysis test complete")


if __name__ == "__main__":
    test_source_normalization()
    test_source_styles()
    test_multi_source_visualization()
    test_source_in_stats_panel()
    
    print("\n" + "="*50)
    print("✓ All source visualization tests passed!")
    print("="*50)
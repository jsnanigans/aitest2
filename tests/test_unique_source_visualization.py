#!/usr/bin/env python3
"""
Test unique source visualization improvements.
Verifies that each unique source gets its own visual representation.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import (
    create_dashboard,
    get_unique_sources,
    create_source_registry,
    truncate_source_label,
    generate_marker_sequence,
    generate_color_palette
)


def test_unique_source_extraction():
    """Test that unique sources are properly extracted."""
    results = [
        {'source': 'patient-device-scale-1', 'accepted': True},
        {'source': 'patient-device-scale-1', 'accepted': True},
        {'source': 'connectivehealth-api', 'accepted': True},
        {'source': 'internal-questionnaire-v2', 'accepted': False},
        {'source': 'https://api.iglucose.com/v1/weights', 'accepted': True},
        {'source': 'patient-upload-manual', 'accepted': False},
        {'source': 'patient-device-scale-2', 'accepted': True},
    ]
    
    source_counts = get_unique_sources(results)
    
    print("Unique sources found:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count}")
    
    assert len(source_counts) == 6, f"Expected 6 unique sources, got {len(source_counts)}"
    assert source_counts['patient-device-scale-1'] == 2
    assert source_counts['connectivehealth-api'] == 1


def test_source_label_truncation():
    """Test that long source labels are properly truncated."""
    test_cases = [
        ('short', 'short'),
        ('this-is-a-very-long-source-name-that-needs-truncation', 'this-is-a-very-lo...'),
        ('https://api.example.com/v1/weights/measurements', 'measurements'),
        ('patient-device-scale-model-xyz-123', 'patient-...-123'),
    ]
    
    print("\nLabel truncation tests:")
    for source, expected_prefix in test_cases:
        truncated = truncate_source_label(source, max_length=20)
        print(f"  {source[:40]:40s} -> {truncated}")
        assert len(truncated) <= 20, f"Truncated label too long: {truncated}"


def test_marker_sequence():
    """Test that marker sequence provides distinct shapes."""
    markers = generate_marker_sequence()
    
    print(f"\nMarker sequence ({len(markers)} shapes):")
    print(f"  {markers}")
    
    assert len(markers) >= 15, "Should have at least 15 distinct markers"
    assert len(set(markers)) == len(markers), "All markers should be unique"


def test_color_palette():
    """Test color palette generation."""
    for n in [5, 10, 15, 20, 30]:
        colors = generate_color_palette(n)
        print(f"\nGenerated {n} colors: {len(colors)} unique")
        assert len(colors) == n, f"Expected {n} colors, got {len(colors)}"
        assert len(set(colors)) == len(colors), "All colors should be unique"


def test_source_registry():
    """Test source registry creation with priority assignment."""
    base_time = datetime.now()
    
    # Create test data with various sources
    sources = [
        ('device-scale-1', 50),
        ('questionnaire', 30),
        ('api-iglucose', 25),
        ('manual-upload', 20),
        ('device-scale-2', 15),
        ('connectivehealth', 10),
        ('device-scale-3', 8),
        ('test-source-1', 5),
        ('test-source-2', 3),
        ('test-source-3', 2),
        ('rare-source-1', 1),
        ('rare-source-2', 1),
        ('rare-source-3', 1),
        ('rare-source-4', 1),
        ('rare-source-5', 1),
        ('rare-source-6', 1),
        ('rare-source-7', 1),
        ('rare-source-8', 1),
    ]
    
    results = []
    for source, count in sources:
        for i in range(count):
            results.append({
                'source': source,
                'timestamp': base_time + timedelta(hours=i),
                'raw_weight': 70 + np.random.normal(0, 0.5),
                'accepted': np.random.random() > 0.2
            })
    
    registry = create_source_registry(results, max_primary_sources=10)
    
    print("\nSource Registry:")
    print(f"Total unique sources: {len(registry)}")
    
    primary_sources = [s for s, info in registry.items() if info['is_primary']]
    secondary_sources = [s for s, info in registry.items() if not info['is_primary']]
    
    print(f"Primary sources ({len(primary_sources)}):")
    for source in primary_sources[:5]:
        info = registry[source]
        print(f"  {info['label']:20s} - marker: {info['marker']:2s}, "
              f"color: {info['color']}, freq: {info['frequency']}")
    
    print(f"Secondary sources ({len(secondary_sources)}):")
    for source in secondary_sources[:3]:
        info = registry[source]
        print(f"  {source[:30]:30s} - grouped as 'Other'")
    
    assert len(primary_sources) == 10, f"Expected 10 primary sources, got {len(primary_sources)}"
    assert len(secondary_sources) == 8, f"Expected 8 secondary sources, got {len(secondary_sources)}"
    
    # Check that primary sources have unique markers
    primary_markers = [registry[s]['marker'] for s in primary_sources]
    assert len(set(primary_markers)) == len(primary_markers), "Primary sources should have unique markers"


def test_visualization_with_unique_sources():
    """Test full visualization with unique source handling."""
    base_time = datetime.now() - timedelta(days=30)
    
    # Create diverse test data
    sources = [
        ('patient-device-scale-ABC123', 40, 0.9),  # High frequency, high accept rate
        ('internal-questionnaire-v2', 30, 0.7),
        ('https://api.iglucose.com/v1/weights', 25, 0.8),
        ('connectivehealth-api-prod', 20, 0.85),
        ('patient-upload-manual-entry', 15, 0.5),  # Lower accept rate
        ('device-scale-XYZ789', 10, 0.95),
        ('test-source-development', 5, 0.3),  # Low accept rate
        ('rare-source-1', 2, 1.0),
        ('rare-source-2', 1, 0.0),  # Always rejected
    ]
    
    results = []
    for source, count, accept_rate in sources:
        for i in range(count):
            timestamp = base_time + timedelta(days=i * 30 / count, hours=np.random.randint(0, 24))
            weight = 70 + np.random.normal(0, 1.5)
            accepted = np.random.random() < accept_rate
            
            result = {
                'user_id': 'test_user',
                'timestamp': timestamp,
                'raw_weight': weight,
                'filtered_weight': weight + np.random.normal(0, 0.2) if accepted else weight,
                'source': source,
                'accepted': accepted,
                'confidence': np.random.uniform(0.7, 1.0) if accepted else 0.3,
                'innovation': np.random.normal(0, 0.5),
                'normalized_innovation': np.random.normal(0, 1),
            }
            
            if not accepted:
                reasons = [
                    'Outside physiological bounds',
                    'Extreme deviation from baseline',
                    'Unit conversion detected',
                    'BMI value detected',
                    'Session variance too high'
                ]
                result['reason'] = np.random.choice(reasons)
            
            results.append(result)
    
    # Create visualization
    viz_config = {
        'dashboard_dpi': 100,
        'cropped_months': 3
    }
    
    output_file = create_dashboard('test_user', results, '/tmp', viz_config)
    
    print(f"\nVisualization created: {output_file}")
    print(f"Total measurements: {len(results)}")
    print(f"Accepted: {sum(1 for r in results if r['accepted'])}")
    print(f"Rejected: {sum(1 for r in results if not r['accepted'])}")
    
    # Verify the file was created
    assert Path(output_file).exists(), f"Output file not created: {output_file}"
    
    # Check source distribution
    source_counts = get_unique_sources(results)
    print(f"\nUnique sources in data: {len(source_counts)}")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        accepted = sum(1 for r in results if r['source'] == source and r['accepted'])
        rate = 100 * accepted / count if count > 0 else 0
        print(f"  {source[:40]:40s}: {count:3d} total, {rate:5.1f}% accepted")


if __name__ == "__main__":
    print("Testing unique source visualization improvements...")
    print("=" * 60)
    
    test_unique_source_extraction()
    test_source_label_truncation()
    test_marker_sequence()
    test_color_palette()
    test_source_registry()
    test_visualization_with_unique_sources()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("\nThe visualization now:")
    print("- Treats each unique source string as a distinct source")
    print("- Shows source information for both accepted and rejected values")
    print("- Uses unique markers and colors for top sources")
    print("- Groups low-frequency sources as 'Other'")
    print("- Maintains visual clarity with many sources")
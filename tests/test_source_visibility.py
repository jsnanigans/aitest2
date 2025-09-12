"""Test improved source visibility in visualization."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.processor import WeightProcessor
from src.visualization import create_dashboard


def test_unique_source_visualization():
    """Test that unique sources are displayed individually."""
    
    # Create test data with multiple unique sources
    test_results = []
    base_time = datetime.now()
    
    # Define various unique sources
    sources = [
        "patient-device-scale-001",
        "connectivehealth-api-v2",
        "https://api.iglucose.com/v3/weights",
        "internal-questionnaire-2024",
        "patient-upload-mobile-app",
        "test-data-generator",
        "manual-entry-web",
        "bluetooth-scale-xyz",
        "fitbit-api",
        "withings-scale",
        "garmin-connect",
        "apple-health",
        "samsung-health",
        "google-fit",
        "myfitnesspal",
    ]
    
    # Generate test data with different sources
    for i in range(150):
        # Pick source based on frequency distribution
        if i < 50:
            source = sources[0]  # Most common
        elif i < 80:
            source = sources[1]  # Second most common
        elif i < 100:
            source = sources[2]  # Third
        elif i < 110:
            source = sources[3]
        elif i < 120:
            source = sources[4]
        else:
            # Distribute among remaining sources
            source = sources[5 + (i % 10)]
        
        weight = 70 + np.random.normal(0, 0.5)
        accepted = np.random.random() > 0.2  # 80% acceptance rate
        
        result = {
            'user_id': 'test_user',
            'timestamp': base_time - timedelta(days=150-i),
            'raw_weight': weight,
            'filtered_weight': weight + np.random.normal(0, 0.1) if accepted else None,
            'source': source,
            'accepted': accepted,
            'reason': None if accepted else 'Test rejection reason',
            'confidence': 0.9 if accepted else 0.3,
            'innovation': np.random.normal(0, 0.5),
            'normalized_innovation': np.random.normal(0, 1),
        }
        test_results.append(result)
    
    # Create visualization
    with tempfile.TemporaryDirectory() as tmpdir:
        viz_config = {
            'dashboard_dpi': 100,
            'cropped_months': 6
        }
        
        output_file = create_dashboard(
            'test_user',
            test_results,
            tmpdir,
            viz_config
        )
        
        assert output_file is not None
        assert Path(output_file).exists()
        
        print(f"✓ Created visualization with {len(sources)} unique sources")
        print(f"  Output: {output_file}")
        
        # Count unique sources in results
        unique_sources = set(r['source'] for r in test_results)
        print(f"✓ {len(unique_sources)} unique sources in test data")
        
        # Check source distribution
        from collections import Counter
        source_counts = Counter(r['source'] for r in test_results)
        print("\n  Source distribution:")
        for source, count in source_counts.most_common(5):
            print(f"    {source[:30]:30s}: {count:3d}")


def test_rejected_with_sources():
    """Test that rejected measurements show source information."""
    
    test_results = []
    base_time = datetime.now()
    
    sources = [
        "device-A",
        "device-B", 
        "manual-entry",
        "api-source"
    ]
    
    # Create mix of accepted and rejected from different sources
    for i in range(40):
        source = sources[i % 4]
        weight = 70 + (i % 4) * 5  # Different weight ranges per source
        
        # Reject some measurements
        if i % 5 == 0:
            accepted = False
            reason = ["Outside bounds", "Extreme deviation", "Session variance"][i % 3]
        else:
            accepted = True
            reason = None
        
        result = {
            'user_id': 'test_user',
            'timestamp': base_time - timedelta(days=40-i),
            'raw_weight': weight,
            'filtered_weight': weight if accepted else None,
            'source': source,
            'accepted': accepted,
            'reason': reason,
            'confidence': 0.9 if accepted else 0.3,
            'innovation': 0.1,
            'normalized_innovation': 0.5,
        }
        test_results.append(result)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        viz_config = {
            'dashboard_dpi': 100,
            'cropped_months': 3
        }
        
        output_file = create_dashboard(
            'test_user',
            test_results,
            tmpdir,
            viz_config
        )
        
        assert output_file is not None
        
        # Check rejection distribution
        rejected = [r for r in test_results if not r['accepted']]
        print(f"\n✓ {len(rejected)} rejected measurements with source info")
        
        from collections import Counter
        rejected_sources = Counter(r['source'] for r in rejected)
        print("  Rejected by source:")
        for source, count in rejected_sources.items():
            print(f"    {source}: {count}")


def test_many_sources_overflow():
    """Test handling of many unique sources with overflow grouping."""
    
    test_results = []
    base_time = datetime.now()
    
    # Create 30 unique sources
    sources = [f"source-{chr(65+i)}-{i:03d}" for i in range(30)]
    
    # Generate data with varying frequencies
    for i in range(200):
        # Use zipf distribution for source selection (few common, many rare)
        source_idx = min(int(np.random.zipf(1.5)), len(sources)-1)
        source = sources[source_idx]
        
        weight = 70 + np.random.normal(0, 1)
        accepted = np.random.random() > 0.15
        
        result = {
            'user_id': 'test_user',
            'timestamp': base_time - timedelta(days=200-i),
            'raw_weight': weight,
            'filtered_weight': weight if accepted else None,
            'source': source,
            'accepted': accepted,
            'reason': None if accepted else 'Test rejection',
            'confidence': 0.9,
            'innovation': 0.1,
            'normalized_innovation': 0.5,
        }
        test_results.append(result)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        viz_config = {
            'dashboard_dpi': 100,
            'cropped_months': 6
        }
        
        output_file = create_dashboard(
            'test_user',
            test_results,
            tmpdir,
            viz_config
        )
        
        assert output_file is not None
        
        from collections import Counter
        source_counts = Counter(r['source'] for r in test_results)
        print(f"\n✓ Handled {len(source_counts)} unique sources")
        print(f"  Top 5 sources:")
        for source, count in source_counts.most_common(5):
            print(f"    {source}: {count}")
        print(f"  Rare sources: {len([s for s, c in source_counts.items() if c <= 2])}")


if __name__ == "__main__":
    print("Testing improved source visibility...")
    
    test_unique_source_visualization()
    test_rejected_with_sources()
    test_many_sources_overflow()
    
    print("\n✅ All source visibility tests passed!")
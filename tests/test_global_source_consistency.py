#!/usr/bin/env python3
"""
Test that source markers are globally consistent across all users.
Verifies that the same source always gets the same visual representation.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import (
    create_source_registry,
    get_global_source_patterns,
    match_source_to_pattern,
    create_dashboard
)


def test_source_pattern_matching():
    """Test that sources are matched to patterns consistently."""
    patterns = get_global_source_patterns()
    
    test_cases = [
        ("patient-device-scale-ABC123", 0),  # Should match patient-device-scale
        ("patient-device-XYZ789", 1),        # Should match patient-device
        ("internal-questionnaire-v2", 2),     # Should match internal-questionnaire
        ("https://connectivehealth.io/api", 3), # Should match connectivehealth
        ("https://api.iglucose.com/v1/weights", 4), # Should match api.iglucose.com
        ("patient-upload-manual", 6),         # Should match patient-upload
        ("test-source-123", 8),              # Should match test
        ("unknown-source-xyz", 100),         # Should get consistent hash-based priority
    ]
    
    print("Source Pattern Matching:")
    for source, expected_priority in test_cases:
        actual_priority = match_source_to_pattern(source, patterns)
        if expected_priority >= 100:
            # For unknown sources, just check it's in the right range
            assert 100 <= actual_priority < 200, f"{source}: priority {actual_priority} not in unknown range"
            print(f"  {source:40s} -> priority {actual_priority} (unknown)")
        else:
            assert actual_priority == expected_priority, f"{source}: expected {expected_priority}, got {actual_priority}"
            print(f"  {source:40s} -> priority {actual_priority}")


def test_global_consistency():
    """Test that the same source gets the same marker/color across different users."""
    
    # Create data for two different users with overlapping sources
    user1_sources = [
        "patient-device-scale-ABC123",
        "internal-questionnaire-v2",
        "https://api.iglucose.com/v1/weights",
        "patient-upload-manual",
        "unknown-source-1",
    ]
    
    user2_sources = [
        "patient-device-scale-ABC123",  # Same source as user1
        "patient-device-XYZ789",         # Different device
        "internal-questionnaire-v2",      # Same as user1
        "https://connectivehealth.io/api",
        "unknown-source-2",
    ]
    
    # Create mock results for each user
    def create_mock_results(sources, user_id):
        results = []
        base_time = datetime.now() - timedelta(days=30)
        for i, source in enumerate(sources):
            # Create multiple measurements per source
            for j in range(5 + i):  # Different frequencies
                results.append({
                    'user_id': user_id,
                    'timestamp': base_time + timedelta(days=i*5 + j),
                    'raw_weight': 70 + np.random.normal(0, 1),
                    'source': source,
                    'accepted': np.random.random() > 0.3
                })
        return results
    
    user1_results = create_mock_results(user1_sources, 'user1')
    user2_results = create_mock_results(user2_sources, 'user2')
    
    # Create registries for both users
    registry1 = create_source_registry(user1_results)
    registry2 = create_source_registry(user2_results)
    
    print("\nGlobal Consistency Check:")
    print("Comparing shared sources between users...")
    
    # Check that shared sources have the same visual properties
    shared_sources = set(user1_sources) & set(user2_sources)
    
    for source in shared_sources:
        style1 = registry1.get(source, {})
        style2 = registry2.get(source, {})
        
        # Check marker consistency
        marker_match = style1.get('marker') == style2.get('marker')
        color_match = style1.get('color') == style2.get('color')
        priority_match = style1.get('priority') == style2.get('priority')
        
        print(f"\n  Source: {source}")
        print(f"    User1: marker={style1.get('marker')}, color={style1.get('color')}, priority={style1.get('priority')}")
        print(f"    User2: marker={style2.get('marker')}, color={style2.get('color')}, priority={style2.get('priority')}")
        print(f"    Consistent: marker={marker_match}, color={color_match}, priority={priority_match}")
        
        assert marker_match, f"Marker mismatch for {source}"
        assert color_match, f"Color mismatch for {source}"
        assert priority_match, f"Priority mismatch for {source}"
    
    print("\n✓ All shared sources have consistent visual properties!")


def test_unknown_source_consistency():
    """Test that unknown sources get consistent (though arbitrary) assignments."""
    
    # The same unknown source should always get the same priority
    unknown_source = "completely-unknown-source-xyz123"
    patterns = get_global_source_patterns()
    
    # Check multiple times to ensure consistency
    priorities = []
    for _ in range(5):
        priority = match_source_to_pattern(unknown_source, patterns)
        priorities.append(priority)
    
    print(f"\nUnknown Source Consistency:")
    print(f"  Source: {unknown_source}")
    print(f"  Priorities across calls: {priorities}")
    
    assert len(set(priorities)) == 1, "Unknown source should get consistent priority"
    assert 100 <= priorities[0] < 200, "Unknown source priority should be in 100-199 range"
    
    print("  ✓ Unknown source gets consistent priority!")


def test_priority_ordering():
    """Test that known patterns have correct priority ordering."""
    patterns = get_global_source_patterns()
    
    print("\nPriority Ordering:")
    print("  Pattern priorities (lower = higher priority):")
    
    for pattern, priority in patterns[:10]:  # Show first 10
        print(f"    {priority:2d}: {pattern}")
    
    # Check that priorities are unique and ordered
    priorities = [p[1] for p in patterns]
    assert priorities == sorted(priorities), "Priorities should be in order"
    assert len(priorities) == len(set(priorities)), "Priorities should be unique"
    
    print("\n  ✓ Pattern priorities are correctly ordered!")


def test_visual_distinction():
    """Test that the visual properties provide good distinction."""
    
    # Create a dataset with many different sources
    sources = [
        "patient-device-scale-ABC123",
        "patient-device-scale-XYZ789",
        "patient-device-other",
        "internal-questionnaire-v1",
        "internal-questionnaire-v2",
        "https://connectivehealth.io/api",
        "https://api.iglucose.com/v1/weights",
        "patient-upload-manual",
        "patient-upload-mobile",
        "test-source-1",
        "unknown-1",
        "unknown-2",
        "unknown-3",
    ]
    
    results = []
    base_time = datetime.now()
    for i, source in enumerate(sources):
        results.append({
            'source': source,
            'timestamp': base_time + timedelta(hours=i),
            'raw_weight': 70,
            'accepted': True
        })
    
    registry = create_source_registry(results)
    
    print("\nVisual Distinction Check:")
    print("  Checking marker and color assignments...")
    
    # Check that primary sources have distinct markers
    primary_sources = [s for s, info in registry.items() if info.get('is_primary')]
    primary_markers = [registry[s]['marker'] for s in primary_sources]
    primary_colors = [registry[s]['color'] for s in primary_sources]
    
    print(f"  Primary sources: {len(primary_sources)}")
    print(f"  Unique markers: {len(set(primary_markers))}")
    print(f"  Unique colors: {len(set(primary_colors))}")
    
    # All primary sources should have unique colors
    assert len(set(primary_colors)) == len(primary_colors), "Primary sources should have unique colors"
    
    print("\n  ✓ Visual properties provide good distinction!")


if __name__ == "__main__":
    print("Testing Global Source Consistency...")
    print("=" * 60)
    
    test_source_pattern_matching()
    test_global_consistency()
    test_unknown_source_consistency()
    test_priority_ordering()
    test_visual_distinction()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("\nKey verifications:")
    print("- Sources match to patterns consistently")
    print("- Same source gets same visual properties across users")
    print("- Unknown sources get consistent (hash-based) assignments")
    print("- Pattern priorities are correctly ordered")
    print("- Visual properties provide good distinction")
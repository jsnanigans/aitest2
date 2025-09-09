#!/usr/bin/env python3
"""Test that daily cleanup keeps all valid measurements."""

from datetime import datetime, date
from src.reprocessor import WeightReprocessor

def test_keeps_multiple_valid_measurements():
    """Test that multiple valid measurements are kept, not just the best one."""
    measurements = [
        {'weight': 75.2, 'timestamp': datetime(2025, 1, 15, 8, 0), 'source': 'scale'},
        {'weight': 75.8, 'timestamp': datetime(2025, 1, 15, 18, 0), 'source': 'scale'},
        {'weight': 75.5, 'timestamp': datetime(2025, 1, 15, 12, 0), 'source': 'manual'}
    ]
    
    selected = WeightReprocessor.select_best_measurements(measurements)
    
    assert len(selected) == 3, f"Should keep all 3 valid measurements, got {len(selected)}"
    
    assert selected[0]['source'] == 'scale', "First should be scale (best source)"
    assert selected[1]['source'] == 'scale', "Second should be scale"
    assert selected[2]['source'] == 'manual', "Third should be manual (lower priority)"
    
    print(f"✓ Kept all {len(selected)} valid measurements")
    for i, m in enumerate(selected):
        print(f"  {i+1}. {m['weight']}kg from {m['source']}")

def test_removes_only_outliers():
    """Test that only extreme outliers are removed."""
    measurements = [
        {'weight': 75.2, 'timestamp': datetime(2025, 1, 15, 8, 0), 'source': 'scale'},
        {'weight': 152.4, 'timestamp': datetime(2025, 1, 15, 9, 0), 'source': 'manual'},  # Clear error (2x)
        {'weight': 75.8, 'timestamp': datetime(2025, 1, 15, 18, 0), 'source': 'scale'},
        {'weight': 76.1, 'timestamp': datetime(2025, 1, 15, 20, 0), 'source': 'manual'}
    ]
    
    selected = WeightReprocessor.select_best_measurements(measurements)
    
    assert len(selected) == 3, f"Should keep 3 valid, remove 1 outlier, got {len(selected)}"
    
    weights = [m['weight'] for m in selected]
    assert 152.4 not in weights, "Should have removed the 152.4kg outlier"
    assert all(74 < w < 77 for w in weights), "All remaining should be in reasonable range"
    
    print(f"✓ Removed outlier, kept {len(selected)} valid measurements:")
    for m in selected:
        print(f"  {m['weight']}kg from {m['source']}")

def test_with_recent_weight():
    """Test that measurements are sorted by consistency with recent weight."""
    measurements = [
        {'weight': 74.5, 'timestamp': datetime(2025, 1, 15, 8, 0), 'source': 'manual'},
        {'weight': 75.0, 'timestamp': datetime(2025, 1, 15, 12, 0), 'source': 'manual'},
        {'weight': 77.0, 'timestamp': datetime(2025, 1, 15, 18, 0), 'source': 'manual'}
    ]
    
    selected = WeightReprocessor.select_best_measurements(
        measurements, 
        recent_weight=75.2
    )
    
    assert len(selected) == 3, f"Should keep all 3 measurements, got {len(selected)}"
    
    assert selected[0]['weight'] == 75.0, "Closest to recent weight should be first"
    assert selected[1]['weight'] == 74.5, "Second closest should be second"
    assert selected[2]['weight'] == 77.0, "Furthest should be last"
    
    print(f"✓ Sorted by proximity to recent weight (75.2kg):")
    for m in selected:
        deviation = abs(m['weight'] - 75.2)
        print(f"  {m['weight']}kg (deviation: {deviation:.1f}kg)")

def test_extreme_outlier_removal():
    """Test removal of unrealistic weights."""
    measurements = [
        {'weight': 75.0, 'timestamp': datetime(2025, 1, 15, 8, 0), 'source': 'scale'},
        {'weight': 750.0, 'timestamp': datetime(2025, 1, 15, 9, 0), 'source': 'manual'},  # 10x error
        {'weight': 7.5, 'timestamp': datetime(2025, 1, 15, 10, 0), 'source': 'manual'},   # 0.1x error
        {'weight': 75.5, 'timestamp': datetime(2025, 1, 15, 18, 0), 'source': 'scale'}
    ]
    
    selected = WeightReprocessor.select_best_measurements(measurements)
    
    assert len(selected) == 2, f"Should keep 2 valid, remove 2 extreme, got {len(selected)}"
    
    weights = [m['weight'] for m in selected]
    assert all(30 <= w <= 400 for w in weights), "All should be in valid range (30-400kg)"
    
    print(f"✓ Removed extreme outliers, kept {len(selected)} valid:")
    for m in selected:
        print(f"  {m['weight']}kg")

if __name__ == "__main__":
    test_keeps_multiple_valid_measurements()
    print()
    test_removes_only_outliers()
    print()
    test_with_recent_weight()
    print()
    test_extreme_outlier_removal()
    print("\n✅ All tests passed!")
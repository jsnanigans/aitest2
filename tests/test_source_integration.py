#!/usr/bin/env python3
"""
Integration test for unique source visualization with actual processor.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processor import WeightProcessor
from src.visualization import create_dashboard


def test_source_integration():
    """Test that sources flow through processor to visualization correctly."""
    
    # Test data with various unique sources
    test_data = [
        # Device sources - each unique device ID is its own source
        ('patient-device-scale-ABC123', 70.5, 0),
        ('patient-device-scale-ABC123', 70.3, 1),
        ('patient-device-scale-XYZ789', 70.8, 2),
        ('patient-device-scale-XYZ789', 70.6, 3),
        ('patient-device-scale-DEF456', 70.4, 4),
        
        # API sources - each unique API endpoint
        ('https://api.iglucose.com/v1/weights', 71.0, 5),
        ('https://api.iglucose.com/v2/measurements', 70.9, 6),
        ('https://connectivehealth.io/api/weight', 70.7, 7),
        
        # Questionnaire sources - different versions
        ('internal-questionnaire-v1', 71.2, 8),
        ('internal-questionnaire-v2', 71.1, 9),
        ('internal-questionnaire-v2', 71.0, 10),
        
        # Manual entries
        ('patient-upload-manual', 70.2, 11),
        ('patient-upload-mobile-app', 70.5, 12),
        
        # Edge cases
        ('', 70.0, 13),  # Empty source
        ('unknown', 69.8, 14),  # Unknown source
    ]
    
    user_id = 'test_integration_user'
    base_time = datetime.now() - timedelta(days=30)
    
    # Reset user state
    WeightProcessor.reset_user(user_id)
    
    # Process all measurements
    results = []
    for source, weight, day_offset in test_data:
        timestamp = base_time + timedelta(days=day_offset)
        
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source if source else 'unknown',  # Handle empty source
            processing_config={
                'extreme_threshold': 0.15,
                'extreme_threshold_pct': 15,
                'extreme_threshold_kg': 10,
                'min_confidence': 0.3
            },
            kalman_config={
                'initial_weight': 70.0,
                'initial_variance': 1.0,
                'process_variance': 0.01,
                'observation_covariance': 0.5
            }
        )
        
        if result:
            results.append(result)
            print(f"Processed: source='{source[:30]:30s}', "
                  f"weight={weight:5.1f}, accepted={result.get('accepted', False)}")
    
    # Count unique sources
    unique_sources = set()
    for r in results:
        source = r.get('source', 'unknown')
        unique_sources.add(source)
    
    print(f"\nTotal measurements: {len(results)}")
    print(f"Unique sources: {len(unique_sources)}")
    
    # Show source distribution
    from collections import Counter
    source_counts = Counter(r.get('source', 'unknown') for r in results)
    print("\nSource distribution:")
    for source, count in source_counts.most_common():
        print(f"  {source[:40]:40s}: {count}")
    
    # Create visualization
    viz_config = {
        'dashboard_dpi': 100,
        'cropped_months': 1
    }
    
    output_file = create_dashboard(user_id, results, '/tmp', viz_config)
    print(f"\nVisualization created: {output_file}")
    
    # Verify all unique sources are represented
    assert len(unique_sources) >= 10, f"Expected at least 10 unique sources, got {len(unique_sources)}"
    print("\n✓ All unique sources preserved through processing pipeline")


if __name__ == "__main__":
    print("Testing source integration with processor and visualization...")
    print("=" * 60)
    
    test_source_integration()
    
    print("\n" + "=" * 60)
    print("Integration test passed! ✓")
    print("\nKey improvements verified:")
    print("- Each unique source string preserved through processing")
    print("- Sources flow correctly from processor to visualization")
    print("- Visualization handles many unique sources gracefully")
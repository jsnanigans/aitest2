#!/usr/bin/env python3
"""Test that all source types are properly displayed in visualizations."""

from src.visualization.enhanced_dashboard import EnhancedDashboard

def test_source_display():
    """Test that all source types from the CSV are properly mapped."""
    
    # Simulate data with various source types
    test_data = {
        'user_id': 'test_user',
        'time_series': [
            {'date': '2025-01-01', 'weight': 70, 'source': 'patient-device', 'confidence': 0.9},
            {'date': '2025-01-02', 'weight': 71, 'source': 'care-team-upload', 'confidence': 0.95},
            {'date': '2025-01-03', 'weight': 70.5, 'source': 'https://connectivehealth.io', 'confidence': 0.8},
            {'date': '2025-01-04', 'weight': 71.2, 'source': 'https://api.iglucose.com', 'confidence': 0.85},
            {'date': '2025-01-05', 'weight': 70.8, 'source': 'internal-questionnaire', 'confidence': 0.9},
            {'date': '2025-01-06', 'weight': 71.5, 'source': 'patient-upload', 'confidence': 0.7},
            {'date': '2025-01-07', 'weight': 71.0, 'source': 'unknown', 'confidence': 0.5},
            {'date': '2025-01-08', 'weight': 70.9, 'source': None, 'confidence': 0.5},  # Should map to 'unknown'
        ],
        'baseline': {'weight': 70.5},
        'stats': {}
    }
    
    dashboard = EnhancedDashboard('test_user', test_data)
    
    print("Source Style Mappings:")
    print("-" * 60)
    
    # Test each source type
    for ts in test_data['time_series']:
        source = ts.get('source', 'unknown')
        if source is None:
            source = 'unknown'
        
        style = dashboard.SOURCE_STYLES.get(source, dashboard.SOURCE_STYLES['unknown'])
        print(f"Source: '{source:30s}' -> Label: '{style['label']:20s}' (marker={style['marker']}, color={style['color']})")
    
    print("\nAll mapped source types:")
    for source, style in dashboard.SOURCE_STYLES.items():
        print(f"  {source:30s}: {style['label']}")

if __name__ == '__main__':
    test_source_display()
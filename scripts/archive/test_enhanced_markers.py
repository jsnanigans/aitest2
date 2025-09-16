#!/usr/bin/env python3
"""
Test script for enhanced visualization with reset markers and source icons.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from visualization import create_weight_timeline

# Create test data with various sources and a reset event
def create_test_data():
    results = []
    base_weight = 75.0
    current_weight = base_weight
    start_date = datetime(2024, 1, 1)
    
    sources = [
        'care-team-upload',
        'patient-upload', 
        'patient-device',
        'internal-questionnaire',
        'https://connectivehealth.io'
    ]
    
    # First period - normal measurements
    for i in range(30):
        timestamp = start_date + timedelta(days=i)
        source = random.choice(sources)
        noise = random.gauss(0, 0.5)
        weight = current_weight + noise
        
        # Some accepted, some rejected
        if random.random() > 0.2:
            results.append({
                'timestamp': timestamp,
                'raw_weight': weight,
                'filtered_weight': weight - noise * 0.3,
                'source': source,
                'accepted': True,
                'quality_score': random.uniform(0.6, 1.0),
                'confidence': random.uniform(0.7, 0.99),
                'innovation': noise
            })
        else:
            rejection_reasons = [
                'Physiologically impossible change',
                'Extreme deviation from expected',
                'Suspicious weight change',
                'Minor variance detected'
            ]
            results.append({
                'timestamp': timestamp,
                'raw_weight': weight + random.uniform(5, 20),
                'source': source,
                'accepted': False,
                'reason': random.choice(rejection_reasons),
                'quality_score': random.uniform(0.1, 0.5)
            })
        
        current_weight += random.gauss(0, 0.1)
    
    # Gap period - no measurements for 35 days
    
    # Resume after gap with reset
    gap_start = start_date + timedelta(days=30)
    resume_date = gap_start + timedelta(days=35)
    
    # First measurement after gap triggers reset
    results.append({
        'timestamp': resume_date,
        'raw_weight': current_weight + 2.0,
        'filtered_weight': current_weight + 2.0,
        'source': 'patient-upload',
        'accepted': True,
        'quality_score': 0.8,
        'confidence': 0.5,
        'innovation': 0,
        'was_reset': True,
        'reset_reason': 'Gap of 35.0 days',
        'gap_days': 35.0
    })
    
    # Continue with more measurements
    for i in range(1, 20):
        timestamp = resume_date + timedelta(days=i)
        source = random.choice(sources)
        noise = random.gauss(0, 0.5)
        weight = current_weight + 2.0 + noise
        
        if random.random() > 0.15:
            results.append({
                'timestamp': timestamp,
                'raw_weight': weight,
                'filtered_weight': weight - noise * 0.3,
                'source': source,
                'accepted': True,
                'quality_score': random.uniform(0.6, 1.0),
                'confidence': random.uniform(0.8, 0.99),
                'innovation': noise
            })
        else:
            results.append({
                'timestamp': timestamp,
                'raw_weight': weight + random.uniform(3, 15),
                'source': source,
                'accepted': False,
                'reason': random.choice([
                    'Extreme deviation from expected',
                    'Suspicious weight change'
                ]),
                'quality_score': random.uniform(0.2, 0.5)
            })
    
    return results

# Test configuration
config = {
    'visualization': {
        'markers': {
            'show_source_icons': True,
            'show_reset_markers': True,
            'reset_marker_color': '#FF6600'
        },
        'rejection': {
            'show_severity_colors': True,
            'group_by_severity': True
        }
    }
}

# Generate test data
print("Generating test data...")
results = create_test_data()

# Create visualization
print(f"Creating visualization with {len(results)} data points...")
# Test both visualizations
print("\n=== Testing Basic Visualization ===")
output_file_basic = create_weight_timeline(
    results, 
    user_id="TEST_USER_BASIC",
    output_dir="test_output",
    use_enhanced=False,  # Test basic visualization
    config=config
)
print(f"Basic visualization saved to: {output_file_basic}")

print("\n=== Testing Enhanced Visualization ===")
output_file_enhanced = create_weight_timeline(
    results, 
    user_id="TEST_USER_ENHANCED",
    output_dir="test_output",
    use_enhanced=True,  # Test enhanced visualization
    config=config
)
print(f"Enhanced visualization saved to: {output_file_enhanced}")
print("\nTest Summary:")
print(f"- Total measurements: {len(results)}")
print(f"- Accepted: {sum(1 for r in results if r.get('accepted', False))}")
print(f"- Rejected: {sum(1 for r in results if not r.get('accepted', False))}")
print(f"- Reset events: {sum(1 for r in results if r.get('was_reset', False))}")
print(f"- Unique sources: {len(set(r['source'] for r in results))}")
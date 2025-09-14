#!/usr/bin/env python3
"""Simple test of diagnostic dashboard without main.py"""

from src.viz_diagnostic import DiagnosticDashboard, create_diagnostic_report
import pandas as pd

# Create test data with various scenarios
results = [
    # Accepted measurements
    {'timestamp': '2024-01-01', 'accepted': True, 'raw_weight': 70.0, 'filtered_weight': 70.0,
     'source': 'scale', 'quality_score': 0.9, 'innovation': 0.1, 'normalized_innovation': 0.5,
     'confidence': 0.95, 'quality_components': {'safety': 0.95, 'plausibility': 0.9, 
     'consistency': 0.85, 'reliability': 0.9}},
    
    {'timestamp': '2024-01-02', 'accepted': True, 'raw_weight': 70.2, 'filtered_weight': 70.1,
     'source': 'scale', 'quality_score': 0.85, 'innovation': 0.2, 'normalized_innovation': 1.0,
     'confidence': 0.85},
     
    # Rejected measurement
    {'timestamp': '2024-01-03', 'accepted': False, 'raw_weight': 75.0, 'source': 'questionnaire',
     'reason': 'Quality score 0.45 below threshold', 'quality_score': 0.45, 'stage': 'quality_scoring'},
     
    # Gap reset
    {'timestamp': '2024-02-01', 'accepted': True, 'raw_weight': 71.0, 'filtered_weight': 71.0,
     'source': 'scale', 'quality_score': 0.8, 'was_reset': True, 'gap_days': 28,
     'innovation': 0.0, 'normalized_innovation': 0.0, 'confidence': 1.0},
]

# Create dashboard
dashboard = DiagnosticDashboard()
fig = dashboard.create_diagnostic_dashboard(results, 'test_user')

# Save to HTML
output_file = 'test_diagnostic_simple.html'
fig.write_html(output_file)
print(f"Dashboard saved to {output_file}")

# Create report
report = create_diagnostic_report(results, 'test_user')
with open('test_diagnostic_report.txt', 'w') as f:
    f.write(report)
print("Report saved to test_diagnostic_report.txt")

#!/usr/bin/env python3
"""Test script for enhanced Kalman dashboard."""

import sys
from pathlib import Path

# Test direct import of enhanced dashboard
try:
    from src.viz_plotly_enhanced import create_enhanced_dashboard
    from src.viz_kalman import extract_kalman_data, prepare_kalman_visualization_data
    print("✓ Enhanced dashboard modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import enhanced modules: {e}")
    sys.exit(1)

# Load test data
import json

output_dir = Path("output")
results_files = list(output_dir.glob("results_*.json"))

if not results_files:
    print("No results files found. Run main.py first.")
    sys.exit(1)

latest_results = sorted(results_files)[-1]
print(f"Using results from: {latest_results}")

with open(latest_results, 'r') as f:
    data = json.load(f)

users = data.get("users", {})
if not users:
    print("No user data found in results")
    sys.exit(1)

# Get first user's data
user_id = list(users.keys())[0]
user_results = users[user_id]

print(f"\nProcessing user: {user_id}")
print(f"Total measurements: {len(user_results)}")

# Extract Kalman data
kalman_df = extract_kalman_data(user_results)
print(f"Kalman data extracted: {len(kalman_df)} rows")

# Check what Kalman data is available
kalman_cols = [col for col in kalman_df.columns if 'kalman' in col.lower() or 
               col in ['filtered_weight', 'trend', 'innovation', 'confidence']]
print(f"Kalman-related columns: {kalman_cols}")

# Check if we have the key fields
has_filtered = 'filtered_weight' in kalman_df.columns
has_innovation = 'innovation' in kalman_df.columns
has_confidence = 'confidence' in kalman_df.columns

print(f"\nKalman data availability:")
print(f"  Filtered weight: {has_filtered}")
print(f"  Innovation: {has_innovation}")
print(f"  Confidence: {has_confidence}")

if has_filtered:
    filtered_vals = kalman_df['filtered_weight'].dropna()
    if len(filtered_vals) > 0:
        print(f"  Filtered weight range: {filtered_vals.min():.1f} - {filtered_vals.max():.1f}")

# Prepare visualization data
import tomllib
with open("config.toml", "rb") as f:
    config = tomllib.load(f)

kalman_viz_data = prepare_kalman_visualization_data(kalman_df, config)
print(f"\nVisualization data prepared:")
print(f"  Metrics: {len(kalman_viz_data.get('metrics', {}))}")
print(f"  Thresholds: {len(kalman_viz_data.get('thresholds', {}))}")

# Try to create enhanced dashboard
try:
    output_path = create_enhanced_dashboard(user_results, user_id, "output/test_kalman", config)
    if output_path:
        print(f"\n✓ Enhanced dashboard created: {output_path}")
        file_size = Path(output_path).stat().st_size / 1024
        print(f"  File size: {file_size:.1f} KB")
    else:
        print("\n✗ Dashboard creation returned None")
except Exception as e:
    print(f"\n✗ Error creating dashboard: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Enhanced Dashboard Features:")
print("  • Main weight graph with Kalman filtered line")
print("  • Uncertainty bands (±1σ and ±2σ)")
print("  • Innovation and normalized innovation subplots")
print("  • Confidence evolution tracking")
print("  • Daily change distribution")
print("  • Filter performance metrics")
print("  • Kalman statistics panel")
print("="*60)
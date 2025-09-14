#!/usr/bin/env python3
"""
Test script for enhanced weight timeline visualization.
"""

import json
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.viz_enhanced import create_enhanced_weight_timeline


def test_enhanced_visualization():
    """Test the enhanced visualization with sample data."""
    
    # Load test data
    test_files = [
        "test_enhanced_data/results_test_no_date.json",
        "test_index_output/results_test_no_date.json",
        "test_pipeline_output/results_test_no_date.json",
        "test_layout_fix/results_test_no_date.json"
    ]
    
    results_file = None
    for test_file in test_files:
        if Path(test_file).exists():
            results_file = test_file
            break
    
    if not results_file:
        print("No test data found. Running main.py first to generate data...")
        import subprocess
        subprocess.run([sys.executable, "main.py", "data/test_sample.csv"])
        results_file = "output/results_test_no_date.json"
    
    print(f"Loading results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    # Get users
    users = results_data.get('users', {})
    
    if not users:
        print("No users found in results")
        return
    
    # Create enhanced visualization for each user
    output_dir = "test_enhanced_output"
    Path(output_dir).mkdir(exist_ok=True)
    
    for user_id, measurements in users.items():
        if not measurements:
            continue
        
        print(f"\nProcessing user: {user_id}")
        print(f"  Measurements: {len(measurements)}")
        
        # Check if we have enhanced data - check all measurements
        has_quality = False
        has_kalman = False
        for m in measurements:
            if 'quality_score' in m or 'quality_components' in m:
                has_quality = True
            if 'kalman_confidence_upper' in m or 'filtered_weight' in m:
                has_kalman = True
            if has_quality and has_kalman:
                break
        
        print(f"  Has quality data: {has_quality}")
        print(f"  Has Kalman data: {has_kalman}")
        
        try:
            html_path = create_enhanced_weight_timeline(
                measurements,
                user_id,
                output_dir=output_dir
            )
            print(f"  ✓ Created: {html_path}")
            
            # Print some statistics
            accepted = [m for m in measurements if m.get('accepted', False)]
            rejected = [m for m in measurements if not m.get('accepted', False)]
            
            print(f"  Stats: {len(accepted)} accepted, {len(rejected)} rejected")
            
            if accepted and has_quality:
                quality_scores = [m.get('quality_score', 0) for m in accepted if 'quality_score' in m]
                if quality_scores:
                    avg_quality = sum(quality_scores) / len(quality_scores)
                    print(f"  Avg quality score: {avg_quality:.2f}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ Enhanced visualizations created in: {output_dir}/")
    
    # Create an index file
    create_index_html(output_dir, users.keys())


def create_index_html(output_dir: str, user_ids):
    """Create a simple index HTML file."""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Weight Timeline Visualizations</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #2196F3;
            padding-bottom: 10px;
        }
        .user-list {
            list-style: none;
            padding: 0;
        }
        .user-item {
            background: white;
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .user-item:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .user-link {
            text-decoration: none;
            color: #2196F3;
            font-weight: 500;
            font-size: 18px;
        }
        .description {
            color: #666;
            margin-top: 5px;
            font-size: 14px;
        }
        .features {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .features h2 {
            color: #1976D2;
            margin-top: 0;
        }
        .features ul {
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <h1>Enhanced Weight Timeline Visualizations</h1>
    
    <div class="features">
        <h2>New Features</h2>
        <ul>
            <li>📊 Raw data points with quality score coloring</li>
            <li>📈 Kalman filtered line with confidence bands (±2σ)</li>
            <li>🎯 Quality score component breakdown (safety, plausibility, consistency, reliability)</li>
            <li>📉 Innovation/residuals visualization with σ reference lines</li>
            <li>🔄 Interactive layer toggles via dropdown menu</li>
            <li>💡 Rich hover information with all metrics</li>
        </ul>
    </div>
    
    <h2>User Dashboards</h2>
    <ul class="user-list">
"""
    
    for user_id in sorted(user_ids):
        html_content += f"""
        <li class="user-item">
            <a href="{user_id}_enhanced_timeline.html" class="user-link">User: {user_id}</a>
            <div class="description">Enhanced interactive weight timeline with quality insights and Kalman filter details</div>
        </li>
"""
    
    html_content += """
    </ul>
</body>
</html>
"""
    
    index_path = Path(output_dir) / "index.html"
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created index: {index_path}")


if __name__ == "__main__":
    test_enhanced_visualization()
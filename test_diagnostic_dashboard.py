#!/usr/bin/env python3
"""
Test script for the enhanced diagnostic dashboard.
Demonstrates all the new visualization features.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import webbrowser

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.viz_diagnostic import DiagnosticDashboard, create_diagnostic_report
from src.processor import process_measurement
from src.database import ProcessorStateDB
import pandas as pd

def load_test_data(csv_file: str):
    """Load and process test data."""
    df = pd.read_csv(csv_file)
    
    # Initialize database
    db = ProcessorStateDB()
    
    # Load config
    with open("config.toml", "r") as f:
        import toml
        config = toml.load(f)
    
    # Process measurements
    results = []
    for _, row in df.iterrows():
        # Handle different column names
        timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'effectiveDateTime'
        source_col = 'source' if 'source' in df.columns else 'source_type'
        
        result = process_measurement(
            user_id=row['user_id'],
            weight=row['weight'],
            timestamp=pd.to_datetime(row[timestamp_col]),
            source=row.get(source_col, 'unknown'),
            config=config,
            unit=row.get('unit', 'kg'),
            db=db
        )
        results.append(result)
    
    return results

def main():
    """Main test function."""
    
    # Check for test data file
    test_files = [
        "data/test_sample.csv",
        "data/test_multi_user.csv",
        "data/test_index_demo.csv"
    ]
    
    test_file = None
    for file in test_files:
        if Path(file).exists():
            test_file = file
            break
    
    if not test_file:
        print("No test data file found. Please ensure one of these files exists:")
        for file in test_files:
            print(f"  - {file}")
        sys.exit(1)
    
    print(f"Loading test data from {test_file}...")
    
    # Load and process data
    try:
        results = load_test_data(test_file)
        print(f"Processed {len(results)} measurements")
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Group by user
    user_results = {}
    for r in results:
        if r:
            # Get user_id from the result or from the CSV
            user_id = 'test_user'  # Default user ID
            if user_id not in user_results:
                user_results[user_id] = []
            user_results[user_id].append(r)
    
    print(f"Found {len(user_results)} users")
    
    if not user_results:
        print("No valid results to visualize")
        sys.exit(1)
    
    # Create diagnostic dashboard for first user
    first_user = list(user_results.keys())[0]
    user_data = user_results[first_user]
    
    print(f"\nCreating diagnostic dashboard for user {first_user[:8]}...")
    print(f"  - Total measurements: {len(user_data)}")
    print(f"  - Accepted: {sum(1 for r in user_data if r.get('accepted', False))}")
    print(f"  - Rejected: {sum(1 for r in user_data if not r.get('accepted', False))}")
    
    # Print sample of quality scores if available
    quality_scores = [r.get('quality_score') for r in user_data if r.get('quality_score') is not None]
    if quality_scores:
        print(f"  - Quality scores: min={min(quality_scores):.2f}, max={max(quality_scores):.2f}, avg={sum(quality_scores)/len(quality_scores):.2f}")
    
    # Create dashboard
    dashboard = DiagnosticDashboard()
    fig = dashboard.create_diagnostic_dashboard(user_data, first_user)
    
    # Save to HTML
    output_file = "diagnostic_dashboard.html"
    fig.write_html(output_file)
    print(f"\nDashboard saved to {output_file}")
    
    # Create diagnostic report
    report = create_diagnostic_report(user_data, first_user)
    report_file = "diagnostic_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Diagnostic report saved to {report_file}")
    
    # Print summary of features
    print("\n" + "="*60)
    print("ENHANCED DIAGNOSTIC DASHBOARD FEATURES")
    print("="*60)
    print("""
The new diagnostic dashboard provides:

1. MAIN TIMELINE
   - Quality score color coding on accepted points
   - Detailed hover information with ALL quality components
   - Clear rejection markers with categories
   - Gap reset indicators with annotations
   - Uncertainty bands around Kalman filtered line

2. QUALITY SCORE BREAKDOWN
   - Stacked area chart showing all 4 components
   - Safety, Plausibility, Consistency, Reliability
   - Clear threshold line at 0.6

3. REJECTION ANALYSIS
   - Bar chart of rejection categories
   - Sunburst chart for hierarchical view
   - Shows rejection stage and reason

4. KALMAN DIAGNOSTICS
   - Innovation (residuals) with color coding
   - Standard deviation bands (±1σ, ±2σ)
   - Normalized innovation distribution
   - Confidence evolution over time

5. COMPONENT TIMELINE
   - Individual component scores over time
   - Easy identification of weak components

6. GAP RESET TABLE
   - Clear list of all reset events
   - Shows gap duration and weight at reset

7. SOURCE IMPACT ANALYSIS
   - Acceptance rate by data source
   - Identifies reliable vs unreliable sources

8. COMPREHENSIVE HOVER TOOLTIPS
   - Every point shows complete information
   - Quality scores, components, innovation
   - Rejection reasons and stages

9. EXPORTABLE REPORT
   - Text-based diagnostic summary
   - All key statistics and patterns
   - Ready for documentation
    """)
    
    # Try to open in browser
    try:
        webbrowser.open(f"file://{Path(output_file).absolute()}")
        print(f"\nOpening dashboard in browser...")
    except:
        print(f"\nPlease open {output_file} in your browser to view the dashboard")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()

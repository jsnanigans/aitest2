#!/usr/bin/env python
"""
Test script to demonstrate employer filtering with full analysis.
"""

import subprocess
import sys
from pathlib import Path

def run_employer_analysis():
    """Run analysis for largest employer to demonstrate functionality."""

    print("=" * 70)
    print("EMPLOYER FILTERING ANALYSIS TEST")
    print("=" * 70)
    print()

    # Test with largest employer
    employer_id = "0a427a45-cebe-4cec-977b-f65a9b6534bc"

    print(f"Running analysis for employer: {employer_id}")
    print("This will:")
    print("1. Load ALL users from this employer (no max-users limit)")
    print("2. Analyze ALL users' data")
    print("3. Generate visualizations for top 10 most impacted users")
    print()
    print("-" * 70)

    cmd = [
        "uv", "run", "python", "scripts/run_filtering_analysis.py",
        "--filter-employer", employer_id,
        "--output-dir", "reports/test_employer_analysis"
    ]

    print(f"Command: {' '.join(cmd)}")
    print("-" * 70)
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Extract key information from output
    lines = result.stderr.split('\n') if result.stderr else []

    for line in lines:
        if any(keyword in line for keyword in [
            "Loading raw data",
            "Loading filtered data",
            "Found",
            "Will analyze",
            "Generating visualizations for",
            "User.*impact",
            "Analysis Complete",
            "Report:",
            "Metrics:"
        ]):
            print(line)

    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 70)

        # Check generated files
        report_files = list(Path("reports").glob("filtering_analysis_*.md"))
        if report_files:
            latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
            print(f"\n📊 Latest Report: {latest_report}")

            # Read summary from report
            with open(latest_report, 'r') as f:
                lines = f.readlines()
                in_summary = False
                for line in lines[:20]:
                    if "Executive Summary" in line:
                        in_summary = True
                    elif in_summary and line.strip() and not line.startswith("#"):
                        print(line.strip())
                    elif in_summary and line.startswith("##"):
                        break

        viz_dir = Path("reports/test_employer_analysis")
        if viz_dir.exists():
            viz_files = list(viz_dir.glob("*.png"))
            print(f"\n📈 Visualizations Generated: {len(viz_files)} files")
            for viz in viz_files[:5]:
                print(f"   - {viz.name}")

    else:
        print(f"\n❌ Analysis failed with return code: {result.returncode}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(run_employer_analysis())
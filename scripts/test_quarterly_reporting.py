#!/usr/bin/env python
"""
Test script for quarterly reporting functionality.
Tests that the analysis correctly answers: "What is the average weight loss for users in the program for 90+ days?"
"""

import subprocess
import sys
from pathlib import Path
import json


def run_quarterly_test():
    """Run the quarterly reporting analysis test."""

    print("=" * 80)
    print("QUARTERLY REPORTING TEST")
    print("=" * 80)
    print("\nTesting the quarterly reporting functionality that answers:")
    print("\"What is the average weight loss for users in the program for 90+ days?\"")
    print("-" * 80)

    # Run analysis with quarterly reporting
    cmd = [
        "uv", "run", "python", "scripts/run_filtering_analysis.py",
        "--max-users", "100",  # Test with subset
        "--output-dir", "reports/quarterly_test"
    ]

    print(f"\nCommand: {' '.join(cmd)}")
    print("-" * 80)

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check for key outputs
    if result.stderr:
        for line in result.stderr.split('\n'):
            if any(keyword in line for keyword in [
                "quarterly reporting",
                "90+ days",
                "Weight loss",
                "Success rate",
                "cohort progression",
                "Generated"
            ]):
                print(line.replace("INFO - ", ""))

    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("✅ QUARTERLY ANALYSIS COMPLETED")
        print("=" * 80)

        # Check for generated report
        reports = list(Path("reports").glob("filtering_analysis_*.md"))
        if reports:
            latest_report = max(reports, key=lambda p: p.stat().st_mtime)
            print(f"\n📊 Report Generated: {latest_report}")

            # Extract quarterly section from report
            with open(latest_report, 'r') as f:
                content = f.read()
                if "QUARTERLY REPORTING ANALYSIS" in content:
                    print("\n✅ QUARTERLY SECTION FOUND IN REPORT")

                    # Extract the key answer
                    lines = content.split('\n')
                    in_quarterly = False
                    for i, line in enumerate(lines):
                        if "QUARTERLY REPORTING ANALYSIS" in line:
                            in_quarterly = True
                        elif in_quarterly and "What is the average weight loss" in line:
                            # Print the next 10 lines which should contain the answer
                            print("\n" + "=" * 80)
                            print("KEY BUSINESS QUESTION ANSWERED:")
                            print("=" * 80)
                            for j in range(i, min(i + 15, len(lines))):
                                print(lines[j])
                                if "Improvement" in lines[j]:
                                    break
                            break

        # Check for quarterly visualizations
        quarterly_viz_dir = Path("reports/quarterly_test/quarterly")
        if quarterly_viz_dir.exists():
            viz_files = list(quarterly_viz_dir.glob("*.png"))
            print(f"\n📈 Quarterly Visualizations Generated: {len(viz_files)} files")
            for viz in viz_files:
                print(f"   ✓ {viz.name}")

        # Check metrics JSON
        metrics_files = list(Path("reports").glob("filtering_metrics_*.json"))
        if metrics_files:
            latest_metrics = max(metrics_files, key=lambda p: p.stat().st_mtime)
            with open(latest_metrics, 'r') as f:
                metrics = json.load(f)
                if 'quarterly' in metrics:
                    print("\n✅ QUARTERLY METRICS IN JSON")

                    if 'raw_metrics' in metrics['quarterly']:
                        rm = metrics['quarterly']['raw_metrics']
                        fm = metrics['quarterly']['filtered_metrics']

                        print("\n" + "=" * 80)
                        print("SUMMARY OF QUARTERLY METRICS:")
                        print("=" * 80)
                        print(f"Eligible Users (90+ days): {rm.get('eligible_users', 'N/A')}")
                        print(f"Users with Valid Data (Raw): {rm.get('users_with_valid_endpoint', 'N/A')}")
                        print(f"Users with Valid Data (Filtered): {fm.get('users_with_valid_endpoint', 'N/A')}")
                        print(f"\nAverage Weight Loss:")
                        print(f"  Raw Data: {rm.get('mean_weight_loss_pct', 'N/A'):.2f}%")
                        print(f"  Filtered Data: {fm.get('mean_weight_loss_pct', 'N/A'):.2f}%")
                        print(f"  Improvement: {fm.get('mean_weight_loss_pct', 0) - rm.get('mean_weight_loss_pct', 0):+.2f}%")

    else:
        print(f"\n❌ Analysis failed with return code: {result.returncode}")
        if result.stderr:
            print("\nError output:")
            print(result.stderr[-1000:])  # Last 1000 chars of error
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_quarterly_test())
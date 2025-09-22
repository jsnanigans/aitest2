#!/usr/bin/env python
"""
Analyze filtering effectiveness for a specific employer.
This script provides a complete analysis workflow for employer-specific data.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime


def list_available_employers(employer_file: str, top_n: int = 10):
    """List available employers with user counts."""
    try:
        df = pd.read_csv(employer_file)
        employer_counts = df['employer_id'].value_counts()

        print("\n" + "=" * 70)
        print("AVAILABLE EMPLOYERS")
        print("=" * 70)
        print(f"\nTop {min(top_n, len(employer_counts))} employers by user count:\n")
        print(f"{'#':<3} {'Employer ID':<40} {'Users':<10}")
        print("-" * 70)

        for i, (emp_id, count) in enumerate(employer_counts.head(top_n).items(), 1):
            print(f"{i:<3} {str(emp_id)[:40]:<40} {count:<10}")

        print("\n" + "=" * 70)
        return employer_counts

    except Exception as e:
        print(f"Error loading employer data: {e}")
        return None


def run_employer_analysis(employer_id: str, output_dir: str = None, verbose: bool = False):
    """Run filtering analysis for a specific employer."""

    print(f"\nAnalyzing employer: {employer_id}")
    print("-" * 70)

    cmd = [
        "uv", "run", "python", "scripts/run_filtering_analysis.py",
        "--filter-employer", employer_id
    ]

    if output_dir:
        cmd.extend(["--output-dir", output_dir])

    if verbose:
        cmd.append("--verbose")

    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Extract key information
    if result.stderr:
        for line in result.stderr.split('\n'):
            if any(keyword in line for keyword in [
                "Found", "Will analyze", "Generating visualizations for",
                "User.*impact", "Analysis Complete", "Report:", "Metrics:"
            ]):
                print(line.replace("INFO - ", ""))

    return result.returncode == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze filtering effectiveness for specific employers"
    )
    parser.add_argument(
        "employer_id",
        nargs='?',
        help="Employer ID to analyze (shows list if not provided)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available employers"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for reports and visualizations"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    employer_file = "data/2025-09-17-user-employers.csv"

    # List employers if requested or no employer specified
    if args.list or not args.employer_id:
        employer_counts = list_available_employers(employer_file)

        if not args.employer_id:
            print("\nUsage:")
            print("  python scripts/analyze_employer.py <employer_id>")
            print("\nExample:")
            if employer_counts is not None and len(employer_counts) > 0:
                top_employer = employer_counts.index[0]
                print(f"  python scripts/analyze_employer.py {top_employer}")

        return 0

    # Generate default output directory with timestamp
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        employer_short = args.employer_id[:8]
        args.output_dir = f"reports/employer_{employer_short}_{timestamp}"

    # Run analysis
    success = run_employer_analysis(
        args.employer_id,
        args.output_dir,
        args.verbose
    )

    if success:
        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nOutput directory: {args.output_dir}")

        # Check for generated files
        output_path = Path(args.output_dir)
        if output_path.exists():
            viz_files = list(output_path.glob("*.png"))
            print(f"Visualizations generated: {len(viz_files)} files")

            if viz_files:
                print("\nGenerated visualizations:")
                for viz in viz_files:
                    print(f"  - {viz.name}")

        # Find latest report
        reports = list(Path("reports").glob("filtering_analysis_*.md"))
        if reports:
            latest_report = max(reports, key=lambda p: p.stat().st_mtime)
            print(f"\nReport: {latest_report}")

        print("\n" + "=" * 70)

    else:
        print("\n❌ Analysis failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
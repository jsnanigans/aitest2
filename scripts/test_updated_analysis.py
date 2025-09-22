#!/usr/bin/env python
"""
Test the updated filtering analysis with default file values.
"""

import subprocess
import sys
from pathlib import Path

def run_test():
    """Run the analysis with default files and limited users for testing."""

    print("Testing updated filtering analysis with default files...")
    print("=" * 60)

    # Run with defaults, limiting to just 5 users for quick test
    cmd = [
        "uv", "run", "python", "scripts/run_filtering_analysis.py",
        "--max-users", "5",
        "--verbose"
    ]

    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print("STDOUT:")
        print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    if result.returncode == 0:
        print("\n✅ Test completed successfully!")

        # Check for generated files
        reports = list(Path("reports").glob("filtering_analysis_*.md"))
        metrics = list(Path("reports").glob("filtering_metrics_*.json"))

        if reports:
            print(f"✅ Report generated: {reports[-1]}")
        if metrics:
            print(f"✅ Metrics generated: {metrics[-1]}")

        viz_dir = Path("reports/visualizations")
        if viz_dir.exists():
            viz_files = list(viz_dir.glob("*.png"))
            print(f"✅ Visualizations generated: {len(viz_files)} files")

    else:
        print(f"\n❌ Test failed with return code: {result.returncode}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(run_test())
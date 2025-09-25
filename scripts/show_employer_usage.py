#!/usr/bin/env python
"""
Show how to use the employer filtering feature.
"""

import pandas as pd
import sys


def show_usage():
    """Show employer filtering usage examples."""

    print("=" * 70)
    print("EMPLOYER FILTERING USAGE")
    print("=" * 70)
    print()

    # Load employer data
    employer_file = "data/2025-09-17-user-employers.csv"
    try:
        df = pd.read_csv(employer_file)
        employer_counts = df["employer_id"].value_counts()

        print("To run analysis for a specific employer:")
        print("-" * 70)
        print()

        # Show examples with top employers
        top_employers = employer_counts.head(3)

        for i, (emp_id, count) in enumerate(top_employers.items(), 1):
            print(f"Example {i} - Analyze employer with {count} users:")
            print(f"  uv run python scripts/run_filtering_analysis.py \\")
            print(f'    --filter-employer "{emp_id}"')
            print()

        print("Additional options:")
        print("-" * 70)
        print()
        print("Limit number of users analyzed:")
        print("  uv run python scripts/run_filtering_analysis.py \\")
        print(f'    --filter-employer "{top_employers.index[0]}" \\')
        print("    --max-users 100")
        print()

        print("Use custom output directory:")
        print("  uv run python scripts/run_filtering_analysis.py \\")
        print(f'    --filter-employer "{top_employers.index[0]}" \\')
        print("    --output-dir reports/employer_analysis")
        print()

        print("Available Employers:")
        print("-" * 70)
        print(f"{'Employer ID':<40} | {'Users':<10}")
        print("-" * 70)

        for emp_id, count in employer_counts.items():
            print(f"{str(emp_id)[:40]:<40} | {count:<10}")

    except FileNotFoundError:
        print(f"Employer file not found: {employer_file}")
        print("\nPlease ensure the employer data file exists.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    show_usage()

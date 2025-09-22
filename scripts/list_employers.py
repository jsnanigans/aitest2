#!/usr/bin/env python
"""
List available employers in the dataset.
"""

import pandas as pd
import sys

def list_employers():
    """List unique employer IDs and their user counts."""

    # Load employer data
    employer_file = "data/2025-09-17-user-employers.csv"
    try:
        df = pd.read_csv(employer_file)

        # Count users per employer
        employer_counts = df['employer_id'].value_counts()

        print(f"Found {len(employer_counts)} unique employers")
        print(f"Total users with employer data: {len(df)}\n")
        print("Top 20 employers by user count:")
        print("-" * 60)
        print(f"{'Employer ID':<40} | Users")
        print("-" * 60)

        for employer_id, count in employer_counts.head(20).items():
            print(f"{str(employer_id)[:40]:<40} | {count}")

        return employer_counts

    except FileNotFoundError:
        print(f"Employer file not found: {employer_file}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    list_employers()
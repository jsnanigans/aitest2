#!/usr/bin/env python3
"""
Script to add unique measurement_id column to CSV files.
Updates the file in place, adding measurement_id as the first column.
"""

import csv
import sys
import uuid
from pathlib import Path


def add_measurement_ids(csv_file_path: str) -> None:
    """
    Add unique measurement_id as first column to CSV file.

    Args:
        csv_file_path: Path to the CSV file to update
    """
    file_path = Path(csv_file_path)

    if not file_path.exists():
        print(f"Error: File {csv_file_path} does not exist")
        sys.exit(1)

    # Read all rows
    rows = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)

    if not rows:
        print("Error: CSV file is empty")
        sys.exit(1)

    # Check if measurement_id already exists
    header = rows[0]
    if 'measurement_id' in header:
        print("Warning: measurement_id column already exists. Skipping...")
        return

    # Add measurement_id to header
    header.insert(0, 'measurement_id')

    # Add unique IDs to each data row
    for i in range(1, len(rows)):
        measurement_id = str(uuid.uuid4())
        rows[i].insert(0, measurement_id)

    # Write back to file
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    print(f"Successfully added measurement_id column to {len(rows) - 1} rows in {csv_file_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_measurement_ids.py <csv_file_path>")
        print("Example: python add_measurement_ids.py data/2025-09-05_all.csv")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    add_measurement_ids(csv_file_path)
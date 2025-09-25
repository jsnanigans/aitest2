#!/usr/bin/env python3
"""
Filter users by impact score from user analysis results.

Creates a CSV file compatible with the filtered_users_csv config in main.py.
The output format is: user_id,removed_count
"""

import argparse
import csv
from pathlib import Path


def filter_users_by_impact(
    input_file: str, output_file: str, threshold: float = 0.1
) -> int:
    """
    Filter users from analysis results based on impact score.

    Args:
        input_file: Path to user_analysis_results CSV
        output_file: Path to output CSV (format: user_id,removed_count)
        threshold: Minimum impact_score to include user (default: 0.1)

    Returns:
        Number of users written to output
    """
    users_written = 0

    # Read input CSV
    with open(input_file, "r") as infile:
        reader = csv.DictReader(infile)

        # Collect filtered users
        filtered_users = []

        for row in reader:
            try:
                impact_score = float(row.get("impact_score", 0))

                # Check if impact score meets threshold
                if impact_score > threshold:
                    user_id = row.get("user_id", "").strip()

                    # Calculate removed_count from the data
                    # removed_count = raw_measurement_count - filtered_measurement_count
                    raw_count = int(row.get("raw_measurement_count", 0))
                    filtered_count = int(row.get("filtered_measurement_count", 0))
                    removed_count = raw_count - filtered_count

                    if user_id:
                        filtered_users.append(
                            {
                                "user_id": user_id,
                                "removed_count": removed_count,
                                "impact_score": impact_score,  # Keep for reporting
                            }
                        )
            except (ValueError, TypeError) as e:
                print(f"Warning: Skipping row due to error: {e}")
                continue

        # Sort by impact score (descending) for better visibility
        filtered_users.sort(key=lambda x: x["impact_score"], reverse=True)

        # Write output CSV in the format expected by main.py
        with open(output_file, "w", newline="") as outfile:
            writer = csv.writer(outfile)

            # Write header
            writer.writerow(["user_id", "removed_count"])

            # Write user data
            for user in filtered_users:
                writer.writerow([user["user_id"], user["removed_count"]])
                users_written += 1

        # Report summary
        print(f"\nFiltered Users Summary:")
        print(f"  Input file: {input_file}")
        print(f"  Output file: {output_file}")
        print(f"  Impact score threshold: > {threshold}")
        print(f"  Users meeting criteria: {users_written}")

        if users_written > 0:
            print(f"\nTop 5 users by impact score:")
            for i, user in enumerate(filtered_users[:5], 1):
                print(
                    f"  {i}. {user['user_id'][:8]}... - Impact: {user['impact_score']:.3f}, Removed: {user['removed_count']}"
                )

    return users_written


def main():
    parser = argparse.ArgumentParser(
        description="Filter users by impact score from analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Filter users with impact score > 0.1 (default)
  uv run python scripts/filter_users_by_impact.py \\
    apple_0a427a45-cebe-4cec-977b-f65a9b6534bc/user_analysis_results_20250922_155547.csv \\
    filtered_users.csv

  # Filter users with impact score > 0.5
  uv run python scripts/filter_users_by_impact.py \\
    apple_0a427a45-cebe-4cec-977b-f65a9b6534bc/user_analysis_results_20250922_155547.csv \\
    filtered_users.csv \\
    --threshold 0.5

  # Then use with main.py:
  uv run python main.py data/weights.csv --filtered-users-csv filtered_users.csv
        """,
    )

    parser.add_argument("input_file", help="Path to user_analysis_results CSV file")
    parser.add_argument(
        "output_file", help="Path to output CSV file (will be created/overwritten)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Minimum impact score threshold (default: 0.1)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process the file
    users_count = filter_users_by_impact(
        args.input_file, args.output_file, args.threshold
    )

    if users_count == 0:
        print(f"\nWarning: No users found with impact_score > {args.threshold}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

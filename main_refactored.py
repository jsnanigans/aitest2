#!/usr/bin/env python3
"""
Refactored Weight Stream Processor using new component architecture.
"""

import argparse
import sys
from pathlib import Path

from src.batch.csv_processor import CSVBatchProcessor
from src.factories.component_factory import ComponentFactory


def load_config(config_path: str = "config.toml") -> dict:
    """Load configuration from file."""
    return ComponentFactory.get_config('file', config_path)


def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(
        description="Process weight measurements from CSV"
    )
    parser.add_argument(
        "csv_path",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Directory for outputs (default: output)"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.toml",
        help="Configuration file path (default: config.toml)"
    )
    parser.add_argument(
        "--filtered-output", "-f",
        help="Path to write filtered CSV with accepted measurements"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations"
    )
    parser.add_argument(
        "--test-users",
        help="Test user IDs (comma-separated) to process exclusively"
    )
    parser.add_argument(
        "--test-users-file",
        help="File containing test user IDs (one per line)"
    )
    parser.add_argument(
        "--filtered-users-csv",
        help="CSV file containing user IDs to process (from create-report output)"
    )
    parser.add_argument(
        "--min-date",
        help="Minimum date for measurements (ISO format)"
    )
    parser.add_argument(
        "--max-date",
        help="Maximum date for measurements (ISO format)"
    )
    parser.add_argument(
        "--max-users",
        type=int,
        help="Maximum number of users to process"
    )
    parser.add_argument(
        "--user-offset",
        type=int,
        default=0,
        help="Skip first N eligible users (default: 0)"
    )
    parser.add_argument(
        "--min-readings",
        type=int,
        default=0,
        help="Minimum readings required per user (default: 0)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )

    args = parser.parse_args()

    # Validate CSV path
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration")
        config = ComponentFactory.get_config('auto')

    # Override config with command-line arguments
    if args.test_users:
        config.setdefault("data", {})["test_users"] = args.test_users.split(",")
    if args.test_users_file:
        config.setdefault("data", {})["test_users_file"] = args.test_users_file
    if args.filtered_users_csv:
        config.setdefault("data", {})["filtered_users_csv"] = args.filtered_users_csv
    if args.min_date:
        config.setdefault("data", {})["min_date"] = args.min_date
    if args.max_date:
        config.setdefault("data", {})["max_date"] = args.max_date
    if args.max_users is not None:
        config.setdefault("data", {})["max_users"] = args.max_users
    if args.user_offset is not None:
        config.setdefault("data", {})["user_offset"] = args.user_offset
    if args.min_readings is not None:
        config.setdefault("data", {})["min_readings"] = args.min_readings

    # Set visualization flag
    config.setdefault("visualization", {})["enabled"] = args.visualize

    # Create and run processor
    processor = CSVBatchProcessor()

    try:
        stats = processor.process_file(
            csv_path=str(csv_path),
            output_dir=args.output_dir,
            config=config,
            filtered_output=args.filtered_output,
            debug=args.debug
        )

        # Print final summary
        print(f"\n{'=' * 60}")
        print("PROCESSING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Total rows processed: {stats['total_rows']:,}")
        print(f"Measurements accepted: {stats['accepted']:,}")
        print(f"Measurements rejected: {stats['rejected']:,}")

        if stats.get('unit_rejected'):
            print(f"\nUnit validation failures: {stats['unit_rejected']:,}")
            if stats.get('rejected_units'):
                print("Rejected units:")
                for unit, count in sorted(stats['rejected_units'].items(),
                                         key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {unit}: {count:,}")

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError during processing: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
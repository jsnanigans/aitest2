#!/bin/bash
# Run local CSV processing

set -e

# Default values
CSV_FILE="${1:-data/weights.csv}"
CONFIG_FILE="${2:-config/local.toml}"

echo "Running local weight processor..."
echo "  CSV file: $CSV_FILE"
echo "  Config: $CONFIG_FILE"

# Change to local directory
cd local

# Run the processor
python main.py "../$CSV_FILE" --config "../$CONFIG_FILE"

echo "Done!"
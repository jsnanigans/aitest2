#!/bin/bash

# Example pipeline for preprocessing and processing weight data

# Configuration
INPUT_FILE="data/2025-09-05.csv"
PREPROCESSED_FILE="data/2025-09-05_optimized.csv"
MIN_DATE="2020-01-01"
MAX_DATE="2025-12-31"

# Sources to ignore (modify as needed)
IGNORE_SOURCES=(
    "test-source"
    "manual-entry"
    "debug-import"
)

echo "Weight Data Processing Pipeline"
echo "================================"
echo ""

# Step 1: Preprocess the data
echo "Step 1: Preprocessing CSV data..."
echo "  Input: $INPUT_FILE"
echo "  Output: $PREPROCESSED_FILE"
echo "  Date range: $MIN_DATE to $MAX_DATE"
echo ""

IGNORE_ARGS=""
for source in "${IGNORE_SOURCES[@]}"; do
    IGNORE_ARGS="$IGNORE_ARGS --ignore-source \"$source\""
done

eval uv run python scripts/preprocess_csv.py \
    "$INPUT_FILE" \
    -o "$PREPROCESSED_FILE" \
    --min-date "$MIN_DATE" \
    --max-date "$MAX_DATE" \
    $IGNORE_ARGS

if [ $? -ne 0 ]; then
    echo "Error: Preprocessing failed"
    exit 1
fi

echo ""
echo "Step 2: Processing with main weight stream processor..."
echo ""

# Step 2: Run the main processor
uv run python main.py "$PREPROCESSED_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Main processing failed"
    exit 1
fi

echo ""
echo "Pipeline completed successfully!"
echo "Results available in output/ directory"
#!/bin/bash
#
# Run both Python and TypeScript weight processors with verbose logging
# Outputs logs to logs_py.txt and logs_ts.txt
#

set -e

# Configuration
CSV_FILE="${1:-test_user.csv}"
USER_ID="ADC64C0B-CB46-41F9-BDA0-CC11A35942D7"
OUTPUT_DIR="output_local"

echo "========================================="
echo "Running Weight Processors with Logging"
echo "========================================="
echo "Input: $CSV_FILE"
echo "User ID: $USER_ID"
echo ""

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Run Python processor
echo "🐍 Running Python processor..."
VERBOSE_LOGGING=true uv run python local_main.py \
  --csv-file "$CSV_FILE" \
  --user-ids "$USER_ID" \
  --min-readings 0 \
  --output-dir "$OUTPUT_DIR" \
  --filtered-csv filtered_weights_py.csv \
  > logs_py.txt 2>&1

echo "   ✓ Python complete - logs saved to logs_py.txt"

# Run TypeScript processor
echo "📘 Running TypeScript processor..."
VERBOSE_LOGGING=true bun run weight-processor-ts/local_main.ts \
  --csv-file "$CSV_FILE" \
  --user-ids "$USER_ID" \
  --min-readings 0 \
  --output-dir "$OUTPUT_DIR" \
  --filtered-csv filtered_weights_ts.csv \
  > logs_ts.txt 2>&1

echo "   ✓ TypeScript complete - logs saved to logs_ts.txt"

echo ""
echo "========================================="
echo "Summary"
echo "========================================="

# Count log lines
PY_LOGS=$(grep -c "\[PY\]" logs_py.txt || true)
TS_LOGS=$(grep -c "\[TS\]" logs_ts.txt || true)

echo "Python logs:     $PY_LOGS lines"
echo "TypeScript logs: $TS_LOGS lines"

# Count CSV lines
PY_CSV=$(wc -l < filtered_weights_py.csv)
TS_CSV=$(wc -l < filtered_weights_ts.csv)
PY_ACCEPTED=$((PY_CSV - 1))
TS_ACCEPTED=$((TS_CSV - 1))

echo ""
echo "Python accepted:     $PY_ACCEPTED measurements"
echo "TypeScript accepted: $TS_ACCEPTED measurements"

# Calculate difference
DIFF=$((TS_ACCEPTED - PY_ACCEPTED))
if [ $DIFF -eq 0 ]; then
  echo ""
  echo "✅ MATCH: Both implementations accepted the same number of measurements"
else
  echo ""
  echo "❌ DIVERGENCE: Difference of $DIFF measurements"
  if [ $DIFF -gt 0 ]; then
    echo "   (TypeScript accepted $DIFF more)"
  else
    echo "   (Python accepted ${DIFF#-} more)"
  fi
fi

echo ""
echo "Files generated:"
echo "  - logs_py.txt (Python logs)"
echo "  - logs_ts.txt (TypeScript logs)"
echo "  - filtered_weights_py.csv (Python output)"
echo "  - filtered_weights_ts.csv (TypeScript output)"
echo ""

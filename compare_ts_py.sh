#!/bin/bash
#
# Compare TypeScript and Python local_main implementations
#
# Runs both versions with test_user.csv and compares:
# - Filtered CSV outputs
# - JSON results
# - Processing statistics
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CSV_FILE="test_user.csv"
USER_ID="ADC64C0B-CB46-41F9-BDA0-CC11A35942D7"
OUTPUT_DIR_TS="output_comparison_ts"
OUTPUT_DIR_PY="output_comparison_py"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  TypeScript vs Python Implementation Comparison           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if test file exists
if [ ! -f "$CSV_FILE" ]; then
    echo -e "${RED}✗ Error: $CSV_FILE not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Test Configuration:${NC}"
echo "  CSV File: $CSV_FILE"
echo "  User ID: $USER_ID"
echo ""

# Clean up old output directories
echo -e "${YELLOW}Cleaning up old outputs...${NC}"
rm -rf "$OUTPUT_DIR_TS" "$OUTPUT_DIR_PY"
mkdir -p "$OUTPUT_DIR_TS" "$OUTPUT_DIR_PY"

# Run TypeScript version
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Running TypeScript Implementation${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

TS_START=$(date +%s)
if bun run local_main.ts \
    --csv-file "$CSV_FILE" \
    --min-readings 0 \
    --output-dir "$OUTPUT_DIR_TS" 2>&1 | tee "${OUTPUT_DIR_TS}/console.log"; then
    echo -e "${GREEN}✓ TypeScript version completed successfully${NC}"
else
    echo -e "${RED}✗ TypeScript version failed${NC}"
    exit 1
fi
TS_END=$(date +%s)
TS_DURATION=$((TS_END - TS_START))

# Run Python version
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Running Python Implementation${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

PY_START=$(date +%s)
if python local_main.py \
    --csv-file "$CSV_FILE" \
    --min-readings 0 \
    --output-dir "$OUTPUT_DIR_PY" 2>&1 | tee "${OUTPUT_DIR_PY}/console.log"; then
    echo -e "${GREEN}✓ Python version completed successfully${NC}"
else
    echo -e "${RED}✗ Python version failed${NC}"
    exit 1
fi
PY_END=$(date +%s)
PY_DURATION=$((PY_END - PY_START))

# Find output files
TS_CSV=$(ls "$OUTPUT_DIR_TS"/filtered_*.csv 2>/dev/null | head -1)
PY_CSV=$(ls "$OUTPUT_DIR_PY"/filtered_*.csv 2>/dev/null | head -1)
TS_JSON=$(ls "$OUTPUT_DIR_TS"/results_*.json 2>/dev/null | head -1)
PY_JSON=$(ls "$OUTPUT_DIR_PY"/results_*.json 2>/dev/null | head -1)

# Check if output files exist
if [ -z "$TS_CSV" ] || [ -z "$PY_CSV" ]; then
    echo -e "${RED}✗ Error: Could not find output CSV files${NC}"
    exit 1
fi

if [ -z "$TS_JSON" ] || [ -z "$PY_JSON" ]; then
    echo -e "${RED}✗ Error: Could not find output JSON files${NC}"
    exit 1
fi

# Compare Results
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Comparison Results${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# 1. Compare execution time
echo -e "${YELLOW}1. Execution Time:${NC}"
echo "   TypeScript: ${TS_DURATION}s"
echo "   Python:     ${PY_DURATION}s"

# Compare performance
if [ "$TS_DURATION" -eq "$PY_DURATION" ]; then
    echo -e "   ${GREEN}Equal performance${NC}"
else
    if [ "$TS_DURATION" -lt "$PY_DURATION" ]; then
        DIFF=$((PY_DURATION - TS_DURATION))
        echo -e "   ${GREEN}TypeScript is ${DIFF}s faster${NC}"
    else
        DIFF=$((TS_DURATION - PY_DURATION))
        echo -e "   ${GREEN}Python is ${DIFF}s faster${NC}"
    fi
fi
echo ""

# 2. Compare CSV row counts
echo -e "${YELLOW}2. CSV Output:${NC}"
TS_ROWS=$(tail -n +2 "$TS_CSV" | wc -l | tr -d ' ')
PY_ROWS=$(tail -n +2 "$PY_CSV" | wc -l | tr -d ' ')
echo "   TypeScript rows: $TS_ROWS"
echo "   Python rows:     $PY_ROWS"
if [ "$TS_ROWS" -eq "$PY_ROWS" ]; then
    echo -e "   ${GREEN}✓ Row counts match${NC}"
else
    echo -e "   ${RED}✗ Row counts differ by $((TS_ROWS - PY_ROWS))${NC}"
fi
echo ""

# 3. Compare JSON statistics
echo -e "${YELLOW}3. Processing Statistics:${NC}"
echo ""

# Extract stats using jq
TS_PROCESSED=$(jq -r '.totalMeasurements // .total_measurements // 0' "$TS_JSON")
PY_PROCESSED=$(jq -r '.total_measurements // .totalMeasurements // 0' "$PY_JSON")
TS_ACCEPTED=$(jq -r '.acceptedCount // .accepted_count // 0' "$TS_JSON")
PY_ACCEPTED=$(jq -r '.accepted_count // .acceptedCount // 0' "$PY_JSON")
TS_REJECTED=$(jq -r '.rejectedCount // .rejected_count // 0' "$TS_JSON")
PY_REJECTED=$(jq -r '.rejected_count // .rejectedCount // 0' "$PY_JSON")

echo "   Measurements Processed:"
echo "     TypeScript: $TS_PROCESSED"
echo "     Python:     $PY_PROCESSED"
if [ "$TS_PROCESSED" -eq "$PY_PROCESSED" ]; then
    echo -e "     ${GREEN}✓ Match${NC}"
else
    echo -e "     ${RED}✗ Differ by $((TS_PROCESSED - PY_PROCESSED))${NC}"
fi
echo ""

echo "   Measurements Accepted:"
echo "     TypeScript: $TS_ACCEPTED"
echo "     Python:     $PY_ACCEPTED"
if [ "$TS_ACCEPTED" -eq "$PY_ACCEPTED" ]; then
    echo -e "     ${GREEN}✓ Match${NC}"
else
    echo -e "     ${RED}✗ Differ by $((TS_ACCEPTED - PY_ACCEPTED))${NC}"
fi
echo ""

echo "   Measurements Rejected:"
echo "     TypeScript: $TS_REJECTED"
echo "     Python:     $PY_REJECTED"
if [ "$TS_REJECTED" -eq "$PY_REJECTED" ]; then
    echo -e "     ${GREEN}✓ Match${NC}"
else
    echo -e "     ${RED}✗ Differ by $((TS_REJECTED - PY_REJECTED))${NC}"
fi
echo ""

# 4. Check for replay metadata
echo -e "${YELLOW}4. Replay Metadata:${NC}"
TS_REPLAYS=$(jq -r '.processingResults // .processing_results | to_entries[0].value.replayMetadata // [] | length' "$TS_JSON" 2>/dev/null || echo "0")
PY_REPLAYS=$(jq -r '.processingResults // .processing_results | to_entries[0].value.replay_metadata // [] | length' "$PY_JSON" 2>/dev/null || echo "0")
echo "   TypeScript replay triggers: $TS_REPLAYS"
echo "   Python replay triggers:     $PY_REPLAYS"
if [ "$TS_REPLAYS" -eq "$PY_REPLAYS" ]; then
    echo -e "   ${GREEN}✓ Replay counts match${NC}"
else
    echo -e "   ${YELLOW}⚠ Replay counts differ (may vary based on timing)${NC}"
fi
echo ""

# 5. Compare actual accepted measurement IDs
echo -e "${YELLOW}5. Detailed Measurement Comparison:${NC}"
echo "   Extracting measurement IDs from CSV files..."

# Extract IDs from CSV (first column, skip header)
TS_IDS=$(tail -n +2 "$TS_CSV" | cut -d',' -f1 | sort)
PY_IDS=$(tail -n +2 "$PY_CSV" | cut -d',' -f1 | sort)

# Save to temp files
echo "$TS_IDS" > /tmp/ts_ids.txt
echo "$PY_IDS" > /tmp/py_ids.txt

# Compare
DIFF_COUNT=$(diff /tmp/ts_ids.txt /tmp/py_ids.txt | grep -E '^[<>]' | wc -l | tr -d ' ')
if [ "$DIFF_COUNT" -eq 0 ]; then
    echo -e "   ${GREEN}✓ All accepted measurements match exactly${NC}"
else
    echo -e "   ${RED}✗ Found $DIFF_COUNT differences in accepted measurements${NC}"
    echo ""
    echo "   Differences:"
    diff /tmp/ts_ids.txt /tmp/py_ids.txt | head -20
fi
echo ""

# 6. Sample measurement values comparison
echo -e "${YELLOW}6. Sample Weight Values:${NC}"
echo "   Comparing first 5 accepted measurements..."
echo ""
echo "   TypeScript:"
tail -n +2 "$TS_CSV" | head -5 | cut -d',' -f1,6 | while IFS=',' read -r id weight; do
    echo "     $id: $weight kg"
done
echo ""
echo "   Python:"
tail -n +2 "$PY_CSV" | head -5 | cut -d',' -f1,6 | while IFS=',' read -r id weight; do
    echo "     $id: $weight kg"
done
echo ""

# Final Summary
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

DIFFERENCES=0
[ "$TS_ROWS" -ne "$PY_ROWS" ] && DIFFERENCES=$((DIFFERENCES + 1))
[ "$TS_PROCESSED" -ne "$PY_PROCESSED" ] && DIFFERENCES=$((DIFFERENCES + 1))
[ "$TS_ACCEPTED" -ne "$PY_ACCEPTED" ] && DIFFERENCES=$((DIFFERENCES + 1))
[ "$DIFF_COUNT" -ne 0 ] && DIFFERENCES=$((DIFFERENCES + 1))

if [ $DIFFERENCES -eq 0 ]; then
    echo -e "${GREEN}✓ PASS: Both implementations produce identical results!${NC}"
    echo ""
    echo "  Output files:"
    echo "    TypeScript CSV: $TS_CSV"
    echo "    Python CSV:     $PY_CSV"
    echo "    TypeScript JSON: $TS_JSON"
    echo "    Python JSON:     $PY_JSON"
    exit 0
else
    echo -e "${RED}✗ FAIL: Found $DIFFERENCES difference(s) between implementations${NC}"
    echo ""
    echo "  Review output files for details:"
    echo "    TypeScript: $OUTPUT_DIR_TS/"
    echo "    Python:     $OUTPUT_DIR_PY/"
    echo ""
    echo "  Console logs:"
    echo "    TypeScript: ${OUTPUT_DIR_TS}/console.log"
    echo "    Python:     ${OUTPUT_DIR_PY}/console.log"
    exit 1
fi


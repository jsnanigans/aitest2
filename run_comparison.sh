#!/bin/bash
# Quick comparison of TypeScript and Python implementations

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

CSV_FILE="test_user.csv"
OUTPUT_DIR_TS="output_comparison_ts"
OUTPUT_DIR_PY="output_comparison_py"

echo -e "${BLUE}=== TypeScript vs Python Comparison ===${NC}"
echo ""

# Check test file
if [ ! -f "$CSV_FILE" ]; then
    echo -e "${RED}Error: $CSV_FILE not found${NC}"
    exit 1
fi

# Clean up
echo "Cleaning up old outputs..."
rm -rf "$OUTPUT_DIR_TS" "$OUTPUT_DIR_PY"
mkdir -p "$OUTPUT_DIR_TS" "$OUTPUT_DIR_PY"

# Run TypeScript
echo ""
echo -e "${YELLOW}Running TypeScript...${NC}"
bun run local_main.ts \
    --csv-file "$CSV_FILE" \
    --min-readings 0 \
    --output-dir "$OUTPUT_DIR_TS" \
    2>&1 | tee "$OUTPUT_DIR_TS/console.log"

# Run Python
echo ""
echo -e "${YELLOW}Running Python...${NC}"
uv run python local_main.py \
    --csv-file "$CSV_FILE" \
    --min-readings 0 \
    --output-dir "$OUTPUT_DIR_PY" \
    2>&1 | tee "$OUTPUT_DIR_PY/console.log"

# Compare
echo ""
echo -e "${BLUE}=== Comparison ===${NC}"

TS_CSV=$(ls "$OUTPUT_DIR_TS"/filtered_*.csv | head -1)
PY_CSV=$(ls "$OUTPUT_DIR_PY"/filtered_*.csv | head -1)

TS_ROWS=$(tail -n +2 "$TS_CSV" | wc -l | tr -d ' ')
PY_ROWS=$(tail -n +2 "$PY_CSV" | wc -l | tr -d ' ')

echo "CSV Rows:"
echo "  TypeScript: $TS_ROWS"
echo "  Python:     $PY_ROWS"

if [ "$TS_ROWS" = "$PY_ROWS" ]; then
    echo -e "  ${GREEN}✓ MATCH${NC}"

    # Compare actual IDs
    tail -n +2 "$TS_CSV" | cut -d',' -f1 | sort > /tmp/ts_ids.txt
    tail -n +2 "$PY_CSV" | cut -d',' -f1 | sort > /tmp/py_ids.txt

    if diff -q /tmp/ts_ids.txt /tmp/py_ids.txt > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS: All accepted measurements match exactly!${NC}"
        exit 0
    else
        echo -e "${RED}✗ FAIL: Accepted measurements differ${NC}"
        diff /tmp/ts_ids.txt /tmp/py_ids.txt | head -20
        exit 1
    fi
else
    echo -e "  ${RED}✗ FAIL: Row counts differ${NC}"

    # Show which IDs differ
    echo ""
    echo "Comparing accepted IDs..."
    tail -n +2 "$TS_CSV" | cut -d',' -f1 | sort > /tmp/ts_ids.txt
    tail -n +2 "$PY_CSV" | cut -d',' -f1 | sort > /tmp/py_ids.txt

    echo "IDs in TypeScript but not Python:"
    comm -23 /tmp/ts_ids.txt /tmp/py_ids.txt | head -10

    echo ""
    echo "IDs in Python but not TypeScript:"
    comm -13 /tmp/ts_ids.txt /tmp/py_ids.txt | head -10

    exit 1
fi

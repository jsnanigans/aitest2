#!/bin/bash
#
# Quick comparison of TypeScript and Python implementations
#

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
rm -rf "$OUTPUT_DIR_TS" "$OUTPUT_DIR_PY"
mkdir -p "$OUTPUT_DIR_TS" "$OUTPUT_DIR_PY"

# Run TypeScript
echo -e "${YELLOW}Running TypeScript...${NC}"
if bun run local_main.ts --csv-file "$CSV_FILE" --min-readings 0 --output-dir "$OUTPUT_DIR_TS" > "$OUTPUT_DIR_TS/console.log" 2>&1; then
    echo -e "${GREEN}✓ TypeScript completed${NC}"
else
    echo -e "${RED}✗ TypeScript failed${NC}"
    cat "$OUTPUT_DIR_TS/console.log"
    exit 1
fi

# Run Python
echo -e "${YELLOW}Running Python...${NC}"
if python local_main.py --csv-file "$CSV_FILE" --min-readings 0 --output-dir "$OUTPUT_DIR_PY" > "$OUTPUT_DIR_PY/console.log" 2>&1; then
    echo -e "${GREEN}✓ Python completed${NC}"
else
    echo -e "${RED}✗ Python failed${NC}"
    cat "$OUTPUT_DIR_PY/console.log"
    exit 1
fi

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

if [ "$TS_ROWS" -eq "$PY_ROWS" ]; then
    echo -e "  ${GREEN}✓ MATCH${NC}"

    # Compare actual IDs
    TS_IDS=$(tail -n +2 "$TS_CSV" | cut -d',' -f1 | sort)
    PY_IDS=$(tail -n +2 "$PY_CSV" | cut -d',' -f1 | sort)

    echo "$TS_IDS" > /tmp/ts_ids.txt
    echo "$PY_IDS" > /tmp/py_ids.txt

    if diff -q /tmp/ts_ids.txt /tmp/py_ids.txt > /dev/null 2>&1; then
        echo -e "${GREEN}✓ All accepted measurements match exactly!${NC}"
        exit 0
    else
        echo -e "${RED}✗ Accepted measurements differ${NC}"
        exit 1
    fi
else
    echo -e "  ${RED}✗ DIFFER${NC}"
    exit 1
fi

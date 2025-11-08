#!/bin/bash

USER_ID="ADC64C0B-CB46-41F9-BDA0-CC11A35942D7"
CSV_FILE="test_user.csv"

echo "=== Debugging User $USER_ID ==="
echo ""

# Extract just this user's data
echo "Extracting user data..."
head -1 "$CSV_FILE" > /tmp/debug_user.csv
grep "$USER_ID" "$CSV_FILE" | head -30 >> /tmp/debug_user.csv

echo "User measurements:"
cat /tmp/debug_user.csv | head -10

echo ""
echo "Total measurements for user: $(tail -n +2 /tmp/debug_user.csv | wc -l)"
echo ""

# Run TypeScript
echo "=== Running TypeScript ==="
bun local_main.ts \
    --csv-file /tmp/debug_user.csv \
    --min-readings 0 \
    --output-dir /tmp/ts_debug 2>&1 | tee /tmp/ts_debug.log

echo ""
echo "TypeScript accepted IDs:"
tail -n +2 /tmp/ts_debug/filtered_*.csv | cut -d',' -f1 | sort

echo ""
echo "=== Running Python ==="
uv run python local_main.py \
    --csv-file /tmp/debug_user.csv \
    --min-readings 0 \
    --output-dir /tmp/py_debug 2>&1 | tee /tmp/py_debug.log

echo ""
echo "Python accepted IDs:"
tail -n +2 /tmp/py_debug/filtered_*.csv | cut -d',' -f1 | sort

echo ""
echo "=== Differences ==="
tail -n +2 /tmp/ts_debug/filtered_*.csv | cut -d',' -f1 | sort > /tmp/ts_ids_debug.txt
tail -n +2 /tmp/py_debug/filtered_*.csv | cut -d',' -f1 | sort > /tmp/py_ids_debug.txt

echo "TypeScript only:"
comm -23 /tmp/ts_ids_debug.txt /tmp/py_ids_debug.txt

echo ""
echo "Python only:"
comm -13 /tmp/ts_ids_debug.txt /tmp/py_ids_debug.txt
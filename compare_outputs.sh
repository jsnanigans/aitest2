#!/bin/bash

# Compare TypeScript and Python outputs

echo "=== Comparison ==="
echo "CSV Rows:"
echo "  TypeScript: $(tail -n +2 output_comparison_ts/filtered_*.csv | wc -l | tr -d ' ')"
echo "  Python:     $(tail -n +2 output_comparison_py/filtered_*.csv | wc -l | tr -d ' ')"
echo ""

awk -F',' 'NR>1 {print $1}' output_comparison_ts/filtered_*.csv | sort > /tmp/ts_ids.txt
awk -F',' 'NR>1 {print $1}' output_comparison_py/filtered_*.csv | sort > /tmp/py_ids.txt

echo "Comparing accepted IDs..."
comm -13 /tmp/ts_ids.txt /tmp/py_ids.txt > /tmp/only_py.txt
comm -23 /tmp/ts_ids.txt /tmp/py_ids.txt > /tmp/only_ts.txt

if [ -s /tmp/only_ts.txt ]; then
  echo "IDs in TypeScript but not Python:"
  cat /tmp/only_ts.txt
  echo ""
fi

if [ -s /tmp/only_py.txt ]; then
  echo "IDs in Python but not TypeScript:"
  cat /tmp/only_py.txt
  echo ""
fi

if [ ! -s /tmp/only_ts.txt ] && [ ! -s /tmp/only_py.txt ]; then
  echo "✓ PASS: All IDs match!"
else
  echo "✗ FAIL: IDs differ"
fi

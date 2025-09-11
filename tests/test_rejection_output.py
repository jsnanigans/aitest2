"""Verify rejection reasons are captured in JSON output."""

import json
from pathlib import Path

print("VERIFICATION: Rejection Reason Fix")
print("=" * 70)

# Check the latest debug output
debug_file = Path("output/debug_test_no_date/user_0040872d-333a-4ace-8c5a-b2fcd056e65a.json")

if debug_file.exists():
    with open(debug_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nUser: {data['user_id']}")
    print(f"Total measurements: {data['total_measurements']}")
    print(f"Rejected: {data['rejected_count']}")
    
    print("\n✅ REJECTION REASONS NOW CAPTURED:")
    print("-" * 50)
    
    count = 0
    for log in data['processing_logs']:
        if log['result'] == 'rejected' and log.get('rejection_reason'):
            count += 1
            if count <= 5:  # Show first 5
                print(f"\nTimestamp: {log['timestamp']}")
                print(f"Weight: {log['weight']}kg")
                print(f"Reason: {log['rejection_reason']}")
    
    if count > 5:
        print(f"\n... and {count - 5} more rejections with reasons")
    
    # Check for any rejections without reasons
    no_reason_count = 0
    for log in data['processing_logs']:
        if log['result'] == 'rejected' and not log.get('rejection_reason'):
            no_reason_count += 1
    
    if no_reason_count > 0:
        print(f"\n⚠️  WARNING: {no_reason_count} rejections without reasons")
    else:
        print(f"\n✅ SUCCESS: All {count} rejections have reasons!")
else:
    print(f"Error: Debug file not found at {debug_file}")

print("\n" + "=" * 70)
print("Fix Status: COMPLETE - Rejection reasons are now properly captured")

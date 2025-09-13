#!/usr/bin/env python3
"""Investigate user counting issue with min_readings filter"""

import csv
from datetime import datetime
from collections import defaultdict

def parse_timestamp(date_str: str) -> datetime:
    """Parse various timestamp formats."""
    if not date_str:
        return datetime.now()
    
    if "T" in date_str:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    elif " " in date_str:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    else:
        return datetime.strptime(date_str, "%Y-%m-%d")

def investigate_users(csv_path: str, min_readings: int = 20, max_users: int = 100, 
                     min_date_str: str = "2024-11-01", max_date_str: str = "2025-09-05"):
    """Investigate user counting with filters"""
    
    min_date = parse_timestamp(min_date_str) if min_date_str else None
    max_date = parse_timestamp(max_date_str) if max_date_str else None
    
    # First pass: count valid readings per user
    user_reading_counts = defaultdict(int)
    total_rows = 0
    
    print(f"First pass: Counting valid readings per user...")
    print(f"  Date filter: {min_date_str} to {max_date_str}")
    print(f"  Min readings required: {min_readings}")
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            user_id = row.get("user_id")
            if not user_id:
                continue
            
            weight_str = row.get("weight", "").strip()
            if not weight_str or weight_str.upper() == "NULL":
                continue
            
            # Parse and check date
            date_str = row.get("effectiveDateTime")
            try:
                timestamp = parse_timestamp(date_str)
            except:
                continue
            
            # Apply date filters
            if min_date and timestamp < min_date:
                continue
            if max_date and timestamp > max_date:
                continue
            
            # Check if weight is valid
            try:
                float(weight_str)
                user_reading_counts[user_id] += 1
            except (ValueError, TypeError):
                continue
    
    print(f"\nFirst pass complete:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Total unique users: {len(user_reading_counts):,}")
    
    # Categorize users by reading count
    users_with_sufficient_data = []
    users_with_insufficient_data = []
    
    for user_id, count in user_reading_counts.items():
        if count >= min_readings:
            users_with_sufficient_data.append((user_id, count))
        else:
            users_with_insufficient_data.append((user_id, count))
    
    users_with_sufficient_data.sort(key=lambda x: x[1], reverse=True)
    users_with_insufficient_data.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nUser categorization:")
    print(f"  Users with >= {min_readings} readings: {len(users_with_sufficient_data):,}")
    print(f"  Users with < {min_readings} readings: {len(users_with_insufficient_data):,}")
    
    # Show distribution of reading counts for insufficient users
    if users_with_insufficient_data:
        reading_distribution = defaultdict(int)
        for _, count in users_with_insufficient_data:
            reading_distribution[count] += 1
        
        print(f"\nDistribution of insufficient users:")
        for count in sorted(reading_distribution.keys()):
            print(f"    {count:2d} readings: {reading_distribution[count]:4d} users")
    
    # Simulate processing with max_users limit
    print(f"\nSimulating processing with max_users={max_users}:")
    
    # Second pass: simulate actual processing
    seen_users = set()
    processed_users = set()
    users_with_results = set()
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = row.get("user_id")
            if not user_id:
                continue
            
            # Skip users with insufficient data (they don't count towards max_users)
            if user_id in [u[0] for u in users_with_insufficient_data]:
                continue
            
            # Track seen users
            if user_id not in seen_users:
                seen_users.add(user_id)
                
                # Check if we've hit max_users limit
                if max_users > 0 and len(processed_users) >= max_users:
                    continue
                
                processed_users.add(user_id)
            
            # Skip if not in processed users
            if max_users > 0 and user_id not in processed_users:
                continue
            
            # Check if this row would produce a result
            weight_str = row.get("weight", "").strip()
            if not weight_str or weight_str.upper() == "NULL":
                continue
            
            date_str = row.get("effectiveDateTime")
            try:
                timestamp = parse_timestamp(date_str)
            except:
                continue
            
            if min_date and timestamp < min_date:
                continue
            if max_date and timestamp > max_date:
                continue
            
            try:
                float(weight_str)
                users_with_results.add(user_id)
            except:
                continue
    
    print(f"  Users seen (with sufficient data): {len(seen_users):,}")
    print(f"  Users marked as processed: {len(processed_users):,}")
    print(f"  Users with actual results: {len(users_with_results):,}")
    
    # This is the key issue - visualizations are created for users_with_results
    print(f"\n⚠️  ISSUE IDENTIFIED:")
    print(f"  Expected visualizations: {max_users}")
    print(f"  Actual visualizations: {len(users_with_results)}")
    
    if len(users_with_results) < max_users:
        print(f"\n  Possible causes:")
        print(f"  1. Only {len(users_with_sufficient_data)} users have >= {min_readings} readings")
        print(f"  2. Some 'processed' users may have all measurements filtered by dates")
        print(f"  3. The max_users limit is applied when a user is first seen,")
        print(f"     but visualizations are only created for users with results")
    
    # Find processed users without results
    processed_without_results = processed_users - users_with_results
    if processed_without_results:
        print(f"\n  Users processed but no results: {len(processed_without_results)}")
        print(f"  Examples: {list(processed_without_results)[:5]}")

if __name__ == "__main__":
    csv_path = "./data/2025-09-05_optimized.csv"
    investigate_users(csv_path)

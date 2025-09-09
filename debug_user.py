#!/usr/bin/env python3
"""Debug script to trace processing of a specific user"""

import csv
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from collections import defaultdict

def parse_timestamp(date_str):
    """Parse timestamp from various formats."""
    if not date_str:
        return datetime.now()
    
    # Remove timezone if present
    if date_str.endswith('Z'):
        date_str = date_str[:-1]
    
    # Try different formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return datetime.now()

def main():
    target_user = "5e81abae-a19a-4f79-99fe-7165842bbf6a"
    csv_path = "./data/2025-09-05_optimized.csv"
    
    # Load config
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    
    max_users = config["data"].get("max_users", 0)
    min_readings = config["data"].get("min_readings", 0)
    user_offset = config["data"].get("user_offset", 0)
    
    print(f"Config: max_users={max_users}, min_readings={min_readings}, user_offset={user_offset}")
    print(f"Tracking user: {target_user}")
    print()
    
    # First pass: count readings
    user_counts = defaultdict(int)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = row.get("user_id")
            if not user_id:
                continue
            weight_str = row.get("weight", "").strip()
            if not weight_str or weight_str.upper() == "NULL":
                continue
            try:
                float(weight_str)
                user_counts[user_id] += 1
            except:
                continue
    
    insufficient_data_users = {u for u, c in user_counts.items() if c < min_readings}
    print(f"Users with insufficient data (<{min_readings} readings): {len(insufficient_data_users)}")
    print(f"Target user has {user_counts[target_user]} readings")
    print(f"Target user insufficient? {target_user in insufficient_data_users}")
    print()
    
    # Second pass: process
    seen_users = set()
    processed_users = set()
    skipped_users = set()
    target_rows_processed = 0
    row_num = 0
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            row_num += 1
            user_id = row.get("user_id")
            if not user_id:
                continue
            
            # Track when we first see our target
            if user_id == target_user and user_id not in seen_users:
                print(f"Row {row_num}: FIRST ENCOUNTER of target user")
                print(f"  Current seen_users count: {len(seen_users)}")
                print(f"  Current processed_users count: {len(processed_users)}")
            
            # Skip insufficient data users
            if user_id in insufficient_data_users:
                if user_id == target_user:
                    print(f"Row {row_num}: Target SKIPPED (insufficient data)")
                continue
            
            if user_id not in seen_users:
                seen_users.add(user_id)
                
                if len(seen_users) <= user_offset:
                    skipped_users.add(user_id)
                    if user_id == target_user:
                        print(f"Row {row_num}: Target SKIPPED (offset)")
                    continue
                
                if max_users > 0 and len(processed_users) >= max_users:
                    if user_id == target_user:
                        print(f"Row {row_num}: Target NOT ADDED (max_users={max_users} reached)")
                    continue
                
                processed_users.add(user_id)
                if user_id == target_user:
                    print(f"Row {row_num}: Target ADDED to processed_users (position #{len(processed_users)})")
            
            if user_id in skipped_users:
                if user_id == target_user:
                    print(f"Row {row_num}: Target row SKIPPED (in skipped_users)")
                continue
            
            if max_users > 0 and user_id not in processed_users:
                if user_id == target_user:
                    print(f"Row {row_num}: Target row SKIPPED (not in processed_users)")
                continue
            
            # If we get here, the row would be processed
            if user_id == target_user:
                weight_str = row.get("weight", "").strip()
                date_str = row.get("effectiveDateTime")
                target_rows_processed += 1
                print(f"Row {row_num}: Target row PROCESSED #{target_rows_processed} - weight={weight_str}, date={date_str}")
    
    print()
    print(f"Summary for {target_user}:")
    print(f"  In processed_users: {target_user in processed_users}")
    print(f"  Total rows that would be processed: {target_rows_processed}")

if __name__ == "__main__":
    main()
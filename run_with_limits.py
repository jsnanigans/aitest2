#!/usr/bin/env python3
"""
Example runner showing how to use max_users and user_offset
for testing or processing in batches
"""

import sys
import tomllib
from pathlib import Path

def main():
    # Load the base config
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    
    # Override with command line arguments if provided
    if len(sys.argv) > 1:
        try:
            config["data"]["max_users"] = int(sys.argv[1])
            print(f"Limiting to {config['data']['max_users']} users")
        except ValueError:
            print("Usage: python run_with_limits.py [max_users] [user_offset]")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            config["data"]["user_offset"] = int(sys.argv[2])
            print(f"Skipping first {config['data']['user_offset']} users")
        except ValueError:
            print("Usage: python run_with_limits.py [max_users] [user_offset]")
            sys.exit(1)
    
    # Import and run
    from main import stream_process
    
    csv_file = config["data"]["csv_file"]
    if not Path(csv_file).exists():
        print(f"Error: File {csv_file} not found")
        sys.exit(1)
    
    stream_process(csv_file, config["data"]["output_dir"], config)

if __name__ == "__main__":
    main()
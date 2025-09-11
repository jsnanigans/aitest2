#!/usr/bin/env python3
"""Parse height values CSV and keep only the most recent reading per user."""

import csv
from datetime import datetime
from pathlib import Path

def parse_height_data(input_file: str, output_file: str) -> None:
    """Parse height CSV and keep only most recent reading per user."""
    
    user_heights = {}
    
    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            user_id = row['user_id']
            timestamp = datetime.strptime(row['effective_date_time'], '%Y-%m-%d %H:%M:%S')
            
            if user_id not in user_heights or timestamp > user_heights[user_id]['timestamp']:
                user_heights[user_id] = {
                    'timestamp': timestamp,
                    'source_type': row['source_type'],
                    'value_quantity': row['value_quantity'],
                    'unit': row['unit']
                }
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['user_id', 'effective_date_time', 'source_type', 'value_quantity', 'unit']
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        
        writer.writeheader()
        
        for user_id, data in sorted(user_heights.items()):
            writer.writerow({
                'user_id': user_id,
                'effective_date_time': data['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'source_type': data['source_type'],
                'value_quantity': data['value_quantity'],
                'unit': data['unit']
            })
    
    print(f"Processed {len(user_heights)} unique users")
    print(f"Output saved to: {output_file}")

if __name__ == '__main__':
    input_file = 'data/2025-09-11_height_values.csv'
    output_file = 'data/2025-09-11_height_values_latest.csv'
    
    parse_height_data(input_file, output_file)
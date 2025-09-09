#!/usr/bin/env python3
"""
Simplified Weight Stream Processor
Streams CSV data through Kalman filter, then visualizes results
"""

import csv
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from processor import WeightProcessor
from visualization import create_dashboard


def stream_process(csv_path: str, output_dir: str = "output"):
    """
    Stream process weight data from CSV file.
    
    True streaming: processes each user's data as it arrives,
    keeping only current state. Stores results for visualization.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    processors = {}
    results = defaultdict(list)
    stats = {
        'total_rows': 0,
        'total_users': 0,
        'accepted': 0,
        'rejected': 0,
        'start_time': datetime.now()
    }
    
    print(f"Processing {csv_path}...")
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        current_user = None
        
        for row in reader:
            stats['total_rows'] += 1
            
            if stats['total_rows'] % 10000 == 0:
                print(f"  Processed {stats['total_rows']:,} rows, {len(processors)} users...")
            
            user_id = row.get('user_id')
            if not user_id:
                continue
            
            weight_str = row.get('weight', '').strip()
            if not weight_str or weight_str.upper() == 'NULL':
                continue
            
            try:
                weight = float(weight_str)
            except (ValueError, TypeError):
                continue
            
            date_str = row.get('effectivDateTime') or row.get('date')
            source = row.get('source_type') or row.get('source', 'unknown')
            
            if user_id not in processors:
                processors[user_id] = WeightProcessor(user_id)
                stats['total_users'] = len(processors)
            
            processor = processors[user_id]
            
            try:
                timestamp = parse_timestamp(date_str)
            except:
                timestamp = datetime.now()
            
            result = processor.process_weight(weight, timestamp, source)
            
            if result:
                if result['accepted']:
                    stats['accepted'] += 1
                else:
                    stats['rejected'] += 1
                
                results[user_id].append(result)
    
    elapsed = (datetime.now() - stats['start_time']).total_seconds()
    
    print(f"\nProcessing Complete:")
    print(f"  Total rows: {stats['total_rows']:,}")
    print(f"  Total users: {stats['total_users']:,}")
    print(f"  Accepted: {stats['accepted']:,}")
    print(f"  Rejected: {stats['rejected']:,}")
    print(f"  Time: {elapsed:.1f}s ({stats['total_rows']/elapsed:.0f} rows/sec)")
    print(f"  Acceptance rate: {stats['accepted']/(stats['accepted']+stats['rejected']):.1%}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results_file = output_path / f"results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'stats': stats,
            'users': dict(results)
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")
    
    print("\nGenerating visualizations...")
    viz_dir = output_path / f"viz_{timestamp}"
    viz_dir.mkdir(exist_ok=True)
    
    viz_count = 0
    for user_id, user_results in list(results.items())[:10]:
        if len(user_results) >= 20:
            try:
                create_dashboard(user_id, user_results, str(viz_dir))
                viz_count += 1
            except Exception as e:
                print(f"  Failed to visualize {user_id}: {e}")
    
    print(f"Created {viz_count} visualizations in {viz_dir}")
    
    return results, stats


def parse_timestamp(date_str: str) -> datetime:
    """Parse various timestamp formats."""
    if not date_str:
        return datetime.now()
    
    if 'T' in date_str:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    elif ' ' in date_str:
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    else:
        return datetime.strptime(date_str, '%Y-%m-%d')


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "./data/2025-09-05_optimized.csv"
    
    if not Path(csv_file).exists():
        print(f"Error: File {csv_file} not found")
        sys.exit(1)
    
    stream_process(csv_file)
#!/usr/bin/env python3
"""
Test the integration of reset insights with the main visualization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
import json
from src.processor import process_measurement
from src.database import ProcessorDatabase
from src.visualization import create_weight_timeline

def test_with_real_data():
    """Test with real CSV data that has gaps."""
    
    # Process test data
    csv_file = "data/test_user_0672.csv"
    if not Path(csv_file).exists():
        print(f"Test file {csv_file} not found")
        return
    
    print(f"Processing {csv_file}...")
    
    # Initialize database
    db = ProcessorDatabase()
    
    # Load config
    config_file = Path("config.toml")
    if config_file.exists():
        import toml
        config = toml.load(config_file)
    else:
        config = {}
    
    # Process CSV
    results = []
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        
        # Skip header
        if 'effectiveDateTime' in lines[0] or 'timestamp' in lines[0]:
            lines = lines[1:]
        
        for line in lines[:100]:  # Process first 100 measurements
            parts = line.strip().split(',')
            if len(parts) >= 4:
                # Format: user_id, effectiveDateTime, source_type, weight, unit
                timestamp_str = parts[1]
                source = parts[2]
                weight = float(parts[3])
                
                # Parse timestamp
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                except:
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    except:
                        continue
                
                # Process measurement
                result = process_measurement(
                    user_id="test_0672",
                    weight=weight,
                    timestamp=timestamp,
                    source=source,
                    db=db,
                    config=config
                )
                
                results.append(result)
    
    print(f"Processed {len(results)} measurements")
    
    # Create visualization with reset insights
    output_file = create_weight_timeline(
        results=results,
        user_id="test_0672",
        output_dir="output",
        use_enhanced=False,  # Use basic visualization
        config=config
    )
    
    print(f"Visualization saved to: {output_file}")
    
    # Check for reset events
    reset_count = sum(1 for r in results if r.get('reset_event'))
    print(f"Found {reset_count} reset events in the data")

if __name__ == "__main__":
    test_with_real_data()
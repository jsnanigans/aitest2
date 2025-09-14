#!/usr/bin/env python3
"""Test the new timeline visualization"""

import json
from pathlib import Path
from src.visualization import create_weight_timeline

def test_timeline():
    """Test timeline visualization with sample data"""
    
    # Load test data
    results_file = Path("test_index_output/results_test_no_date.json")
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Get user results
    if 'users' not in data:
        print("No users found in results")
        return
    
    users = data['users']
    if not users:
        print("No users found")
        return
    
    # Get first user
    user_id = list(users.keys())[0]
    user_results = users[user_id]
    
    print(f"Creating timeline for user: {user_id}")
    print(f"Total measurements: {len(user_results)}")
    
    # Create visualization
    try:
        output_file = create_weight_timeline(
            results=user_results,
            user_id=user_id,
            output_dir="test_timeline_output"
        )
        print(f"Timeline created: {output_file}")
        
        # Open in browser
        import webbrowser
        webbrowser.open(f"file://{Path(output_file).absolute()}")
        
    except Exception as e:
        print(f"Error creating timeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_timeline()
#!/usr/bin/env python3
"""
Debug script to test reset parameter processing for specific users.
"""

import sys
import os
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import toml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import process_measurement
from src.database import ProcessorDatabase, get_state_db
from src.reset_manager import ResetManager, ResetType

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_user_processing(user_id: str, csv_file: str, config_file: str = 'config.toml'):
    """Debug processing for a specific user with detailed logging."""
    
    # Load config
    config = toml.load(config_file)
    
    # Load data
    df = pd.read_csv(csv_file)
    df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'])
    user_data = df[df['user_id'] == user_id].sort_values('effectiveDateTime')
    
    if user_data.empty:
        logger.error(f"No data found for user {user_id}")
        return
    
    logger.info(f"Processing {len(user_data)} measurements for user {user_id}")
    
    # Initialize database
    db = get_state_db()
    
    # Create debug log for this user
    debug_dir = Path('output/debug')
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_file = debug_dir / f'user_{user_id[:8]}_{datetime.now():%Y%m%d_%H%M%S}.json'
    
    debug_log = {
        'user_id': user_id,
        'config': {
            'initial_reset': config.get('kalman', {}).get('reset', {}).get('initial', {}),
            'hard_reset': config.get('kalman', {}).get('reset', {}).get('hard', {}),
            'soft_reset': config.get('kalman', {}).get('reset', {}).get('soft', {})
        },
        'measurements': []
    }
    
    # Process each measurement
    for idx, row in user_data.iterrows():
        state = db.get_state(user_id)
        
        # Log pre-processing state
        measurement_log = {
            'index': len(debug_log['measurements']),
            'timestamp': str(row['effectiveDateTime']),
            'weight': row['weight'],
            'source': row['source_type'],
            'pre_state': {
                'has_kalman': bool(state and state.get('kalman_params')),
                'last_weight': state.get('last_accepted_weight') if state else None,
                'measurements_since_reset': state.get('measurements_since_reset', 0) if state else 0,
                'in_adaptation': state.get('in_adaptation_phase', False) if state else False
            }
        }
        
        # Check for reset trigger
        reset_type = ResetManager.should_trigger_reset(
            state, row['weight'], row['effectiveDateTime'], 
            row['source_type'], config
        )
        
        if reset_type:
            reset_params = ResetManager.get_reset_parameters(reset_type, config)
            measurement_log['reset_triggered'] = {
                'type': reset_type.value,
                'parameters': reset_params
            }
            logger.info(f"Reset triggered: {reset_type.value} at {row['effectiveDateTime']}")
        
        # Process measurement (it will get state from db internally)
        result = process_measurement(
            user_id=user_id,
            weight=row['weight'],
            timestamp=row['effectiveDateTime'],
            source=row['source_type'],
            config=config,
            db=db
        )
        
        # Log result
        measurement_log['result'] = {
            'accepted': result.get('accepted', False),
            'rejection_reason': result.get('rejection_reason'),
            'quality_score': result.get('quality_score'),
            'quality_components': result.get('quality_components')
        }
        
        # Log post-processing state if accepted
        if result.get('accepted'):
            # Get the updated state from database
            new_state = db.get_state(user_id)
            if new_state:
                measurement_log['post_state'] = {
                    'kalman_weight': new_state.get('kalman_params', {}).get('mean', [None])[0] if new_state.get('kalman_params') else None,
                    'measurements_since_reset': new_state.get('measurements_since_reset', 0),
                    'in_adaptation': new_state.get('in_adaptation_phase', False),
                    'adaptation_factor': new_state.get('adaptation_factor', 1.0)
                }
        
        debug_log['measurements'].append(measurement_log)
        
        # Print summary
        status = "✓" if result.get('accepted') else "✗"
        reason = f" ({result.get('rejection_reason', '')})" if not result.get('accepted') else ""
        adaptation = " [ADAPT]" if measurement_log['pre_state']['in_adaptation'] else ""
        reset = f" [RESET:{reset_type.value}]" if reset_type else ""
        
        print(f"{status} {len(debug_log['measurements'])-1:3d}: {row['effectiveDateTime']} - {row['weight']:6.1f}kg - "
              f"{row['source_type']:20s}{adaptation}{reset}{reason}")
    
    # Calculate statistics
    accepted = sum(1 for m in debug_log['measurements'] if m['result']['accepted'])
    rejected = len(debug_log['measurements']) - accepted
    
    debug_log['statistics'] = {
        'total': len(debug_log['measurements']),
        'accepted': accepted,
        'rejected': rejected,
        'acceptance_rate': accepted / len(debug_log['measurements']) * 100 if debug_log['measurements'] else 0,
        'resets': sum(1 for m in debug_log['measurements'] if 'reset_triggered' in m)
    }
    
    # Save debug log
    with open(debug_file, 'w') as f:
        json.dump(debug_log, f, indent=2, default=str)
    
    print(f"\n=== Summary ===")
    print(f"Total: {debug_log['statistics']['total']}")
    print(f"Accepted: {accepted} ({debug_log['statistics']['acceptance_rate']:.1f}%)")
    print(f"Rejected: {rejected}")
    print(f"Resets: {debug_log['statistics']['resets']}")
    print(f"\nDebug log saved to: {debug_file}")
    
    return debug_log

def main():
    parser = argparse.ArgumentParser(description='Debug reset processing for specific users')
    parser.add_argument('user_id', help='User ID to debug')
    parser.add_argument('--csv', default='./data/2025-09-05_nocon.csv',
                       help='Input CSV file')
    parser.add_argument('--config', default='config.toml',
                       help='Configuration file')
    
    args = parser.parse_args()
    
    debug_user_processing(args.user_id, args.csv, args.config)

if __name__ == '__main__':
    main()

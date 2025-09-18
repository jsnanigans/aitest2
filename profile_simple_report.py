#!/usr/bin/env python3
import cProfile
import pstats
import io
from pathlib import Path
import sys

# Add the project directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the main function from simple_report
from simple_report import main

def profile_report():
    """Profile the simple_report.py execution"""
    pr = cProfile.Profile()
    pr.enable()
    
    # Run the main function
    try:
        main()
    except SystemExit:
        pass  # Ignore exit codes
    
    pr.disable()
    
    # Get statistics
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions by cumulative time
    
    print("\n=== PROFILING RESULTS ===")
    print(s.getvalue())
    
    # Also print by total time
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
    ps2.print_stats(20)  # Top 20 by total time
    
    print("\n=== BY TOTAL TIME ===")
    print(s2.getvalue())

if __name__ == "__main__":
    profile_report()

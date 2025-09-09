"""
Matplotlib configuration fixes to prevent common warnings
"""

import matplotlib
import matplotlib.pyplot as plt
import warnings

def configure_matplotlib():
    """Configure matplotlib to avoid common issues."""
    
    # Suppress specific matplotlib warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    # Set backend to non-interactive to avoid GUI issues
    matplotlib.use('Agg')
    
    # Increase max ticks if needed (but we fixed the root cause)
    matplotlib.rcParams['axes.formatter.limits'] = (-7, 7)
    
    # Set reasonable DPI for file output
    matplotlib.rcParams['savefig.dpi'] = 100
    
    # Ensure tight layout by default
    matplotlib.rcParams['figure.autolayout'] = True
    
    # Set reasonable figure size defaults
    matplotlib.rcParams['figure.figsize'] = [15, 10]
    
    # Avoid too many ticks
    matplotlib.rcParams['xtick.major.pad'] = 4
    matplotlib.rcParams['ytick.major.pad'] = 4

# Apply configuration on import
configure_matplotlib()
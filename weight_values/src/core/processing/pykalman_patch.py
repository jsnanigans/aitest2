"""
Monkey patch for pykalman to work with Python 3.11+
Fixes the deprecated inspect.getargspec issue.
"""

import inspect
import sys

def patch_pykalman():
    """Apply compatibility patches for pykalman on Python 3.11+"""
    if sys.version_info >= (3, 11):
        # Replace getargspec with getfullargspec for Python 3.11+
        if not hasattr(inspect, 'getargspec'):
            inspect.getargspec = inspect.getfullargspec
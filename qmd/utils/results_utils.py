"""
Shared utilities for managing results directory structure.
"""

import os

def ensure_results_dir(subdir=""):
    """Ensure results directory structure exists."""
    results_path = os.path.join(".", "results", subdir) if subdir else "./results"
    os.makedirs(results_path, exist_ok=True)
    return results_path

def get_results_path(filename, subdir=""):
    """Get full path for output file in results directory."""
    results_dir = ensure_results_dir(subdir)
    return os.path.join(results_dir, filename)
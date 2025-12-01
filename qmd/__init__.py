"""
Quantum Molecular Dynamics (QMD) Package

A high-performance quantum molecular dynamics simulation package for proteins
with integrated visualization and analysis tools.
"""

__version__ = "0.1.0"
__author__ = "QMD Developer"

from .core.system import System
from .core.integrator import BornOppenheimerIntegrator
from .quantum.calculator import QuantumCalculator
from .visualization.viewer import MolecularViewer
from .utils.trajectory import TrajectoryAnalyzer

__all__ = [
    "System",
    "BornOppenheimerIntegrator",
    "QuantumCalculator",
    "MolecularViewer",
    "TrajectoryAnalyzer",
]
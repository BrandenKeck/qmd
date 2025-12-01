"""
Base quantum calculator interface for molecular dynamics.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple
from ase import Atoms


class QuantumCalculator(ABC):
    """
    Abstract base class for quantum chemistry calculators.

    This provides a standard interface for different quantum chemistry
    backends (PySCF, ORCA, etc.) to compute energies and forces.
    """

    def __init__(self, method: str = "B3LYP", basis: str = "6-31G*", **kwargs):
        """
        Initialize quantum calculator.

        Args:
            method: Quantum chemistry method (e.g., 'B3LYP', 'PBE', 'HF')
            basis: Basis set (e.g., '6-31G*', 'cc-pVDZ')
            **kwargs: Additional calculator-specific options
        """
        self.method = method
        self.basis = basis
        self.options = kwargs
        self.energy: Optional[float] = None
        self.forces: Optional[np.ndarray] = None
        self.charges: Optional[np.ndarray] = None
        self.dipole: Optional[np.ndarray] = None

    @abstractmethod
    def calculate(self, atoms: Atoms) -> Dict[str, Any]:
        """
        Perform quantum chemistry calculation.

        Args:
            atoms: ASE Atoms object with current geometry

        Returns:
            Dictionary containing energy, forces, and other properties
        """
        pass

    @abstractmethod
    def get_energy(self, atoms: Atoms) -> float:
        """
        Get total electronic energy.

        Args:
            atoms: ASE Atoms object

        Returns:
            Total energy in eV
        """
        pass

    @abstractmethod
    def get_forces(self, atoms: Atoms) -> np.ndarray:
        """
        Get forces on atoms.

        Args:
            atoms: ASE Atoms object

        Returns:
            Forces array of shape (n_atoms, 3) in eV/Angstrom
        """
        pass

    def get_charges(self, atoms: Atoms) -> Optional[np.ndarray]:
        """
        Get atomic charges (if available).

        Args:
            atoms: ASE Atoms object

        Returns:
            Atomic charges array or None if not available
        """
        return self.charges

    def get_dipole(self, atoms: Atoms) -> Optional[np.ndarray]:
        """
        Get electric dipole moment (if available).

        Args:
            atoms: ASE Atoms object

        Returns:
            Dipole moment vector or None if not available
        """
        return self.dipole

    def reset(self) -> None:
        """Reset calculator state."""
        self.energy = None
        self.forces = None
        self.charges = None
        self.dipole = None

    def set_method(self, method: str) -> None:
        """Change quantum chemistry method."""
        self.method = method
        self.reset()

    def set_basis(self, basis: str) -> None:
        """Change basis set."""
        self.basis = basis
        self.reset()

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(method={self.method}, basis={self.basis})"
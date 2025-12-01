"""
Atom and Molecule classes for quantum molecular dynamics.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Atom:
    """
    Individual atom representation.
    """
    name: str
    element: str
    position: np.ndarray
    mass: float
    index: int
    charge: float = 0.0
    velocity: Optional[np.ndarray] = None
    force: Optional[np.ndarray] = None

    def __post_init__(self):
        """Initialize velocity and force if not provided."""
        if self.velocity is None:
            self.velocity = np.zeros(3)
        if self.force is None:
            self.force = np.zeros(3)

    @property
    def kinetic_energy(self) -> float:
        """Calculate kinetic energy of this atom."""
        if self.velocity is None:
            return 0.0
        # Convert to atomic units (Hartree)
        conversion = 1.036427e-4  # amu * (Angstrom/fs)^2 to Hartree
        return 0.5 * self.mass * np.sum(self.velocity**2) * conversion

    def distance_to(self, other: 'Atom') -> float:
        """Calculate distance to another atom."""
        return np.linalg.norm(self.position - other.position)


class Molecule:
    """
    Molecule (residue) representation.
    """

    def __init__(self, name: str, resid: int, atoms: List[Atom]):
        """
        Initialize molecule.

        Args:
            name: Residue name (e.g., 'ALA', 'GLY')
            resid: Residue ID number
            atoms: List of atoms in this molecule
        """
        self.name = name
        self.resid = resid
        self.atoms = atoms
        self._atom_dict = {atom.name: atom for atom in atoms}

    def get_atom(self, name: str) -> Optional[Atom]:
        """Get atom by name."""
        return self._atom_dict.get(name)

    def get_backbone_atoms(self) -> List[Atom]:
        """Get backbone atoms (N, CA, C, O)."""
        backbone_names = ['N', 'CA', 'C', 'O']
        return [self.get_atom(name) for name in backbone_names if self.get_atom(name)]

    def get_sidechain_atoms(self) -> List[Atom]:
        """Get sidechain atoms (everything except backbone)."""
        backbone_names = {'N', 'CA', 'C', 'O', 'H', 'HA'}
        return [atom for atom in self.atoms if atom.name not in backbone_names]

    @property
    def center_of_mass(self) -> np.ndarray:
        """Calculate center of mass of the molecule."""
        total_mass = sum(atom.mass for atom in self.atoms)
        if total_mass == 0:
            return np.zeros(3)

        com = np.zeros(3)
        for atom in self.atoms:
            com += atom.mass * atom.position
        return com / total_mass

    @property
    def total_mass(self) -> float:
        """Total mass of the molecule."""
        return sum(atom.mass for atom in self.atoms)

    @property
    def total_charge(self) -> float:
        """Total charge of the molecule."""
        return sum(atom.charge for atom in self.atoms)

    def __len__(self) -> int:
        """Number of atoms in molecule."""
        return len(self.atoms)

    def __repr__(self) -> str:
        """String representation."""
        return f"Molecule({self.name}-{self.resid}, {len(self.atoms)} atoms)"
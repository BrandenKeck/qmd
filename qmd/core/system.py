"""
Core System class for quantum molecular dynamics simulations.
"""

import numpy as np
from typing import List, Optional, Dict, Any
import MDAnalysis as mda
from ase import Atoms
from Bio.PDB import PDBParser
import warnings

from .atoms import Atom, Molecule


class System:
    """
    Main system class for quantum molecular dynamics simulations.

    This class handles protein structures, coordinates, velocities, and
    interfaces with quantum chemistry calculators.
    """

    def __init__(self, pdb_file: Optional[str] = None, atoms: Optional[Atoms] = None):
        """
        Initialize the system.

        Args:
            pdb_file: Path to PDB file containing protein structure
            atoms: ASE Atoms object (alternative to PDB file)
        """
        self.atoms: Optional[Atoms] = None
        self.universe: Optional[mda.Universe] = None
        self.molecules: List[Molecule] = []
        self.positions: Optional[np.ndarray] = None
        self.velocities: Optional[np.ndarray] = None
        self.forces: Optional[np.ndarray] = None
        self.masses: Optional[np.ndarray] = None
        self.charges: Optional[np.ndarray] = None
        self.energy: float = 0.0
        self.temperature: float = 298.15  # Kelvin
        self.timestep: float = 0.5  # femtoseconds

        if pdb_file:
            self.load_pdb(pdb_file)
        elif atoms:
            self.load_atoms(atoms)

    def load_pdb(self, pdb_file: str) -> None:
        """
        Load protein structure from PDB file.

        Args:
            pdb_file: Path to PDB file
        """
        try:
            # Load with MDAnalysis for comprehensive analysis
            self.universe = mda.Universe(pdb_file)

            # Convert to ASE Atoms object for quantum calculations
            positions = self.universe.atoms.positions
            symbols = [atom.element for atom in self.universe.atoms]

            # Handle cases where element is not specified
            symbols = [sym if sym else self._guess_element(atom.name)
                      for sym, atom in zip(symbols, self.universe.atoms)]

            self.atoms = Atoms(symbols=symbols, positions=positions)

            # Extract additional information
            self.positions = positions.copy()
            self.masses = np.array([atom.mass for atom in self.universe.atoms])

            # Initialize velocities (Maxwell-Boltzmann distribution)
            self._initialize_velocities()

            # Parse molecules (residues)
            self._parse_molecules()

            print(f"Loaded protein with {len(self.atoms)} atoms")
            print(f"Found {len(self.molecules)} residues")

        except Exception as e:
            raise RuntimeError(f"Failed to load PDB file {pdb_file}: {e}")

    def load_atoms(self, atoms: Atoms) -> None:
        """
        Load system from ASE Atoms object.

        Args:
            atoms: ASE Atoms object
        """
        self.atoms = atoms.copy()
        self.positions = atoms.positions.copy()
        self.masses = atoms.get_masses()
        self._initialize_velocities()

    def _guess_element(self, atom_name: str) -> str:
        """Guess element from atom name."""
        name = atom_name.strip()
        if name.startswith('C'):
            return 'C'
        elif name.startswith('N'):
            return 'N'
        elif name.startswith('O'):
            return 'O'
        elif name.startswith('S'):
            return 'S'
        elif name.startswith('H'):
            return 'H'
        elif name.startswith('P'):
            return 'P'
        else:
            warnings.warn(f"Unknown atom type {name}, assuming carbon")
            return 'C'

    def _initialize_velocities(self) -> None:
        """Initialize velocities from Maxwell-Boltzmann distribution."""
        if self.masses is None:
            self.masses = self.atoms.get_masses()

        # Boltzmann constant in atomic units
        kb = 8.617333e-5  # eV/K

        n_atoms = len(self.atoms)
        self.velocities = np.zeros((n_atoms, 3))

        for i in range(n_atoms):
            # Standard deviation for Maxwell-Boltzmann distribution
            sigma = np.sqrt(kb * self.temperature / self.masses[i])
            self.velocities[i] = np.random.normal(0, sigma, 3)

        # Remove center of mass motion
        self._remove_com_motion()

    def _remove_com_motion(self) -> None:
        """Remove center of mass motion."""
        if self.velocities is not None and self.masses is not None:
            total_mass = np.sum(self.masses)
            com_velocity = np.sum(self.masses[:, np.newaxis] * self.velocities, axis=0) / total_mass
            self.velocities -= com_velocity

    def _parse_molecules(self) -> None:
        """Parse molecules (residues) from the universe."""
        if self.universe is None:
            return

        self.molecules = []
        for residue in self.universe.residues:
            molecule = Molecule(
                name=residue.resname,
                resid=residue.resid,
                atoms=[Atom(
                    name=atom.name,
                    element=atom.element or self._guess_element(atom.name),
                    position=atom.position,
                    mass=atom.mass,
                    index=atom.index
                ) for atom in residue.atoms]
            )
            self.molecules.append(molecule)

    def get_kinetic_energy(self) -> float:
        """Calculate total kinetic energy."""
        if self.velocities is None or self.masses is None:
            return 0.0

        # Convert to atomic units (Hartree)
        # 1 amu * (Angstrom/fs)^2 = 1.036427e-4 Hartree
        conversion = 1.036427e-4

        ke = 0.5 * np.sum(self.masses[:, np.newaxis] * self.velocities**2) * conversion
        return ke

    def get_temperature(self) -> float:
        """Calculate instantaneous temperature from kinetic energy."""
        ke = self.get_kinetic_energy()
        # 3/2 * N * kb * T = KE
        kb_hartree = 3.166811e-6  # Hartree/K
        n_dof = 3 * len(self.atoms) - 6  # Remove translational and rotational DOF
        if n_dof <= 0:
            return 0.0
        return 2 * ke / (n_dof * kb_hartree)

    def update_positions(self, new_positions: np.ndarray) -> None:
        """Update atomic positions."""
        self.positions = new_positions.copy()
        if self.atoms is not None:
            self.atoms.positions = new_positions

    def update_velocities(self, new_velocities: np.ndarray) -> None:
        """Update atomic velocities."""
        self.velocities = new_velocities.copy()

    def update_forces(self, new_forces: np.ndarray) -> None:
        """Update forces on atoms."""
        self.forces = new_forces.copy()

    def get_center_of_mass(self) -> np.ndarray:
        """Calculate center of mass."""
        if self.positions is None or self.masses is None:
            raise ValueError("Positions and masses must be set")

        total_mass = np.sum(self.masses)
        com = np.sum(self.masses[:, np.newaxis] * self.positions, axis=0) / total_mass
        return com

    def apply_pbc(self, box_size: Optional[np.ndarray] = None) -> None:
        """Apply periodic boundary conditions (if needed)."""
        if box_size is not None and self.positions is not None:
            self.positions = self.positions % box_size
            if self.atoms is not None:
                self.atoms.positions = self.positions

    def write_pdb(self, filename: str) -> None:
        """Write current structure to PDB file."""
        if self.universe is not None:
            self.universe.atoms.positions = self.positions
            self.universe.atoms.write(filename)
        else:
            raise ValueError("No universe loaded to write PDB")

    def __len__(self) -> int:
        """Return number of atoms."""
        return len(self.atoms) if self.atoms else 0

    def __repr__(self) -> str:
        """String representation."""
        n_atoms = len(self) if self.atoms else 0
        n_molecules = len(self.molecules)
        return f"System(atoms={n_atoms}, molecules={n_molecules}, T={self.temperature:.1f}K)"
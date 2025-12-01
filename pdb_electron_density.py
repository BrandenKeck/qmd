#!/usr/bin/env python3
"""
PDB Electron Density Calculator with GPU Acceleration

This script downloads a PDB file and calculates electron density maps
using GPU acceleration with CuPy.

Requirements:
    pip install biopython cupy matplotlib numpy requests
"""

import requests
import numpy as np
try:
    import cupy as cp
    # Test if CuPy actually works by trying a simple operation
    test_array = cp.array([1, 2, 3])
    _ = cp.sum(test_array)  # This will fail if CUDA runtime is not available
    GPU_AVAILABLE = True
    print("CuPy available - GPU acceleration enabled")
except (ImportError, RuntimeError) as e:
    print(f"CuPy not available or CUDA runtime error, falling back to CPU computation: {str(e)[:100]}...")
    import numpy as cp
    GPU_AVAILABLE = False

from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import tempfile
from typing import Tuple, List, Dict


class GPUElectronDensityCalculator:
    """GPU-accelerated electron density calculator for PDB structures."""

    def __init__(self, grid_spacing: float = 0.5, grid_size: Tuple[int, int, int] = (100, 100, 100)):
        """
        Initialize the calculator.

        Args:
            grid_spacing: Distance between grid points in Angstroms
            grid_size: Number of grid points in each dimension (x, y, z)
        """
        self.grid_spacing = grid_spacing
        self.grid_size = grid_size

        # Atomic form factors for common elements (simplified Gaussian approximation)
        # These are coefficients for the form factor equation: sum(a_i * exp(-b_i * s^2))
        self.form_factors = {
            'C': {'a': [2.31, 1.02, 1.59, 0.87], 'b': [20.8, 10.2, 0.57, 51.7]},
            'N': {'a': [12.2, 3.13, 2.01, 1.17], 'b': [0.006, 9.89, 28.9, 0.58]},
            'O': {'a': [3.05, 2.29, 1.55, 0.87], 'b': [13.3, 5.70, 0.32, 32.9]},
            'S': {'a': [6.91, 5.20, 1.44, 1.59], 'b': [1.47, 22.2, 0.25, 56.2]},
            'P': {'a': [6.43, 4.18, 1.78, 1.49], 'b': [1.91, 27.2, 0.53, 68.2]},
            'H': {'a': [0.49, 0.32, 0.14, 0.04], 'b': [10.5, 26.1, 3.14, 57.8]}
        }

    def download_pdb(self, pdb_id: str, output_dir: str = None) -> str:
        """
        Download a PDB file from the Protein Data Bank.

        Args:
            pdb_id: 4-character PDB identifier
            output_dir: Directory to save the file (uses temp dir if None)

        Returns:
            Path to the downloaded PDB file
        """
        pdb_id = pdb_id.lower()
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

        if output_dir is None:
            output_dir = tempfile.gettempdir()

        pdb_path = os.path.join(output_dir, f"{pdb_id}.pdb")

        print(f"Downloading PDB file {pdb_id} from RCSB...")
        response = requests.get(url)
        response.raise_for_status()

        with open(pdb_path, 'w') as f:
            f.write(response.text)

        print(f"Downloaded to: {pdb_path}")
        return pdb_path

    def parse_pdb(self, pdb_path: str) -> List[Dict]:
        """
        Parse PDB file and extract atomic coordinates and properties.

        Args:
            pdb_path: Path to PDB file

        Returns:
            List of atom dictionaries with coordinates and properties
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)

        atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        coord = atom.get_coord()
                        element = atom.element.strip()
                        if not element:  # Try to guess from atom name
                            element = atom.name[0]

                        atoms.append({
                            'coord': coord,
                            'element': element,
                            'bfactor': atom.get_bfactor(),
                            'occupancy': atom.get_occupancy()
                        })

        print(f"Parsed {len(atoms)} atoms from PDB file")
        return atoms

    def create_grid(self, atoms: List[Dict]) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Create 3D grid for electron density calculation.

        Args:
            atoms: List of atom dictionaries

        Returns:
            Tuple of (X, Y, Z) grid coordinate arrays
        """
        # Find bounding box of all atoms
        coords = cp.array([atom['coord'] for atom in atoms])
        min_coords = cp.min(coords, axis=0)
        max_coords = cp.max(coords, axis=0)

        # Extend bounding box slightly
        margin = 5.0  # Angstroms
        min_coords -= margin
        max_coords += margin

        # Create grid
        x = cp.linspace(min_coords[0], max_coords[0], self.grid_size[0])
        y = cp.linspace(min_coords[1], max_coords[1], self.grid_size[1])
        z = cp.linspace(min_coords[2], max_coords[2], self.grid_size[2])

        X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

        return X, Y, Z

    def calculate_electron_density(self, atoms: List[Dict]) -> cp.ndarray:
        """
        Calculate electron density map using GPU acceleration.

        Args:
            atoms: List of atom dictionaries

        Returns:
            3D electron density array
        """
        X, Y, Z = self.create_grid(atoms)
        density = cp.zeros(self.grid_size, dtype=cp.float32)

        print("Calculating electron density on GPU...")

        # Process atoms in batches to manage memory
        batch_size = 100
        for i in range(0, len(atoms), batch_size):
            batch_atoms = atoms[i:i + batch_size]

            for atom in batch_atoms:
                coord = cp.array(atom['coord'])
                element = atom['element']
                bfactor = atom['bfactor']
                occupancy = atom['occupancy']

                # Calculate distances from atom to all grid points
                dx = X - coord[0]
                dy = Y - coord[1]
                dz = Z - coord[2]
                r_squared = dx**2 + dy**2 + dz**2

                # Get form factor coefficients
                if element in self.form_factors:
                    ff = self.form_factors[element]
                else:
                    # Default to carbon if element not found
                    ff = self.form_factors['C']

                # Calculate atomic form factor contribution
                # Using simplified Gaussian approximation: f = sum(a_i * exp(-b_i * r^2))
                atom_density = cp.zeros_like(r_squared)
                for a, b in zip(ff['a'], ff['b']):
                    atom_density += a * cp.exp(-b * r_squared / (4 * cp.pi**2))

                # Apply temperature factor (B-factor) and occupancy
                temp_factor = cp.exp(-bfactor * r_squared / (8 * cp.pi**2))
                atom_density *= temp_factor * occupancy

                density += atom_density

        print("Electron density calculation complete!")
        return density

    def visualize_density(self, density: cp.ndarray, slice_index: int = None):
        """
        Visualize electron density as 2D slices.

        Args:
            density: 3D electron density array
            slice_index: Z-slice to visualize (middle slice if None)
        """
        if GPU_AVAILABLE:
            density_cpu = cp.asnumpy(density)
        else:
            density_cpu = density

        if slice_index is None:
            slice_index = density_cpu.shape[2] // 2

        plt.figure(figsize=(12, 5))

        # XY slice
        plt.subplot(1, 3, 1)
        plt.imshow(density_cpu[:, :, slice_index], cmap='viridis', origin='lower')
        plt.title(f'XY slice (Z={slice_index})')
        plt.colorbar(label='Electron Density')

        # XZ slice
        plt.subplot(1, 3, 2)
        plt.imshow(density_cpu[:, density_cpu.shape[1]//2, :], cmap='viridis', origin='lower')
        plt.title('XZ slice (middle Y)')
        plt.colorbar(label='Electron Density')

        # YZ slice
        plt.subplot(1, 3, 3)
        plt.imshow(density_cpu[density_cpu.shape[0]//2, :, :], cmap='viridis', origin='lower')
        plt.title('YZ slice (middle X)')
        plt.colorbar(label='Electron Density')

        plt.tight_layout()
        plt.show()

    def save_density_map(self, density: cp.ndarray, filename: str):
        """
        Save electron density map to a file.

        Args:
            density: 3D electron density array
            filename: Output filename
        """
        if GPU_AVAILABLE:
            density_cpu = cp.asnumpy(density)
        else:
            density_cpu = density

        np.save(filename, density_cpu)
        print(f"Density map saved to: {filename}")


def main():
    """Main function to demonstrate the electron density calculator."""

    # Example usage
    calculator = GPUElectronDensityCalculator(
        grid_spacing=0.3,  # Angstroms
        grid_size=(80, 80, 80)  # Grid points
    )

    # Download a small protein structure (1CRN - crambin, 46 residues)
    pdb_id = "1crn"
    pdb_path = calculator.download_pdb(pdb_id)

    # Parse the PDB file
    atoms = calculator.parse_pdb(pdb_path)

    # Calculate electron density
    print(f"Using {'GPU' if GPU_AVAILABLE else 'CPU'} for calculations")
    density = calculator.calculate_electron_density(atoms)

    # Visualize the results
    calculator.visualize_density(density)

    # Save the density map
    output_file = f"{pdb_id}_density_map.npy"
    calculator.save_density_map(density, output_file)

    # Print some statistics
    if GPU_AVAILABLE:
        density_cpu = cp.asnumpy(density)
    else:
        density_cpu = density

    print(f"\nElectron Density Statistics:")
    print(f"Min density: {np.min(density_cpu):.4f}")
    print(f"Max density: {np.max(density_cpu):.4f}")
    print(f"Mean density: {np.mean(density_cpu):.4f}")
    print(f"Grid shape: {density_cpu.shape}")


if __name__ == "__main__":
    main()
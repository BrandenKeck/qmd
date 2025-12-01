#!/usr/bin/env python3
"""
PDB Electron Density Calculator (CPU Version)

This script downloads a PDB file and calculates electron density maps
using CPU computation with NumPy.

Requirements:
    pip install biopython matplotlib numpy requests
"""

import requests
import numpy as np
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import tempfile
from typing import Tuple, List, Dict


class CPUElectronDensityCalculator:
    """CPU-based electron density calculator for PDB structures."""

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

    def create_grid(self, atoms: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create 3D grid for electron density calculation.

        Args:
            atoms: List of atom dictionaries

        Returns:
            Tuple of (X, Y, Z) grid coordinate arrays
        """
        # Find bounding box of all atoms
        coords = np.array([atom['coord'] for atom in atoms])
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)

        # Extend bounding box slightly
        margin = 5.0  # Angstroms
        min_coords -= margin
        max_coords += margin

        # Create grid
        x = np.linspace(min_coords[0], max_coords[0], self.grid_size[0])
        y = np.linspace(min_coords[1], max_coords[1], self.grid_size[1])
        z = np.linspace(min_coords[2], max_coords[2], self.grid_size[2])

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        return X, Y, Z

    def calculate_electron_density(self, atoms: List[Dict]) -> np.ndarray:
        """
        Calculate electron density map using CPU computation.

        Args:
            atoms: List of atom dictionaries

        Returns:
            3D electron density array
        """
        X, Y, Z = self.create_grid(atoms)
        density = np.zeros(self.grid_size, dtype=np.float32)

        print("Calculating electron density on CPU...")
        print(f"Processing {len(atoms)} atoms on grid {self.grid_size}")

        # Process atoms in batches to manage memory
        batch_size = 50  # Smaller batches for CPU
        for batch_idx in range(0, len(atoms), batch_size):
            batch_atoms = atoms[batch_idx:batch_idx + batch_size]

            if batch_idx % 100 == 0:
                print(f"  Processing atoms {batch_idx + 1}-{min(batch_idx + batch_size, len(atoms))}...")

            for atom in batch_atoms:
                coord = np.array(atom['coord'])
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
                atom_density = np.zeros_like(r_squared)
                for a, b in zip(ff['a'], ff['b']):
                    atom_density += a * np.exp(-b * r_squared / (4 * np.pi**2))

                # Apply temperature factor (B-factor) and occupancy
                temp_factor = np.exp(-bfactor * r_squared / (8 * np.pi**2))
                atom_density *= temp_factor * occupancy

                density += atom_density

        print("Electron density calculation complete!")
        return density

    def visualize_density(self, density: np.ndarray, slice_index: int = None, show_plots: bool = True):
        """
        Visualize electron density as 2D slices.

        Args:
            density: 3D electron density array
            slice_index: Z-slice to visualize (middle slice if None)
            show_plots: Whether to display plots (False for headless systems)
        """
        if slice_index is None:
            slice_index = density.shape[2] // 2

        plt.figure(figsize=(15, 5))

        # XY slice
        plt.subplot(1, 3, 1)
        plt.imshow(density[:, :, slice_index], cmap='viridis', origin='lower')
        plt.title(f'XY slice (Z={slice_index})')
        plt.colorbar(label='Electron Density')
        plt.xlabel('X (grid points)')
        plt.ylabel('Y (grid points)')

        # XZ slice
        plt.subplot(1, 3, 2)
        plt.imshow(density[:, density.shape[1]//2, :], cmap='viridis', origin='lower')
        plt.title('XZ slice (middle Y)')
        plt.colorbar(label='Electron Density')
        plt.xlabel('Z (grid points)')
        plt.ylabel('X (grid points)')

        # YZ slice
        plt.subplot(1, 3, 3)
        plt.imshow(density[density.shape[0]//2, :, :], cmap='viridis', origin='lower')
        plt.title('YZ slice (middle X)')
        plt.colorbar(label='Electron Density')
        plt.xlabel('Z (grid points)')
        plt.ylabel('Y (grid points)')

        plt.tight_layout()

        # Save the plot
        plot_filename = 'electron_density_visualization.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {plot_filename}")

        if show_plots:
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display plots (headless system?): {e}")

    def save_density_map(self, density: np.ndarray, filename: str):
        """
        Save electron density map to a file.

        Args:
            density: 3D electron density array
            filename: Output filename
        """
        np.save(filename, density)
        print(f"Density map saved to: {filename}")

    def analyze_density(self, density: np.ndarray) -> Dict:
        """
        Analyze electron density statistics.

        Args:
            density: 3D electron density array

        Returns:
            Dictionary with analysis results
        """
        stats = {
            'shape': density.shape,
            'min': np.min(density),
            'max': np.max(density),
            'mean': np.mean(density),
            'std': np.std(density),
            'total_electrons': np.sum(density) * (self.grid_spacing**3),
            'nonzero_points': np.count_nonzero(density)
        }

        # Find high-density regions
        threshold = stats['mean'] + 2 * stats['std']
        high_density_mask = density > threshold
        stats['high_density_points'] = np.sum(high_density_mask)
        stats['high_density_threshold'] = threshold

        return stats


def main():
    """Main function to demonstrate the electron density calculator."""
    print("=== CPU-Based PDB Electron Density Calculator ===")

    # Example usage with a small protein
    calculator = CPUElectronDensityCalculator(
        grid_spacing=0.4,     # 0.4 Angstrom spacing
        grid_size=(60, 60, 60)  # 60x60x60 grid (reasonable for CPU)
    )

    # Download a small protein structure (1CRN - crambin, 46 residues)
    pdb_id = "1crn"
    pdb_path = calculator.download_pdb(pdb_id)

    # Parse the PDB file
    atoms = calculator.parse_pdb(pdb_path)

    # Calculate electron density
    print(f"Using CPU for calculations")
    density = calculator.calculate_electron_density(atoms)

    # Analyze the results
    stats = calculator.analyze_density(density)
    print(f"\nElectron Density Analysis:")
    print(f"Grid shape: {stats['shape']}")
    print(f"Min density: {stats['min']:.4f}")
    print(f"Max density: {stats['max']:.4f}")
    print(f"Mean density: {stats['mean']:.4f}")
    print(f"Std density: {stats['std']:.4f}")
    print(f"Total electrons (approx): {stats['total_electrons']:.1f}")
    print(f"Non-zero grid points: {stats['nonzero_points']:,} / {np.prod(stats['shape']):,}")
    print(f"High-density points (>{stats['high_density_threshold']:.4f}): {stats['high_density_points']:,}")

    # Visualize the results
    calculator.visualize_density(density, show_plots=False)  # Don't try to show in headless

    # Save the density map
    output_file = f"{pdb_id}_density_map_cpu.npy"
    calculator.save_density_map(density, output_file)

    print(f"\nCompleted processing {pdb_id} using CPU computation!")
    print("Files generated:")
    print(f"  - {output_file} (3D density data)")
    print(f"  - electron_density_visualization.png (visualization)")


if __name__ == "__main__":
    main()
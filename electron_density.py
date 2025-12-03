"""
Electron Density Calculation Engine for Quantum MD Simulations
"""

import numpy as np
import cupy as cp  # GPU acceleration
from scipy.spatial.distance import cdist
from typing import Dict, Tuple, Optional, Union, Any
import h5py
import argparse
import yaml


class ElectronDensityCalculator:
    """Calculate electron density from molecular structures."""

    def __init__(self, use_gpu: bool = True, config: Dict[str, Any] = None):
        if config is None:
            config = self.load_config()
        self.config = config
        self.use_gpu = use_gpu and cp.cuda.is_available()
        self.xp = cp if self.use_gpu else np

        # Atomic properties
        self.atomic_numbers = {
            'H': 1, 'C': 6, 'N': 7, 'O': 8, 'P': 15, 'S': 16
        }

        # Slater-type orbital parameters (simplified)
        self.sto_exponents = {
            1: 1.24,   # H
            6: 1.625,  # C
            7: 1.925,  # N
            8: 2.275,  # O
            15: 1.827, # P
            16: 1.827  # S
        }

    @staticmethod
    def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    def load_structure_coords(self, mmcif_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load atomic coordinates and numbers from mmCIF file."""
        import gemmi

        structure = gemmi.read_structure(mmcif_file)
        coords = []
        atomic_nums = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                        element = atom.element.name
                        atomic_nums.append(self.atomic_numbers.get(element, 6))

        return np.array(coords), np.array(atomic_nums)

    def gaussian_density(self, coords: Union[np.ndarray, cp.ndarray],
                        atomic_nums: Union[np.ndarray, cp.ndarray],
                        grid_points: Union[np.ndarray, cp.ndarray],
                        sigma: float = None) -> Union[np.ndarray, cp.ndarray]:
        """Calculate electron density using Gaussian approximation."""

        if sigma is None:
            sigma = self.config.get('gaussian_sigma', 1.0)

        if self.use_gpu:
            coords = cp.asarray(coords)
            atomic_nums = cp.asarray(atomic_nums)
            grid_points = cp.asarray(grid_points)

        density = self.xp.zeros(len(grid_points))

        for i, (coord, z) in enumerate(zip(coords, atomic_nums)):
            # Distance from each grid point to atom
            distances = self.xp.linalg.norm(grid_points - coord, axis=1)

            # Gaussian approximation of electron density
            atomic_density = z * self.xp.exp(-distances**2 / (2 * sigma**2))
            density += atomic_density

        return density

    def sto_density(self, coords: Union[np.ndarray, cp.ndarray],
                   atomic_nums: Union[np.ndarray, cp.ndarray],
                   grid_points: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        """Calculate electron density using Slater-type orbitals."""

        if self.use_gpu:
            coords = cp.asarray(coords)
            atomic_nums = cp.asarray(atomic_nums)
            grid_points = cp.asarray(grid_points)

        density = self.xp.zeros(len(grid_points))

        for coord, z in zip(coords, atomic_nums):
            distances = self.xp.linalg.norm(grid_points - coord, axis=1)

            # STO approximation
            zeta = self.sto_exponents.get(int(z), 1.5)
            sto_density = (zeta**3 / self.xp.pi) * self.xp.exp(-2 * zeta * distances)
            density += z * sto_density

        return density

    def create_grid(self, coords: np.ndarray,
                   spacing: float = None,
                   padding: float = None) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """Create 3D grid for density calculation."""

        if spacing is None:
            spacing = self.config.get('grid_spacing', 0.5)
        if padding is None:
            padding = self.config.get('grid_padding', 5.0)

        # Determine grid bounds
        min_coords = coords.min(axis=0) - padding
        max_coords = coords.max(axis=0) + padding

        # Grid dimensions
        nx = int((max_coords[0] - min_coords[0]) / spacing) + 1
        ny = int((max_coords[1] - min_coords[1]) / spacing) + 1
        nz = int((max_coords[2] - min_coords[2]) / spacing) + 1

        # Create grid
        x = np.linspace(min_coords[0], max_coords[0], nx)
        y = np.linspace(min_coords[1], max_coords[1], ny)
        z = np.linspace(min_coords[2], max_coords[2], nz)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        return grid_points, (nx, ny, nz)

    def calculate_density(self, mmcif_file: str,
                         method: str = None,
                         spacing: float = None,
                         output_file: str = None) -> str:
        """Main density calculation function."""

        if method is None:
            method = self.config.get('density_method', 'gaussian')
        if spacing is None:
            spacing = self.config.get('grid_spacing', 0.5)
        if output_file is None:
            output_file = self.config.get('density_h5_filename', 'density.h5')

        # Load structure
        coords, atomic_nums = self.load_structure_coords(mmcif_file)

        # Create grid
        grid_points, grid_shape = self.create_grid(coords, spacing)

        # Calculate density
        if method == 'gaussian':
            density = self.gaussian_density(coords, atomic_nums, grid_points)
        elif method == 'sto':
            density = self.sto_density(coords, atomic_nums, grid_points)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Convert back to numpy if using GPU
        if self.use_gpu:
            density = cp.asnumpy(density)

        # Reshape to 3D grid
        density_3d = density.reshape(grid_shape)

        # Save to HDF5
        self.save_density(density_3d, grid_points, grid_shape, output_file)

        return output_file

    def save_density(self, density_3d: np.ndarray,
                    grid_points: np.ndarray,
                    grid_shape: Tuple[int, int, int],
                    output_file: str):
        """Save density data to HDF5 format."""

        with h5py.File(output_file, 'w') as f:
            f.create_dataset('density', data=density_3d)
            f.create_dataset('grid_points', data=grid_points)
            f.attrs['grid_shape'] = grid_shape
            f.attrs['total_electrons'] = float(np.sum(density_3d))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate electron density from mmCIF structure')
    parser.add_argument('--input', '-i', required=True, help='Input mmCIF file')
    parser.add_argument('--method', '-m', choices=['gaussian', 'sto'], default='gaussian',
                       help='Density calculation method')
    parser.add_argument('--spacing', '-s', type=float, default=0.5, help='Grid spacing (Angstrom)')
    parser.add_argument('--output', '-o', default='density.h5', help='Output HDF5 file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')

    args = parser.parse_args()

    calculator = ElectronDensityCalculator(use_gpu=args.gpu)
    result = calculator.calculate_density(args.input, args.method, args.spacing, args.output)
    print(f"Density saved to: {result}")
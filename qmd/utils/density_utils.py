"""
Electron density analysis utilities for QMD framework with GPU acceleration.
"""

import numpy as np
from scipy.linalg import svd
from ..utils.gpu_utils import (GPUConfig, get_array_module, ensure_array,
                              to_cpu, to_gpu, memory_pool)

class MolecularAligner:
    """Kabsch algorithm for molecular alignment."""

    @staticmethod
    def kabsch_align(coords1, coords2, weights=None):
        """
        Align coords2 to coords1 using Kabsch algorithm.
        Handles different sized coordinate systems by finding best matching subset.

        Returns: aligned_coords2, rotation_matrix, rmsd
        """
        # Handle different sized coordinate systems
        if coords1.shape[0] != coords2.shape[0]:
            # Use smaller system as reference, find best subset in larger
            if coords1.shape[0] < coords2.shape[0]:
                # coords1 is smaller - find best matching subset in coords2
                ref_coords = coords1
                target_coords = coords2
                subset_indices = MolecularAligner._find_best_subset(ref_coords, target_coords)
                coords2_subset = target_coords[subset_indices]

                # Perform alignment with subset
                aligned_subset, R, rmsd = MolecularAligner._kabsch_core(ref_coords, coords2_subset, weights)

                # Apply transformation to all coords2
                centroid2 = np.mean(target_coords, axis=0)
                coords2_centered = target_coords - centroid2
                centroid1 = np.mean(ref_coords, axis=0)
                aligned_coords2 = (coords2_centered @ R.T) + centroid1

            else:
                # coords2 is smaller - find best matching subset in coords1
                ref_coords = coords2
                target_coords = coords1
                subset_indices = MolecularAligner._find_best_subset(ref_coords, target_coords)
                coords1_subset = target_coords[subset_indices]

                # Perform alignment with subset
                aligned_coords2, R, rmsd = MolecularAligner._kabsch_core(coords1_subset, ref_coords, weights)

        else:
            # Same size - use original algorithm
            aligned_coords2, R, rmsd = MolecularAligner._kabsch_core(coords1, coords2, weights)

        return aligned_coords2, R, rmsd

    @staticmethod
    def _find_best_subset(ref_coords, target_coords):
        """
        Find best matching subset of target_coords that matches ref_coords size.
        Uses distance-based matching from centroids.
        """
        ref_centroid = np.mean(ref_coords, axis=0)
        target_centroid = np.mean(target_coords, axis=0)

        # Calculate distances from target centroid
        distances = np.linalg.norm(target_coords - target_centroid, axis=1)

        # Select closest N points where N = len(ref_coords)
        n_points = len(ref_coords)
        closest_indices = np.argsort(distances)[:n_points]

        return closest_indices

    @staticmethod
    def _kabsch_core(coords1, coords2, weights=None):
        """
        Core Kabsch algorithm implementation with GPU acceleration for same-sized coordinate sets.
        """
        # Get appropriate array module (numpy or cupy)
        xp = get_array_module()

        # Convert inputs to appropriate arrays
        coords1 = ensure_array(coords1)
        coords2 = ensure_array(coords2)

        if weights is None:
            weights = xp.ones(len(coords1))
        else:
            weights = ensure_array(weights)
        weights = weights / xp.sum(weights)

        # Center coordinates
        centroid1 = xp.average(coords1, axis=0, weights=weights)
        centroid2 = xp.average(coords2, axis=0, weights=weights)

        coords1_centered = coords1 - centroid1
        coords2_centered = coords2 - centroid2

        # Apply weights
        coords1_weighted = coords1_centered * xp.sqrt(weights)[:, None]
        coords2_weighted = coords2_centered * xp.sqrt(weights)[:, None]

        # SVD for rotation matrix
        H = coords2_weighted.T @ coords1_weighted

        # Use appropriate SVD function
        if GPUConfig.is_enabled():
            try:
                import cupy.linalg as cp_linalg
                U, S, Vt = cp_linalg.svd(H)
            except (ImportError, AttributeError):
                # Fallback to CPU
                H_cpu = to_cpu(H)
                U, S, Vt = svd(H_cpu)
                U, Vt = ensure_array(U), ensure_array(Vt)
        else:
            U, S, Vt = svd(to_cpu(H))
            U, Vt = ensure_array(U), ensure_array(Vt)

        R = Vt.T @ U.T

        # Ensure proper rotation
        if float(xp.linalg.det(R)) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Apply transformation
        aligned_coords2 = (coords2_centered @ R.T) + centroid1

        # Calculate RMSD
        diff = aligned_coords2 - coords1
        rmsd = float(xp.sqrt(xp.average(xp.sum(diff**2, axis=1), weights=weights)))

        # Return as CPU arrays for compatibility
        return to_cpu(aligned_coords2), to_cpu(R), rmsd

class ElectronDensityCalculator:
    """Calculate electron density on 3D grids using PySCF with GPU acceleration."""

    def __init__(self, calculator, use_gpu: bool = None):
        """
        Initialize density calculator.

        Args:
            calculator: Quantum chemistry calculator
            use_gpu: Enable GPU acceleration (None = auto-detect)
        """
        self.calculator = calculator
        if use_gpu is None:
            self.use_gpu = GPUConfig.is_enabled()
        else:
            self.use_gpu = use_gpu and GPUConfig.is_available()

        if self.use_gpu:
            print("GPU acceleration enabled for density calculations")

    def calculate_density_grid(self, system, grid_points=30, margin=2.0, show_progress=True):
        """
        Calculate electron density on 3D grid with GPU acceleration.

        Returns: density_array, (x, y, z) grid coordinates
        """
        try:
            from tqdm import tqdm
        except ImportError:
            show_progress = False
        # Get appropriate array module
        xp = get_array_module()

        coords = system.atoms.get_positions()

        # Grid boundaries
        min_coords = np.min(coords, axis=0) - margin
        max_coords = np.max(coords, axis=0) + margin

        # Create grid
        x = np.linspace(min_coords[0], max_coords[0], grid_points)
        y = np.linspace(min_coords[1], max_coords[1], grid_points)
        z = np.linspace(min_coords[2], max_coords[2], grid_points)

        # Use GPU memory management
        with memory_pool:
            # Calculate total grid points for better progress tracking
            total_points = grid_points ** 3

            if show_progress:
                main_pbar = tqdm(total=100, desc="Density calculation", unit="%",
                               bar_format='{l_bar}{bar}| {n:.1f}/{total}% [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
                main_pbar.set_postfix({"stage": "grid setup", "points": f"{total_points:,}"})

            if self.use_gpu:
                # Convert grid arrays to GPU arrays first
                x_gpu = ensure_array(x)
                y_gpu = ensure_array(y)
                z_gpu = ensure_array(z)
                xx, yy, zz = xp.meshgrid(x_gpu, y_gpu, z_gpu, indexing='ij')
                grid_3d = xp.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
                grid_3d_bohr = grid_3d / 0.529177249  # Angstrom to Bohr
            else:
                xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
                grid_3d = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
                grid_3d_bohr = grid_3d / 0.529177249

            if show_progress:
                main_pbar.update(15)
                main_pbar.set_postfix({"stage": "quantum calculation", "atoms": len(system.atoms)})

            # Run calculation
            results = self.calculator.calculate(system.atoms)

            if show_progress:
                main_pbar.update(15)
                main_pbar.set_postfix({"stage": "density calculation", "grid": f"{grid_points}³"})

            # Get density from PySCF
            mol = self.calculator.mol
            mf = self.calculator.mf

            if not hasattr(mf, 'mo_coeff') or mf.mo_coeff is None:
                raise RuntimeError("MO coefficients not available")

            # Convert atomic coordinates to appropriate array type
            atomic_coords = ensure_array(system.atoms.get_positions() / 0.529177249)  # Bohr

            # GPU-accelerated density calculation with progress
            density_1d = self._calculate_density_vectorized(grid_3d_bohr, atomic_coords, xp,
                                                          show_progress=show_progress,
                                                          main_pbar=main_pbar if show_progress else None)

            if show_progress:
                main_pbar.update(10)
                main_pbar.set_postfix({"stage": "reshaping", "memory": "optimizing"})

            # Reshape to 3D grid
            if self.use_gpu:
                density = density_1d.reshape(grid_points, grid_points, grid_points)
                density = to_cpu(density)  # Convert back to CPU for compatibility
            else:
                density = density_1d.reshape(grid_points, grid_points, grid_points)

            if show_progress:
                main_pbar.update(10)
                main_pbar.set_postfix({"stage": "complete", "shape": str(density.shape)})
                main_pbar.close()

        return density, (x, y, z)

    def _calculate_density_vectorized(self, grid_points, atomic_coords, xp, show_progress=False, main_pbar=None):
        """
        Vectorized density calculation using GPU acceleration.

        Args:
            grid_points: Grid coordinates (N, 3)
            atomic_coords: Atomic positions (M, 3)
            xp: Array module (numpy or cupy)
            show_progress: Show progress bar for large calculations
            main_pbar: Main progress bar to update

        Returns:
            density: 1D density array (N,)
        """
        n_grid = len(grid_points)
        n_atoms = len(atomic_coords)

        # Use smaller chunk size for more frequent updates
        chunk_size = max(1000, n_grid // 100)  # Adaptive chunk size, minimum 1000, aim for ~100 chunks

        # Always show chunk progress for better feedback if we have multiple chunks
        if show_progress and n_grid > chunk_size:
            try:
                from tqdm import tqdm
                chunks = [grid_points[i:i+chunk_size] for i in range(0, n_grid, chunk_size)]
                density_chunks = []

                chunk_pbar = tqdm(total=len(chunks),
                                desc=f"Processing {n_grid:,} grid points",
                                unit="chunk",
                                leave=True,
                                position=1,
                                bar_format='{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

                for i, chunk in enumerate(chunks):
                    chunk_density = self._process_density_chunk(chunk, atomic_coords, xp)
                    density_chunks.append(chunk_density)

                    # Update chunk progress
                    progress_pct = (i + 1) / len(chunks) * 100
                    chunk_pbar.set_postfix({
                        "points": f"{(i+1)*chunk_size:,}",
                        "atoms": n_atoms,
                        "progress": f"{progress_pct:.1f}%"
                    })
                    chunk_pbar.update(1)

                    # Update main progress bar (50% of the total progress is for density calculation)
                    if main_pbar:
                        main_pbar.update(50 / len(chunks))

                chunk_pbar.close()
                return xp.concatenate(density_chunks)
            except ImportError:
                pass

        # Standard vectorized calculation for smaller grids or when no progress needed
        # Update main progress bar if available (for small grids processed in one go)
        if main_pbar:
            main_pbar.set_postfix({"stage": "computing density", "points": f"{n_grid:,}"})

        # Reshape for broadcasting: grid (N, 1, 3), atoms (1, M, 3)
        grid_expanded = grid_points[:, None, :]  # (N, 1, 3)
        atoms_expanded = atomic_coords[None, :, :]  # (1, M, 3)

        # Calculate squared distances: (N, M)
        dist_sq = xp.sum((grid_expanded - atoms_expanded)**2, axis=2)

        if main_pbar:
            main_pbar.update(25)
            main_pbar.set_postfix({"stage": "applying gaussians", "atoms": n_atoms})

        # Apply Gaussian kernel with different widths for different atom types
        # Simple approximation: use atomic number based scaling
        if hasattr(self.calculator, 'mol') and self.calculator.mol:
            # Get atomic charges for better approximation
            try:
                atomic_charges = np.array([self.calculator.mol.atom_charge(i)
                                         for i in range(self.calculator.mol.natm)])
                atomic_charges = ensure_array(atomic_charges)
                # Scale Gaussian width by atomic size (inverse of charge)
                width_scaling = 1.0 / xp.maximum(atomic_charges, 1.0)
                gaussian_widths = 2.0 * width_scaling[None, :]  # (1, M)
            except:
                # Fallback to uniform width
                gaussian_widths = 2.0
        else:
            gaussian_widths = 2.0

        # Calculate Gaussian contributions: (N, M)
        gaussian_contrib = xp.exp(-gaussian_widths * dist_sq)

        if main_pbar:
            main_pbar.update(25)
            main_pbar.set_postfix({"stage": "summing contributions", "grid_shape": f"{n_grid:,}x{n_atoms}"})

        # Sum over all atoms: (N,)
        density = xp.sum(gaussian_contrib, axis=1)

        return density

    def _process_density_chunk(self, chunk_points, atomic_coords, xp):
        """Process a chunk of grid points for density calculation."""
        # Reshape for broadcasting: grid (N, 1, 3), atoms (1, M, 3)
        grid_expanded = chunk_points[:, None, :]  # (N, 1, 3)
        atoms_expanded = atomic_coords[None, :, :]  # (1, M, 3)

        # Calculate squared distances: (N, M)
        dist_sq = xp.sum((grid_expanded - atoms_expanded)**2, axis=2)

        # Apply Gaussian kernel with different widths for different atom types
        if hasattr(self.calculator, 'mol') and self.calculator.mol:
            try:
                atomic_charges = np.array([self.calculator.mol.atom_charge(i)
                                         for i in range(self.calculator.mol.natm)])
                atomic_charges = ensure_array(atomic_charges)
                width_scaling = 1.0 / xp.maximum(atomic_charges, 1.0)
                gaussian_widths = 2.0 * width_scaling[None, :]  # (1, M)
            except:
                gaussian_widths = 2.0
        else:
            gaussian_widths = 2.0

        # Calculate Gaussian contributions: (N, M)
        gaussian_contrib = xp.exp(-gaussian_widths * dist_sq)

        # Sum over all atoms: (N,)
        return xp.sum(gaussian_contrib, axis=1)

    def calculate_density_difference(self, system1, system2, grid_points=30, margin=2.0):
        """
        Calculate electron density difference between two systems with GPU acceleration.

        Returns: density_diff, (x, y, z) grid coordinates
        """
        print("Calculating density difference with GPU acceleration...")

        # Calculate individual densities
        density1, grid_coords = self.calculate_density_grid(system1, grid_points, margin)
        density2, _ = self.calculate_density_grid(system2, grid_points, margin)

        # Calculate difference
        if self.use_gpu:
            xp = get_array_module()
            diff = ensure_array(density1) - ensure_array(density2)
            density_diff = to_cpu(diff)
        else:
            density_diff = density1 - density2

        return density_diff, grid_coords
"""
Molecular orbital visualization using PySCF data.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage import measure

class MolecularOrbitalVisualizer:
    """
    Visualize molecular orbitals calculated with PySCF.
    """

    def __init__(self, calculator):
        """
        Initialize with PySCF calculator.

        Parameters:
        -----------
        calculator : PySCFCalculator
            Configured PySCF calculator with completed calculation
        """
        self.calculator = calculator
        self.mol = None
        self.mf = None

    def calculate_orbital_grid(self, system, orbital_idx, grid_points=30, margin=2.0):
        """
        Calculate molecular orbital on 3D grid.

        Parameters:
        -----------
        system : System
            QMD system object
        orbital_idx : int
            Index of molecular orbital (0-based)
        grid_points : int
            Grid resolution
        margin : float
            Grid margin in Angstrom

        Returns:
        --------
        orbital_values : np.ndarray
            3D orbital values
        grid_coords : tuple
            (x, y, z) coordinate arrays
        orbital_info : dict
            Orbital energy and occupation
        """
        # Run calculation if needed
        energy, forces = self.calculator.calculate(system)

        self.mol = self.calculator.mol
        self.mf = self.calculator.mf

        if not hasattr(self.mf, 'mo_coeff') or self.mf.mo_coeff is None:
            raise RuntimeError("No molecular orbitals available")

        if orbital_idx >= self.mf.mo_coeff.shape[1]:
            raise ValueError(f"Orbital index {orbital_idx} out of range")

        # Create grid
        coords = system.atoms.get_positions()
        min_coords = np.min(coords, axis=0) - margin
        max_coords = np.max(coords, axis=0) + margin

        x = np.linspace(min_coords[0], max_coords[0], grid_points)
        y = np.linspace(min_coords[1], max_coords[1], grid_points)
        z = np.linspace(min_coords[2], max_coords[2], grid_points)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid_3d = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        grid_3d_bohr = grid_3d / 0.529177249  # Convert to Bohr

        # Evaluate molecular orbital
        from pyscf import dft
        ao_values = dft.numint.eval_ao(self.mol, grid_3d_bohr.T)
        mo_values = ao_values @ self.mf.mo_coeff[:, orbital_idx]

        # Reshape to 3D
        orbital_values = mo_values.reshape(grid_points, grid_points, grid_points)

        # Orbital information
        orbital_info = {
            'energy': self.mf.mo_energy[orbital_idx],
            'occupation': self.mf.mo_occ[orbital_idx],
            'index': orbital_idx
        }

        return orbital_values, (x, y, z), orbital_info

    def plot_orbital_isosurface(self, orbital_values, grid_coords, orbital_info,
                               isovalue=None, show_both_phases=True):
        """
        Plot molecular orbital isosurface with matplotlib.

        Parameters:
        -----------
        orbital_values : np.ndarray
            3D orbital values
        grid_coords : tuple
            (x, y, z) coordinate arrays
        orbital_info : dict
            Orbital information
        isovalue : float, optional
            Isosurface value (defaults to 10% of max |value|)
        show_both_phases : bool
            Show both positive and negative phases

        Returns:
        --------
        fig : matplotlib figure
        """
        if isovalue is None:
            isovalue = 0.1 * np.max(np.abs(orbital_values))

        x, y, z = grid_coords

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # Positive phase
        try:
            verts_pos, faces_pos, _, _ = measure.marching_cubes(
                orbital_values, level=isovalue,
                spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0])
            )
            verts_pos[:, 0] += x[0]
            verts_pos[:, 1] += y[0]
            verts_pos[:, 2] += z[0]

            mesh_pos = Poly3DCollection(verts_pos[faces_pos], alpha=0.7,
                                      facecolor='red', edgecolor='none',
                                      label='Positive phase')
            ax.add_collection3d(mesh_pos)

        except Exception as e:
            print(f"Could not generate positive isosurface: {e}")

        # Negative phase
        if show_both_phases:
            try:
                verts_neg, faces_neg, _, _ = measure.marching_cubes(
                    orbital_values, level=-isovalue,
                    spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0])
                )
                verts_neg[:, 0] += x[0]
                verts_neg[:, 1] += y[0]
                verts_neg[:, 2] += z[0]

                mesh_neg = Poly3DCollection(verts_neg[faces_neg], alpha=0.7,
                                          facecolor='blue', edgecolor='none',
                                          label='Negative phase')
                ax.add_collection3d(mesh_neg)

            except Exception as e:
                print(f"Could not generate negative isosurface: {e}")

        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')

        title = f"MO {orbital_info['index']} (E = {orbital_info['energy']:.3f} eV, occ = {orbital_info['occupation']:.1f})"
        ax.set_title(title)
        ax.legend()

        return fig

    def plot_interactive_orbital(self, orbital_values, grid_coords, orbital_info,
                                isovalue=None, show_both_phases=True):
        """
        Create interactive orbital plot with plotly.

        Returns:
        --------
        fig : plotly figure
        """
        if isovalue is None:
            isovalue = 0.1 * np.max(np.abs(orbital_values))

        x, y, z = grid_coords

        data = []

        # Positive phase
        try:
            verts_pos, faces_pos, _, _ = measure.marching_cubes(
                orbital_values, level=isovalue,
                spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0])
            )
            verts_pos[:, 0] += x[0]
            verts_pos[:, 1] += y[0]
            verts_pos[:, 2] += z[0]

            mesh_pos = go.Mesh3d(
                x=verts_pos[:, 0], y=verts_pos[:, 1], z=verts_pos[:, 2],
                i=faces_pos[:, 0], j=faces_pos[:, 1], k=faces_pos[:, 2],
                color='red', opacity=0.7, name='Positive phase'
            )
            data.append(mesh_pos)

        except Exception as e:
            print(f"Could not generate positive isosurface: {e}")

        # Negative phase
        if show_both_phases:
            try:
                verts_neg, faces_neg, _, _ = measure.marching_cubes(
                    orbital_values, level=-isovalue,
                    spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0])
                )
                verts_neg[:, 0] += x[0]
                verts_neg[:, 1] += y[0]
                verts_neg[:, 2] += z[0]

                mesh_neg = go.Mesh3d(
                    x=verts_neg[:, 0], y=verts_neg[:, 1], z=verts_neg[:, 2],
                    i=faces_neg[:, 0], j=faces_neg[:, 1], k=faces_neg[:, 2],
                    color='blue', opacity=0.7, name='Negative phase'
                )
                data.append(mesh_neg)

            except Exception as e:
                print(f"Could not generate negative isosurface: {e}")

        fig = go.Figure(data=data)

        title = f"MO {orbital_info['index']} (E = {orbital_info['energy']:.3f} eV, occ = {orbital_info['occupation']:.1f})"

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )

        return fig

    def plot_frontier_orbitals(self, system, grid_points=25, margin=2.0):
        """
        Plot HOMO and LUMO orbitals.

        Returns:
        --------
        homo_fig, lumo_fig : matplotlib figures
        """
        # Run calculation
        energy, forces = self.calculator.calculate(system)

        # Find HOMO and LUMO indices
        occupied = self.mf.mo_occ > 0
        homo_idx = np.where(occupied)[0][-1]  # Last occupied
        lumo_idx = homo_idx + 1  # First unoccupied

        print(f"HOMO index: {homo_idx}, LUMO index: {lumo_idx}")

        # Calculate orbitals
        homo_values, grid_coords, homo_info = self.calculate_orbital_grid(
            system, homo_idx, grid_points, margin
        )

        lumo_values, _, lumo_info = self.calculate_orbital_grid(
            system, lumo_idx, grid_points, margin
        )

        # Plot both
        homo_fig = self.plot_orbital_isosurface(homo_values, grid_coords, homo_info)
        lumo_fig = self.plot_orbital_isosurface(lumo_values, grid_coords, lumo_info)

        return homo_fig, lumo_fig
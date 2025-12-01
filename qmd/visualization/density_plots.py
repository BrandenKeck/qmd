"""
Visualization utilities for electron density comparison.
"""

import numpy as np
import matplotlib.pyplot as plt

def compare_densities(density1, density2, grid_coords, titles=("Mol 1", "Mol 2")):
    """
    Create 2x2 comparison plot of electron densities.

    Parameters:
    -----------
    density1, density2 : np.ndarray
        3D density arrays
    grid_coords : tuple
        (x, y, z) coordinate arrays
    titles : tuple
        Molecule titles
    """
    x, y, z = grid_coords
    z_center_idx = np.argmin(np.abs(z))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Density 1
    im1 = axes[0, 0].contourf(x, y, density1[:, :, z_center_idx].T,
                              levels=20, cmap='viridis')
    axes[0, 0].set_title(f'{titles[0]} - Density (z=0)')
    axes[0, 0].set_xlabel('x (Å)')
    axes[0, 0].set_ylabel('y (Å)')
    plt.colorbar(im1, ax=axes[0, 0])

    # Density 2
    im2 = axes[0, 1].contourf(x, y, density2[:, :, z_center_idx].T,
                              levels=20, cmap='viridis')
    axes[0, 1].set_title(f'{titles[1]} - Density (z=0)')
    axes[0, 1].set_xlabel('x (Å)')
    axes[0, 1].set_ylabel('y (Å)')
    plt.colorbar(im2, ax=axes[0, 1])

    # Difference
    diff = density1 - density2
    im3 = axes[1, 0].contourf(x, y, diff[:, :, z_center_idx].T,
                              levels=20, cmap='RdBu_r')
    axes[1, 0].set_title('Difference (1 - 2)')
    axes[1, 0].set_xlabel('x (Å)')
    axes[1, 0].set_ylabel('y (Å)')
    plt.colorbar(im3, ax=axes[1, 0])

    # Correlation
    flat1, flat2 = density1.flatten(), density2.flatten()
    axes[1, 1].scatter(flat1, flat2, alpha=0.1, s=1)
    axes[1, 1].plot([flat1.min(), flat1.max()], [flat1.min(), flat1.max()], 'r--')
    axes[1, 1].set_xlabel(f'{titles[0]} Density')
    axes[1, 1].set_ylabel(f'{titles[1]} Density')
    axes[1, 1].set_title('Density Correlation')

    # Stats
    correlation = np.corrcoef(flat1, flat2)[0, 1]
    axes[1, 1].text(0.05, 0.95, f'R = {correlation:.3f}',
                    transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig

def print_density_stats(density1, density2, titles=("Mol 1", "Mol 2")):
    """Print density comparison statistics."""
    diff = density1 - density2
    flat1, flat2 = density1.flatten(), density2.flatten()
    correlation = np.corrcoef(flat1, flat2)[0, 1]

    print(f"\nDensity Statistics:")
    print(f"Mean {titles[0]}: {np.mean(density1):.6f}")
    print(f"Mean {titles[1]}: {np.mean(density2):.6f}")
    print(f"Max {titles[0]}: {np.max(density1):.6f}")
    print(f"Max {titles[1]}: {np.max(density2):.6f}")
    print(f"RMS difference: {np.sqrt(np.mean(diff**2)):.6f}")
    print(f"Correlation: {correlation:.6f}")
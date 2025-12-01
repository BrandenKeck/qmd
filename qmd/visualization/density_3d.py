"""
3D electron density visualization using matplotlib and plotly with GPU acceleration.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import warnings
from ..utils.gpu_utils import GPUConfig, get_array_module, ensure_array, to_cpu, to_gpu
from ..utils.results_utils import get_results_path

def create_isosurface_matplotlib(density, grid_coords, isovalue=None, opacity=0.7, use_gpu=None, save_plot=False, filename="isosurface.png"):
    """
    Create 3D isosurface using matplotlib with GPU acceleration for preprocessing.

    Parameters:
    -----------
    density : np.ndarray
        3D density array
    grid_coords : tuple
        (x, y, z) coordinate arrays
    isovalue : float, optional
        Isosurface value (defaults to 10% of max density)
    opacity : float
        Surface transparency (0-1)
    use_gpu : bool, optional
        Enable GPU acceleration for preprocessing (None = auto-detect)
    save_plot : bool
        Save plot to results/visualizations/ directory
    filename : str
        Filename for saved plot

    Returns:
    --------
    fig : matplotlib figure
    """
    # Determine GPU usage
    if use_gpu is None:
        use_gpu = GPUConfig.is_enabled()

    # Ensure density is on CPU for skimage compatibility
    density_cpu = to_cpu(density)

    if isovalue is None:
        if use_gpu:
            # Use GPU for max calculation if possible
            density_gpu = ensure_array(density)
            xp = get_array_module(density_gpu)
            isovalue = float(0.1 * xp.max(density_gpu))
        else:
            isovalue = 0.1 * np.max(density_cpu)

    x, y, z = grid_coords

    # GPU-accelerated preprocessing if available
    if use_gpu:
        print("Using GPU acceleration for isosurface preprocessing...")
        density_processed = _preprocess_density_gpu(density_cpu, isovalue)
    else:
        density_processed = density_cpu

    # Extract isosurface using marching cubes (CPU-only due to skimage)
    try:
        verts, faces, normals, values = measure.marching_cubes(
            density_processed, level=isovalue, spacing=(
                x[1]-x[0], y[1]-y[0], z[1]-z[0]
            )
        )

        # Adjust vertex coordinates to match grid
        verts[:, 0] += x[0]
        verts[:, 1] += y[0]
        verts[:, 2] += z[0]

    except Exception as e:
        raise ValueError(f"Could not generate isosurface: {e}")

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot isosurface
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh = Poly3DCollection(verts[faces], alpha=opacity, facecolor='blue', edgecolor='none')
    ax.add_collection3d(mesh)

    # Set axes properties
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'Electron Density Isosurface (ρ = {isovalue:.4f})')

    if save_plot:
        output_path = get_results_path(filename, "visualizations")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Isosurface plot saved to: {output_path}")

    return fig

def _preprocess_density_gpu(density, isovalue):
    """
    GPU-accelerated density preprocessing for better isosurface generation.

    Args:
        density: 3D density array
        isovalue: Target isosurface value

    Returns:
        Preprocessed density array (CPU)
    """
    try:
        xp = get_array_module()
        if xp != np:  # GPU available
            density_gpu = ensure_array(density)

            # Apply smoothing and noise reduction
            try:
                import cupyx.scipy.ndimage as cp_ndimage
                # Gaussian smoothing
                density_smooth = cp_ndimage.gaussian_filter(density_gpu, sigma=0.5)
                # Enhance contrast around isovalue
                density_enhanced = xp.where(density_smooth > isovalue * 0.5,
                                          density_smooth * 1.2, density_smooth * 0.8)
                return to_cpu(density_enhanced)
            except ImportError:
                warnings.warn("CuPy ndimage not available, using basic GPU operations")
                # Basic GPU preprocessing
                density_gpu = xp.where(density_gpu > isovalue * 0.1, density_gpu, 0)
                return to_cpu(density_gpu)
        else:
            return density
    except Exception as e:
        warnings.warn(f"GPU preprocessing failed: {e}")
        return density

def create_density_slice_3d(density, grid_coords, slice_planes=['xy', 'xz', 'yz'], save_plot=False, filename="density_slices_3d.png"):
    """
    Create 3D visualization with multiple density slices.

    Parameters:
    -----------
    density : np.ndarray
        3D density array
    grid_coords : tuple
        (x, y, z) coordinate arrays
    slice_planes : list
        Planes to show ('xy', 'xz', 'yz')
    save_plot : bool
        Save plot to results/visualizations/ directory
    filename : str
        Filename for saved plot

    Returns:
    --------
    fig : matplotlib figure
    """
    x, y, z = grid_coords
    nx, ny, nz = density.shape

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrids for each plane
    if 'xy' in slice_planes:
        # XY plane at z=0
        z_idx = nz // 2
        xx_xy, yy_xy = np.meshgrid(x, y, indexing='ij')
        zz_xy = np.full_like(xx_xy, z[z_idx])
        ax.contourf(xx_xy, yy_xy, density[:, :, z_idx],
                   zdir='z', offset=z[z_idx], levels=20, cmap='viridis', alpha=0.7)

    if 'xz' in slice_planes:
        # XZ plane at y=0
        y_idx = ny // 2
        xx_xz, zz_xz = np.meshgrid(x, z, indexing='ij')
        yy_xz = np.full_like(xx_xz, y[y_idx])
        ax.contourf(xx_xz, density[:, y_idx, :], zz_xz,
                   zdir='y', offset=y[y_idx], levels=20, cmap='plasma', alpha=0.7)

    if 'yz' in slice_planes:
        # YZ plane at x=0
        x_idx = nx // 2
        yy_yz, zz_yz = np.meshgrid(y, z, indexing='ij')
        xx_yz = np.full_like(yy_yz, x[x_idx])
        ax.contourf(density[x_idx, :, :], yy_yz, zz_yz,
                   zdir='x', offset=x[x_idx], levels=20, cmap='cividis', alpha=0.7)

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('3D Electron Density Cross-Sections')

    if save_plot:
        output_path = get_results_path(filename, "visualizations")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Density slices plot saved to: {output_path}")

    return fig

def create_dual_isosurface(density1, density2, grid_coords, isovalue=None,
                          colors=['blue', 'red'], labels=['Mol 1', 'Mol 2'],
                          save_plot=False, filename="dual_isosurface.png"):
    """
    Create dual isosurface comparison plot.

    Parameters:
    -----------
    density1, density2 : np.ndarray
        3D density arrays to compare
    grid_coords : tuple
        (x, y, z) coordinate arrays
    isovalue : float, optional
        Isosurface value
    colors : list
        Colors for each isosurface
    labels : list
        Labels for legend

    Returns:
    --------
    fig : matplotlib figure
    """
    if isovalue is None:
        isovalue = 0.1 * max(np.max(density1), np.max(density2))

    x, y, z = grid_coords

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Process both densities
    for i, (density, color, label) in enumerate(zip([density1, density2], colors, labels)):
        try:
            verts, faces, _, _ = measure.marching_cubes(
                density, level=isovalue, spacing=(
                    x[1]-x[0], y[1]-y[0], z[1]-z[0]
                )
            )

            # Adjust coordinates
            verts[:, 0] += x[0]
            verts[:, 1] += y[0]
            verts[:, 2] += z[0]

            # Create mesh with offset for visibility
            offset = i * 0.5  # Small offset for second molecule
            verts[:, 0] += offset

            mesh = Poly3DCollection(verts[faces], alpha=0.6,
                                  facecolor=color, edgecolor='none', label=label)
            ax.add_collection3d(mesh)

        except Exception as e:
            print(f"Warning: Could not generate isosurface for {label}: {e}")

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'Dual Isosurface Comparison (ρ = {isovalue:.4f})')

    # Set equal aspect ratio
    max_range = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min())
    ax.set_xlim(x.min(), x.min() + max_range)
    ax.set_ylim(y.min(), y.min() + max_range)
    ax.set_zlim(z.min(), z.min() + max_range)

    if save_plot:
        output_path = get_results_path(filename, "visualizations")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Dual isosurface plot saved to: {output_path}")

    return fig
"""
Interactive 3D electron density visualization using plotly.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage import measure

def create_interactive_isosurface(density, grid_coords, isovalue=None,
                                 title="Electron Density Isosurface"):
    """
    Create interactive 3D isosurface using plotly.

    Parameters:
    -----------
    density : np.ndarray
        3D density array
    grid_coords : tuple
        (x, y, z) coordinate arrays
    isovalue : float, optional
        Isosurface value (defaults to 10% of max)
    title : str
        Plot title

    Returns:
    --------
    fig : plotly figure
    """
    if isovalue is None:
        isovalue = 0.1 * np.max(density)

    x, y, z = grid_coords

    try:
        # Generate isosurface vertices and faces
        verts, faces, _, _ = measure.marching_cubes(
            density, level=isovalue, spacing=(
                x[1]-x[0], y[1]-y[0], z[1]-z[0]
            )
        )

        # Adjust vertex coordinates
        verts[:, 0] += x[0]
        verts[:, 1] += y[0]
        verts[:, 2] += z[0]

        # Create plotly mesh3d
        fig = go.Figure(data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightblue',
                opacity=0.7,
                name=f'ρ = {isovalue:.4f}'
            )
        ])

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

    except Exception as e:
        # Fallback to volume rendering if isosurface fails
        print(f"Isosurface failed, using volume plot: {e}")
        fig = create_volume_plot(density, grid_coords, title)

    return fig

def create_volume_plot(density, grid_coords, title="Electron Density Volume"):
    """
    Create interactive volume rendering.

    Parameters:
    -----------
    density : np.ndarray
        3D density array
    grid_coords : tuple
        (x, y, z) coordinate arrays
    title : str
        Plot title

    Returns:
    --------
    fig : plotly figure
    """
    x, y, z = grid_coords

    fig = go.Figure(data=go.Volume(
        x=x, y=y, z=z,
        value=density.flatten(),
        isomin=0.01 * np.max(density),
        isomax=0.5 * np.max(density),
        opacity=0.1,
        surface_count=15,
        colorscale='Viridis'
    ))

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

def create_interactive_comparison(density1, density2, grid_coords,
                                 isovalue=None, labels=['Mol 1', 'Mol 2']):
    """
    Create side-by-side interactive density comparison.

    Parameters:
    -----------
    density1, density2 : np.ndarray
        3D density arrays
    grid_coords : tuple
        (x, y, z) coordinate arrays
    isovalue : float, optional
        Isosurface value
    labels : list
        Molecule labels

    Returns:
    --------
    fig : plotly figure
    """
    if isovalue is None:
        isovalue = 0.1 * max(np.max(density1), np.max(density2))

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=labels
    )

    x, y, z = grid_coords
    colors = ['lightblue', 'lightcoral']

    for i, (density, label, color) in enumerate(zip([density1, density2], labels, colors)):
        try:
            verts, faces, _, _ = measure.marching_cubes(
                density, level=isovalue, spacing=(
                    x[1]-x[0], y[1]-y[0], z[1]-z[0]
                )
            )

            verts[:, 0] += x[0]
            verts[:, 1] += y[0]
            verts[:, 2] += z[0]

            mesh = go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                opacity=0.7,
                name=label
            )

            fig.add_trace(mesh, row=1, col=i+1)

        except Exception as e:
            print(f"Could not create isosurface for {label}: {e}")

    fig.update_layout(
        title=f'Interactive Density Comparison (ρ = {isovalue:.4f})',
        width=1200,
        height=600
    )

    return fig

def create_density_animation(densities, grid_coords, titles=None, isovalue=None):
    """
    Create animated density visualization for trajectory data.

    Parameters:
    -----------
    densities : list of np.ndarray
        List of 3D density arrays
    grid_coords : tuple
        (x, y, z) coordinate arrays
    titles : list, optional
        Frame titles
    isovalue : float, optional
        Isosurface value

    Returns:
    --------
    fig : plotly figure
    """
    if isovalue is None:
        isovalue = 0.1 * max(np.max(d) for d in densities)

    if titles is None:
        titles = [f'Frame {i}' for i in range(len(densities))]

    x, y, z = grid_coords

    # Create frames
    frames = []
    for i, (density, title) in enumerate(zip(densities, titles)):
        try:
            verts, faces, _, _ = measure.marching_cubes(
                density, level=isovalue, spacing=(
                    x[1]-x[0], y[1]-y[0], z[1]-z[0]
                )
            )

            verts[:, 0] += x[0]
            verts[:, 1] += y[0]
            verts[:, 2] += z[0]

            frame = go.Frame(
                data=[go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color='lightblue',
                    opacity=0.7
                )],
                name=str(i)
            )
            frames.append(frame)

        except Exception as e:
            print(f"Could not create frame {i}: {e}")

    # Initial frame
    if frames:
        initial_data = frames[0].data
    else:
        initial_data = []

    fig = go.Figure(
        data=initial_data,
        frames=frames
    )

    # Add animation controls
    fig.update_layout(
        title='Animated Electron Density',
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            aspectmode='cube'
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {'label': 'Play', 'method': 'animate',
                 'args': [None, {'frame': {'duration': 500}}]},
                {'label': 'Pause', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
            ]
        }],
        width=800,
        height=600
    )

    return fig
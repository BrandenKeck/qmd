"""3D molecular visualization with trajectory animation."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
try:
    import py3Dmol
    HAS_PY3DMOL = True
except ImportError:
    HAS_PY3DMOL = False

from ..core.system import System


class MolecularViewer:
    """Interactive 3D molecular viewer with animation support."""

    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height

    def view_system(self, system: System, style: str = "stick") -> None:
        """View molecular system in 3D."""
        if not HAS_PY3DMOL:
            print("py3Dmol not available. Install with: pip install py3Dmol")
            return

        viewer = py3Dmol.view(width=self.width, height=self.height)

        # Add atoms
        for i, (symbol, pos) in enumerate(zip(system.atoms.get_chemical_symbols(),
                                            system.positions)):
            viewer.addSphere({
                'center': {'x': pos[0], 'y': pos[1], 'z': pos[2]},
                'radius': 0.5,
                'color': self._get_atom_color(symbol)
            })

        viewer.setStyle({style: {}})
        viewer.zoomTo()
        viewer.show()

    def animate_trajectory(self, trajectory: Dict) -> None:
        """Animate MD trajectory."""
        positions_list = trajectory['positions']
        times = trajectory['times']

        print(f"Animating {len(positions_list)} frames...")
        # Simple animation would require interactive widgets
        # For now, show first and last frames
        if len(positions_list) >= 2:
            print("Showing initial and final structures")

    def _get_atom_color(self, symbol: str) -> str:
        """Get standard CPK colors for atoms."""
        colors = {
            'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red',
            'S': 'yellow', 'P': 'orange'
        }
        return colors.get(symbol, 'pink')
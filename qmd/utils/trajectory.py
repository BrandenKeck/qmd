"""Trajectory analysis tools."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from ..utils.results_utils import get_results_path


class TrajectoryAnalyzer:
    """Analysis tools for MD trajectories."""

    def __init__(self, trajectory: Dict):
        self.trajectory = trajectory

    def plot_energies(self, save_plot=True, filename="energy_plot.png") -> None:
        """Plot energy vs time."""
        energies = self.trajectory['energies']
        times = [e['time'] for e in energies]
        ke = [e['kinetic_energy'] for e in energies]
        pe = [e['potential_energy'] for e in energies]
        total = [e['total_energy'] for e in energies]

        plt.figure(figsize=(10, 6))
        plt.plot(times, ke, label='Kinetic', alpha=0.8)
        plt.plot(times, pe, label='Potential', alpha=0.8)
        plt.plot(times, total, label='Total', linewidth=2)
        plt.xlabel('Time (fs)')
        plt.ylabel('Energy (eV)')
        plt.legend()
        plt.title('Energy Conservation')
        plt.grid(True, alpha=0.3)

        if save_plot:
            output_path = get_results_path(filename, "plots")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Energy plot saved to: {output_path}")

        plt.show()

    def plot_temperature(self, save_plot=True, filename="temperature_plot.png") -> None:
        """Plot temperature vs time."""
        energies = self.trajectory['energies']
        times = [e['time'] for e in energies]
        temps = [e['temperature'] for e in energies]

        plt.figure(figsize=(8, 5))
        plt.plot(times, temps, linewidth=2)
        plt.xlabel('Time (fs)')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature Evolution')
        plt.grid(True, alpha=0.3)

        if save_plot:
            output_path = get_results_path(filename, "plots")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Temperature plot saved to: {output_path}")

        plt.show()

    def calculate_rmsd(self, reference_idx: int = 0) -> np.ndarray:
        """Calculate RMSD from reference structure."""
        positions = self.trajectory['positions']
        ref_pos = positions[reference_idx]

        rmsds = []
        for pos in positions:
            rmsd = np.sqrt(np.mean((pos - ref_pos)**2))
            rmsds.append(rmsd)

        return np.array(rmsds)
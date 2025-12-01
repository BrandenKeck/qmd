"""
Born-Oppenheimer molecular dynamics integrator.
"""

import numpy as np
from typing import Optional, Callable, Dict, Any
from ..quantum.calculator import QuantumCalculator
from .system import System


class BornOppenheimerIntegrator:
    """
    Born-Oppenheimer molecular dynamics integrator using velocity Verlet algorithm.
    """

    def __init__(self, calculator: QuantumCalculator, timestep: float = 0.5):
        """
        Initialize integrator.

        Args:
            calculator: Quantum chemistry calculator
            timestep: Time step in femtoseconds
        """
        self.calculator = calculator
        self.timestep = timestep  # fs
        self.step_count = 0

        # Convert timestep to atomic units
        self.dt_au = timestep * 41.341374575751  # fs to atomic time units

    def step(self, system: System) -> Dict[str, float]:
        """
        Perform one MD step using velocity Verlet algorithm.

        Args:
            system: Molecular system

        Returns:
            Dictionary with energies and properties
        """
        # Get current forces
        if system.forces is None:
            system.forces = self.calculator.get_forces(system.atoms)

        # Store old positions and forces
        old_forces = system.forces.copy()

        # Update positions: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        # Convert forces to accelerations (F = ma, so a = F/m)
        accelerations = system.forces / system.masses[:, np.newaxis]

        # Convert units for integration
        dt_ang_per_fs = self.timestep  # Already in correct units

        new_positions = (system.positions +
                        system.velocities * dt_ang_per_fs +
                        0.5 * accelerations * dt_ang_per_fs**2)

        system.update_positions(new_positions)

        # Calculate new forces at new positions
        new_forces = self.calculator.get_forces(system.atoms)
        system.update_forces(new_forces)

        # Update velocities: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        new_accelerations = new_forces / system.masses[:, np.newaxis]
        new_velocities = (system.velocities +
                         0.5 * (accelerations + new_accelerations) * dt_ang_per_fs)

        system.update_velocities(new_velocities)

        # Get energies
        kinetic_energy = system.get_kinetic_energy()
        potential_energy = self.calculator.get_energy(system.atoms)
        total_energy = kinetic_energy + potential_energy

        self.step_count += 1

        return {
            'step': self.step_count,
            'time': self.step_count * self.timestep,
            'kinetic_energy': kinetic_energy,
            'potential_energy': potential_energy,
            'total_energy': total_energy,
            'temperature': system.get_temperature()
        }

    def run(self, system: System, steps: int,
            output_frequency: int = 10) -> Dict[str, Any]:
        """
        Run MD simulation.

        Args:
            system: Molecular system
            steps: Number of MD steps
            output_frequency: Write output every N steps

        Returns:
            Trajectory data
        """
        trajectory = {
            'positions': [],
            'velocities': [],
            'forces': [],
            'energies': [],
            'times': []
        }

        print(f"Starting MD simulation for {steps} steps...")

        for i in range(steps):
            result = self.step(system)

            if i % output_frequency == 0:
                trajectory['positions'].append(system.positions.copy())
                trajectory['velocities'].append(system.velocities.copy())
                trajectory['forces'].append(system.forces.copy())
                trajectory['energies'].append(result)
                trajectory['times'].append(result['time'])

                print(f"Step {i}: E_total = {result['total_energy']:.6f} eV, "
                      f"T = {result['temperature']:.1f} K")

        return trajectory
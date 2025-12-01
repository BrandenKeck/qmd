"""
PySCF interface for quantum chemistry calculations with GPU acceleration support.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from ase import Atoms
import warnings
import os

try:
    import pyscf
    from pyscf import gto, scf, dft, grad
    HAS_PYSCF = True

    # Check for PySCF GPU support
    try:
        from pyscf import gpu4pyscf
        HAS_PYSCF_GPU = True
    except ImportError:
        HAS_PYSCF_GPU = False

except ImportError:
    HAS_PYSCF = False
    HAS_PYSCF_GPU = False
    warnings.warn("PySCF not available. Install with: pip install pyscf")

from .calculator import QuantumCalculator
from ..utils.gpu_utils import GPUConfig, get_array_module, ensure_array, to_cpu, to_gpu


class PySCFCalculator(QuantumCalculator):
    """
    PySCF-based quantum chemistry calculator for molecular dynamics with GPU acceleration.
    """

    def __init__(self, method: str = "B3LYP", basis: str = "6-31G*", charge: int = 0,
                 spin: int = 0, max_memory: int = 4000, use_gpu: Optional[bool] = None, **kwargs):
        """
        Initialize PySCF calculator.

        Args:
            method: DFT functional or HF method
            basis: Basis set
            charge: Molecular charge
            spin: Spin multiplicity - 1 (0 = singlet, 1 = doublet, etc.)
            max_memory: Memory limit in MB
            use_gpu: Enable GPU acceleration (None = auto-detect)
            **kwargs: Additional PySCF options
        """
        if not HAS_PYSCF:
            raise ImportError("PySCF is required but not installed")

        super().__init__(method, basis, **kwargs)
        self.charge = charge
        self.spin = spin
        self.max_memory = max_memory
        self.mol: Optional[gto.Mole] = None
        self.mf: Optional[scf.RHF] = None
        self._gradient_calculator = None

        # GPU configuration
        if use_gpu is None:
            self.use_gpu = GPUConfig.is_enabled()
        else:
            self.use_gpu = use_gpu and GPUConfig.is_available()

        if self.use_gpu and not HAS_PYSCF_GPU:
            warnings.warn("GPU requested but PySCF GPU module not available. "
                         "Install with: pip install pyscf[gpu]")
            self.use_gpu = False

        # Setup GPU environment for PySCF
        if self.use_gpu:
            os.environ['PYSCF_MAX_MEMORY'] = str(max_memory)
            from ..utils.gpu_utils import _GPU_DEVICE
            os.environ['CUDA_VISIBLE_DEVICES'] = str(_GPU_DEVICE)
            print(f"PySCF GPU acceleration enabled on device {_GPU_DEVICE}")

    def _build_molecule(self, atoms: Atoms) -> gto.Mole:
        """
        Build PySCF molecule object from ASE atoms.

        Args:
            atoms: ASE Atoms object

        Returns:
            PySCF Mole object
        """
        # Convert ASE atoms to PySCF format
        atom_string = []
        for symbol, pos in zip(atoms.get_chemical_symbols(), atoms.positions):
            atom_string.append(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")

        mol = gto.Mole()
        mol.atom = '; '.join(atom_string)
        mol.basis = self.basis
        mol.charge = self.charge
        mol.spin = self.spin
        mol.max_memory = self.max_memory
        mol.verbose = 0  # Suppress output
        mol.build()

        return mol

    def _create_mean_field(self, mol: gto.Mole) -> scf.RHF:
        """
        Create mean-field object for SCF calculation with optional GPU acceleration.

        Args:
            mol: PySCF molecule object

        Returns:
            Mean-field object
        """
        # Use GPU-accelerated versions if available
        if self.use_gpu and HAS_PYSCF_GPU:
            try:
                # Import GPU modules
                from pyscf.gpu4pyscf import scf as gpu_scf
                from pyscf.gpu4pyscf import dft as gpu_dft

                if self.method.upper() == 'HF':
                    if mol.spin == 0:
                        mf = gpu_scf.RHF(mol)
                    else:
                        mf = gpu_scf.UHF(mol)
                else:
                    # DFT calculation
                    if mol.spin == 0:
                        mf = gpu_dft.RKS(mol)
                    else:
                        mf = gpu_dft.UKS(mol)
                    mf.xc = self.method

                print("Using GPU-accelerated PySCF")

            except Exception as e:
                warnings.warn(f"GPU acceleration failed, falling back to CPU: {e}")
                self.use_gpu = False

        # Fallback to CPU version
        if not self.use_gpu:
            if self.method.upper() == 'HF':
                if mol.spin == 0:
                    mf = scf.RHF(mol)
                else:
                    mf = scf.UHF(mol)
            else:
                # DFT calculation
                if mol.spin == 0:
                    mf = dft.RKS(mol)
                else:
                    mf = dft.UKS(mol)
                mf.xc = self.method

        # Apply any additional options
        for key, value in self.options.items():
            if hasattr(mf, key):
                setattr(mf, key, value)

        return mf

    def calculate(self, atoms: Atoms) -> Dict[str, Any]:
        """
        Perform quantum chemistry calculation.

        Args:
            atoms: ASE Atoms object

        Returns:
            Dictionary with energy, forces, and other properties
        """
        # Build molecule
        self.mol = self._build_molecule(atoms)

        # Create mean-field object
        self.mf = self._create_mean_field(self.mol)

        # Run SCF calculation
        self.energy = self.mf.kernel()

        # Calculate forces (gradients)
        if hasattr(self.mf, 'xc'):
            # DFT method (RKS or UKS)
            grad_calc = grad.RKS(self.mf)
        else:
            # Hartree-Fock method (RHF or UHF)
            grad_calc = grad.RHF(self.mf)
        gradients = grad_calc.kernel()

        # Convert gradients to forces (negative gradient)
        # Convert from Hartree/Bohr to eV/Angstrom
        hartree_to_ev = 27.211386245988
        bohr_to_angstrom = 0.529177210903
        conversion = hartree_to_ev / bohr_to_angstrom

        self.forces = -gradients * conversion

        # Calculate additional properties
        self._calculate_properties()

        return {
            'energy': self.energy * hartree_to_ev,  # Convert to eV
            'forces': self.forces,
            'charges': self.charges,
            'dipole': self.dipole
        }

    def _calculate_properties(self) -> None:
        """Calculate additional molecular properties."""
        if self.mf is None or self.mol is None:
            return

        try:
            # Mulliken population analysis
            mulliken = self.mf.mulliken_pop(verbose=0)
            if len(mulliken) >= 2:
                self.charges = mulliken[1]  # Atomic charges

            # Dipole moment
            dipole = self.mf.dip_moment(verbose=0)
            self.dipole = np.array(dipole)

        except Exception as e:
            warnings.warn(f"Could not calculate additional properties: {e}")

    def get_energy(self, atoms: Atoms) -> float:
        """
        Get total electronic energy.

        Args:
            atoms: ASE Atoms object

        Returns:
            Total energy in eV
        """
        if self.energy is None:
            self.calculate(atoms)

        # Convert from Hartree to eV
        return self.energy * 27.211386245988

    def get_forces(self, atoms: Atoms) -> np.ndarray:
        """
        Get forces on atoms.

        Args:
            atoms: ASE Atoms object

        Returns:
            Forces array in eV/Angstrom
        """
        if self.forces is None:
            self.calculate(atoms)

        return self.forces.copy()

    def optimize_geometry(self, atoms: Atoms, max_steps: int = 100,
                         convergence: float = 1e-4) -> Atoms:
        """
        Optimize molecular geometry.

        Args:
            atoms: Initial ASE Atoms object
            max_steps: Maximum optimization steps
            convergence: Force convergence criterion in eV/Angstrom

        Returns:
            Optimized ASE Atoms object
        """
        from ase.optimize import BFGS
        from ase.calculators.calculator import Calculator

        # Create ASE calculator wrapper
        class PySCFASECalculator(Calculator):
            implemented_properties = ['energy', 'forces']

            def __init__(self, pyscf_calc):
                super().__init__()
                self.pyscf_calc = pyscf_calc

            def calculate(self, atoms, properties, system_changes):
                Calculator.calculate(self, atoms, properties, system_changes)
                results = self.pyscf_calc.calculate(atoms)
                self.results['energy'] = results['energy']
                self.results['forces'] = results['forces']

        # Set up ASE calculator
        atoms.calc = PySCFASECalculator(self)

        # Optimize
        optimizer = BFGS(atoms)
        optimizer.run(fmax=convergence, steps=max_steps)

        return atoms

    def reset(self) -> None:
        """Reset calculator state."""
        super().reset()
        self.mol = None
        self.mf = None
        self._gradient_calculator = None
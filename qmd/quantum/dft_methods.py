"""
Common DFT methods and basis sets for quantum molecular dynamics.
"""

from enum import Enum
from typing import Dict, List, Tuple


class DFTMethod(Enum):
    """Common DFT functionals for molecular dynamics."""

    # GGA functionals
    PBE = "PBE"
    BLYP = "BLYP"
    PW91 = "PW91"

    # Hybrid functionals
    B3LYP = "B3LYP"
    PBE0 = "PBE0"
    HSE06 = "HSE06"

    # Meta-GGA
    TPSS = "TPSS"
    M06L = "M06L"

    # Range-separated
    CAM_B3LYP = "CAM-B3LYP"
    WB97XD = "wB97X-D"

    # Hartree-Fock
    HF = "HF"


class BasisSet(Enum):
    """Common basis sets."""

    # Pople basis sets
    STO_3G = "STO-3G"
    THREETHIRTYOONE_G = "3-21G"
    SIXTHIRTYOONE_G = "6-31G"
    SIXTHIRTYOONE_GS = "6-31G*"
    SIXTHIRTYOONE_GSS = "6-31G**"
    SIXTHIRTYOOONE_PLUS_GS = "6-311+G*"
    SIXTHIRTYOOONE_PLUS_GSS = "6-311+G**"

    # Correlation consistent
    CC_PVDZ = "cc-pVDZ"
    CC_PVTZ = "cc-pVTZ"
    CC_PVQZ = "cc-pVQZ"
    AUG_CC_PVDZ = "aug-cc-pVDZ"
    AUG_CC_PVTZ = "aug-cc-pVTZ"

    # Def2 basis sets
    DEF2_SVP = "def2-SVP"
    DEF2_SVPD = "def2-SVPD"
    DEF2_TZVP = "def2-TZVP"
    DEF2_TZVPD = "def2-TZVPD"


# Recommended method/basis combinations for different scenarios
RECOMMENDED_COMBINATIONS = {
    "fast": (DFTMethod.PBE, BasisSet.DEF2_SVP),
    "balanced": (DFTMethod.B3LYP, BasisSet.SIXTHIRTYOONE_GS),
    "accurate": (DFTMethod.PBE0, BasisSet.DEF2_TZVP),
    "high_accuracy": (DFTMethod.CAM_B3LYP, BasisSet.AUG_CC_PVTZ),
}

# Typical computational costs (relative)
COMPUTATIONAL_COST = {
    DFTMethod.HF: 1.0,
    DFTMethod.PBE: 1.2,
    DFTMethod.BLYP: 1.3,
    DFTMethod.B3LYP: 2.0,
    DFTMethod.PBE0: 2.2,
    DFTMethod.CAM_B3LYP: 3.5,
    DFTMethod.WB97XD: 4.0,
}

BASIS_SIZE = {
    BasisSet.STO_3G: 1,
    BasisSet.THREETHIRTYOONE_G: 2,
    BasisSet.SIXTHIRTYOONE_G: 3,
    BasisSet.SIXTHIRTYOONE_GS: 4,
    BasisSet.DEF2_SVP: 4,
    BasisSet.SIXTHIRTYOONE_GSS: 5,
    BasisSet.CC_PVDZ: 6,
    BasisSet.DEF2_TZVP: 8,
    BasisSet.CC_PVTZ: 12,
    BasisSet.AUG_CC_PVTZ: 16,
}


def get_recommended_setup(scenario: str = "balanced") -> Tuple[DFTMethod, BasisSet]:
    """
    Get recommended method/basis combination for a given scenario.

    Args:
        scenario: One of 'fast', 'balanced', 'accurate', 'high_accuracy'

    Returns:
        Tuple of (method, basis_set)
    """
    if scenario not in RECOMMENDED_COMBINATIONS:
        raise ValueError(f"Unknown scenario '{scenario}'. "
                        f"Available: {list(RECOMMENDED_COMBINATIONS.keys())}")

    return RECOMMENDED_COMBINATIONS[scenario]


def estimate_relative_cost(method: DFTMethod, basis: BasisSet, n_atoms: int) -> float:
    """
    Estimate relative computational cost.

    Args:
        method: DFT method
        basis: Basis set
        n_atoms: Number of atoms

    Returns:
        Relative cost factor
    """
    method_cost = COMPUTATIONAL_COST.get(method, 2.0)
    basis_cost = BASIS_SIZE.get(basis, 4)

    # Rough scaling: O(N^3) for SCF, additional factors for basis size
    return method_cost * (basis_cost ** 2) * (n_atoms ** 2.5) / 1000


def suggest_method_basis(n_atoms: int, accuracy: str = "balanced",
                        max_cost: float = 100.0) -> Tuple[DFTMethod, BasisSet]:
    """
    Suggest method and basis set based on system size and requirements.

    Args:
        n_atoms: Number of atoms in the system
        accuracy: Desired accuracy level
        max_cost: Maximum acceptable relative cost

    Returns:
        Recommended (method, basis_set) combination
    """
    # Start with recommended combination
    method, basis = get_recommended_setup(accuracy)

    # Check if cost is acceptable
    cost = estimate_relative_cost(method, basis, n_atoms)

    if cost > max_cost:
        # Try to reduce cost
        if n_atoms > 100:
            # For large systems, use smaller basis
            if accuracy == "high_accuracy":
                return suggest_method_basis(n_atoms, "accurate", max_cost)
            elif accuracy == "accurate":
                return suggest_method_basis(n_atoms, "balanced", max_cost)
            else:
                return suggest_method_basis(n_atoms, "fast", max_cost)
        else:
            # For smaller systems but high cost, reduce method complexity
            if method in [DFTMethod.CAM_B3LYP, DFTMethod.WB97XD]:
                return DFTMethod.B3LYP, basis
            elif method == DFTMethod.B3LYP:
                return DFTMethod.PBE, basis

    return method, basis
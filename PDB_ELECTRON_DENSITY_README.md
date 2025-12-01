# PDB Electron Density Calculator with GPU Acceleration

This tool provides GPU-accelerated electron density calculation for protein structures downloaded from the Protein Data Bank (PDB).

## Features

- **Automatic PDB Download**: Downloads protein structures directly from the RCSB Protein Data Bank
- **GPU Acceleration**: Uses CuPy for fast electron density calculations on NVIDIA GPUs
- **Flexible Grid Resolution**: Configurable grid spacing and size for different accuracy needs
- **Atomic Form Factors**: Includes realistic atomic form factors for common elements (C, N, O, S, P, H)
- **Temperature Factor Support**: Incorporates B-factors and occupancy from PDB files
- **Visualization**: Built-in 2D slice visualization of electron density maps
- **Batch Processing**: Support for processing multiple structures

## Requirements

**Key requirements:**
- `cupy-cuda11x` or `cupy-cuda12x` (for GPU acceleration)
- `biopython` (for PDB parsing)
- `matplotlib` (for visualization)
- `numpy` and `requests`

**Note**: If you don't have CUDA/CuPy installed, the code will automatically fall back to CPU computation using NumPy.

## Quick Start

### Basic Usage

```python
from pdb_electron_density import GPUElectronDensityCalculator

# Initialize calculator
calculator = GPUElectronDensityCalculator(
    grid_spacing=0.3,      # Angstrom resolution
    grid_size=(80, 80, 80) # Grid dimensions
)

# Download and process a protein
pdb_path = calculator.download_pdb("1crn")  # Crambin protein
atoms = calculator.parse_pdb(pdb_path)
density = calculator.calculate_electron_density(atoms)

# Visualize and save results
calculator.visualize_density(density)
calculator.save_density_map(density, "crambin_density.npy")
```

### Run Examples

```bash
# Quick example with small protein
python3 example_usage.py

# Or run the full main script
python3 pdb_electron_density.py
```

## Parameters

### Grid Settings
- **grid_spacing**: Distance between grid points in Angstroms (smaller = higher resolution)
- **grid_size**: Number of grid points in each dimension (larger = more detail)

### Recommended Settings
- **Fast Preview**: `grid_spacing=0.5, grid_size=(60, 60, 60)`
- **Standard Quality**: `grid_spacing=0.3, grid_size=(80, 80, 80)`
- **High Resolution**: `grid_spacing=0.2, grid_size=(150, 150, 150)`

## Performance

With GPU acceleration (CuPy), calculation times are typically:
- Small protein (300 atoms): ~1-2 seconds
- Medium protein (1000 atoms): ~5-10 seconds
- Large protein (3000+ atoms): ~30-60 seconds

Without GPU (CPU fallback), times will be 10-100x longer.

## Files Created

- **pdb_electron_density.py**: Main calculator class
- **example_usage.py**: Usage examples and demonstrations
- **requirements.txt**: Python package dependencies (updated with requests)

## Example Proteins

Good test proteins with their PDB IDs:
- **1CRN**: Crambin (46 residues, very small)
- **1UBQ**: Ubiquitin (76 residues, small)
- **1LYZ**: Lysozyme (129 residues, medium)
- **1MBN**: Myoglobin (153 residues, medium)

## Technical Details

The electron density calculation uses:
- Atomic form factors with Gaussian approximation
- Temperature factors (B-factors) from PDB files
- Occupancy values for partially occupied atoms
- Distance-based density contribution for each atom

## Troubleshooting

**CUDA/CuPy Issues**:
- Ensure you have CUDA installed and a compatible GPU
- The code automatically falls back to CPU if CuPy is unavailable

**Memory Issues**:
- Reduce grid_size for large proteins
- Increase grid_spacing to reduce memory usage
- The code processes atoms in batches to manage GPU memory
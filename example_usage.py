#!/usr/bin/env python3
"""
Example usage of the PDB Electron Density Calculator

This script demonstrates how to use the GPUElectronDensityCalculator
to download, parse, and calculate electron density for PDB structures.
"""

from pdb_electron_density import GPUElectronDensityCalculator
import numpy as np


def quick_example():
    """Quick example with a small protein."""
    print("=== Quick Example: Small Protein (1CRN - Crambin) ===")

    # Initialize calculator with moderate resolution
    calculator = GPUElectronDensityCalculator(
        grid_spacing=0.4,     # 0.4 Angstrom spacing
        grid_size=(60, 60, 60)  # 60x60x60 grid
    )

    # Download and process a small protein
    pdb_id = "1crn"  # Crambin - 46 residues, good for testing
    pdb_path = calculator.download_pdb(pdb_id)
    atoms = calculator.parse_pdb(pdb_path)

    # Calculate electron density
    density = calculator.calculate_electron_density(atoms)

    # Show visualization (will display matplotlib plots)
    calculator.visualize_density(density)

    # Save the results
    calculator.save_density_map(density, f"{pdb_id}_density.npy")

    print(f"Completed processing {pdb_id}")


def high_resolution_example():
    """High-resolution example with a larger protein."""
    print("\n=== High Resolution Example: Lysozyme (1LYZ) ===")

    # Initialize calculator with high resolution
    calculator = GPUElectronDensityCalculator(
        grid_spacing=0.2,       # 0.2 Angstrom spacing (high resolution)
        grid_size=(150, 150, 150)  # Large grid for detail
    )

    # Download lysozyme - classic protein structure
    pdb_id = "1lyz"
    pdb_path = calculator.download_pdb(pdb_id)
    atoms = calculator.parse_pdb(pdb_path)

    print(f"Processing {len(atoms)} atoms at high resolution...")
    print("This may take a few minutes with GPU acceleration...")

    # Calculate high-resolution electron density
    density = calculator.calculate_electron_density(atoms)

    # Visualize multiple slices
    calculator.visualize_density(density, slice_index=density.shape[2]//3)
    calculator.visualize_density(density, slice_index=2*density.shape[2]//3)

    # Save the high-resolution map
    calculator.save_density_map(density, f"{pdb_id}_high_res_density.npy")

    print(f"Completed high-resolution processing of {pdb_id}")


def custom_visualization_example():
    """Example showing custom analysis of electron density."""
    print("\n=== Custom Analysis Example ===")

    calculator = GPUElectronDensityCalculator(
        grid_spacing=0.3,
        grid_size=(80, 80, 80)
    )

    # Use a DNA structure
    pdb_id = "1bna"  # B-form DNA
    pdb_path = calculator.download_pdb(pdb_id)
    atoms = calculator.parse_pdb(pdb_path)

    density = calculator.calculate_electron_density(atoms)

    # Convert to numpy for analysis
    if hasattr(density, 'get'):  # CuPy array
        density_np = density.get()
    else:  # Already numpy
        density_np = density

    # Custom analysis
    print(f"Density Statistics:")
    print(f"  Shape: {density_np.shape}")
    print(f"  Min/Max: {density_np.min():.4f} / {density_np.max():.4f}")
    print(f"  Mean/Std: {density_np.mean():.4f} / {density_np.std():.4f}")

    # Find high-density regions
    threshold = density_np.mean() + 2 * density_np.std()
    high_density_points = np.where(density_np > threshold)
    print(f"  High-density points (>{threshold:.4f}): {len(high_density_points[0])}")

    # Save both visualization and raw data
    calculator.visualize_density(density)
    calculator.save_density_map(density, f"{pdb_id}_custom_analysis.npy")

    print(f"Completed custom analysis of {pdb_id}")


def batch_processing_example():
    """Example of processing multiple PDB structures."""
    print("\n=== Batch Processing Example ===")

    # List of interesting small proteins
    pdb_ids = ["1crn", "1ubq", "1mbn"]  # Crambin, Ubiquitin, Myoglobin

    calculator = GPUElectronDensityCalculator(
        grid_spacing=0.35,
        grid_size=(70, 70, 70)
    )

    results = {}

    for pdb_id in pdb_ids:
        print(f"\nProcessing {pdb_id}...")

        try:
            pdb_path = calculator.download_pdb(pdb_id)
            atoms = calculator.parse_pdb(pdb_path)
            density = calculator.calculate_electron_density(atoms)

            # Save results
            output_file = f"{pdb_id}_batch_density.npy"
            calculator.save_density_map(density, output_file)

            # Store summary statistics
            if hasattr(density, 'get'):
                density_np = density.get()
            else:
                density_np = density

            results[pdb_id] = {
                'num_atoms': len(atoms),
                'density_max': density_np.max(),
                'density_mean': density_np.mean()
            }

            print(f"  {pdb_id}: {len(atoms)} atoms, max density: {density_np.max():.4f}")

        except Exception as e:
            print(f"  Error processing {pdb_id}: {e}")
            continue

    print(f"\n=== Batch Processing Summary ===")
    for pdb_id, stats in results.items():
        print(f"{pdb_id}: {stats['num_atoms']} atoms, "
              f"mean density: {stats['density_mean']:.4f}")


if __name__ == "__main__":
    # Run examples (comment out any you don't want to run)

    # Quick test with small protein
    quick_example()

    # Uncomment for more examples:
    # high_resolution_example()
    # custom_visualization_example()
    # batch_processing_example()

    print("\n=== All Examples Complete ===")
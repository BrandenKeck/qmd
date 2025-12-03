"""
Quantum Mechanics Molecular Dynamics Package
Main entry point for antibody analysis workflow

This package is config-driven and requires a YAML configuration file
containing antibody sequences and analysis parameters.
"""

import argparse
import os
from pathlib import Path
import yaml
from typing import Dict, Any
from antibody_converter import convert_antibody
from electron_density import ElectronDensityCalculator
from visualizer import DensityExporter


class QMDPipeline:
    """
    Main pipeline for antibody quantum MD analysis.

    Requires a configuration dictionary containing at minimum:
    - heavy_chain: Heavy chain amino acid sequence
    - light_chain: Light chain amino acid sequence

    Optional config values:
    - output_dir: Output directory path (default: 'qmd_output')
    - density_method: Electron density calculation method (default: 'gaussian')
    - grid_spacing: Grid spacing in Angstroms (default: 0.5)
    """

    def __init__(self, use_gpu: bool = True, config: Dict[str, Any] = None):
        if config is None:
            raise ValueError("Config dictionary is required. Use QMDPipeline.load_config() to load from file.")
        self.config = config
        self.use_gpu = use_gpu
        self.density_calc = ElectronDensityCalculator(use_gpu=use_gpu, config=config)
        self.exporter = DensityExporter()

    @staticmethod
    def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
        """Load configuration from YAML file. Config file must exist."""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                if config is None:
                    raise ValueError(f"Config file '{config_file}' is empty or invalid")
                return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file '{config_file}' not found. A config file is required.")

    def run_full_pipeline(self) -> dict:
        """Run complete antibody QM/MD analysis pipeline using config values only."""

        # Get required values from config
        heavy_chain = self.config.get('heavy_chain')
        light_chain = self.config.get('light_chain')
        output_dir = self.config.get('output_dir', 'qmd_output')
        density_method = self.config.get('density_method', 'gaussian')
        grid_spacing = self.config.get('grid_spacing', 0.5)

        # Validate required config values
        if not heavy_chain:
            raise ValueError("'heavy_chain' must be specified in config")
        if not light_chain:
            raise ValueError("'light_chain' must be specified in config")

        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        results = {}

        # Step 1: Convert sequences to mmCIF
        print("Step 1: Converting antibody sequences to mmCIF...")
        mmcif_filename = self.config.get('mmcif_filename', 'antibody.cif')
        mmcif_file = os.path.join(output_dir, mmcif_filename)
        mmcif_result = convert_antibody(heavy_chain, light_chain, mmcif_file, self.config)
        results['mmcif'] = mmcif_result
        print(f"  Generated: {mmcif_result}")

        # Step 2: Calculate electron density
        print(f"Step 2: Calculating electron density ({density_method} method)...")
        h5_filename = self.config.get('density_h5_filename', 'density.h5')
        h5_file = os.path.join(output_dir, h5_filename)
        density_result = self.density_calc.calculate_density(
            mmcif_file, density_method, grid_spacing, h5_file
        )
        results['density_h5'] = density_result
        print(f"  Generated: {density_result}")

        # Step 3: Export density data
        print("Step 3: Exporting density data...")
        cube_filename = self.config.get('density_cube_filename', 'density.cube')
        npy_filename = self.config.get('density_npy_filename', 'density.npy')
        cube_file = os.path.join(output_dir, cube_filename)
        npy_file = os.path.join(output_dir, npy_filename)

        export_results = self.exporter.export_density(
            h5_file, cube_file, npy_file, grid_spacing
        )
        results['exports'] = export_results

        for export_result in export_results:
            print(f"  {export_result}")

        print(f"\nPipeline complete! Results in: {output_dir}")
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Quantum Mechanics Molecular Dynamics Package for Antibodies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (requires config.yaml with heavy_chain and light_chain)
  python qmd_package.py

  # With custom config file
  python qmd_package.py --config my_config.yaml

  # With GPU acceleration
  python qmd_package.py --gpu

  # Run individual steps
  python qmd_package.py --step convert
  python qmd_package.py --step density --gpu
  python qmd_package.py --step export

Config file must contain:
  heavy_chain: "QVQLVQSGAEV..."
  light_chain: "DIQMTQSPSSL..."
  output_dir: "qmd_output"
  density_method: "gaussian"
  grid_spacing: 0.5
        """
    )

    parser.add_argument('--config', '-c', default='config.yaml',
                       help='Config file path (default: config.yaml)')
    parser.add_argument('--gpu', action='store_true',
                       help='Enable GPU acceleration')
    parser.add_argument('--step', choices=['convert', 'density', 'export', 'all'],
                       default='all', help='Run specific pipeline step')

    args = parser.parse_args()

    # Load required config file
    try:
        config = QMDPipeline.load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print("Please ensure a valid config file exists.")
        return 1

    # Initialize pipeline with config
    pipeline = QMDPipeline(use_gpu=args.gpu, config=config)

    if args.step == 'all':
        # Run full pipeline (will use config internally)
        try:
            results = pipeline.run_full_pipeline()
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    else:
        # Run individual steps
        print(f"Running step: {args.step}")
        output_dir = config.get('output_dir', 'qmd_output')

        if args.step == 'convert':
            # Validate required config values
            if not config.get('heavy_chain'):
                print("Error: 'heavy_chain' must be specified in config file")
                return 1
            if not config.get('light_chain'):
                print("Error: 'light_chain' must be specified in config file")
                return 1

            mmcif_filename = config.get('mmcif_filename', 'antibody.cif')
            mmcif_file = os.path.join(output_dir, mmcif_filename)
            Path(output_dir).mkdir(exist_ok=True)
            result = convert_antibody(config['heavy_chain'], config['light_chain'], mmcif_file, config)
            print(f"Generated: {result}")

        elif args.step == 'density':
            mmcif_filename = config.get('mmcif_filename', 'antibody.cif')
            h5_filename = config.get('density_h5_filename', 'density.h5')
            mmcif_file = os.path.join(output_dir, mmcif_filename)
            h5_file = os.path.join(output_dir, h5_filename)
            result = pipeline.density_calc.calculate_density(mmcif_file)
            print(f"Generated: {result}")

        elif args.step == 'export':
            h5_filename = config.get('density_h5_filename', 'density.h5')
            cube_filename = config.get('density_cube_filename', 'density.cube')
            npy_filename = config.get('density_npy_filename', 'density.npy')
            h5_file = os.path.join(output_dir, h5_filename)
            cube_file = os.path.join(output_dir, cube_filename)
            npy_file = os.path.join(output_dir, npy_filename)
            results = pipeline.exporter.export_density(
                h5_file, cube_file, npy_file, config.get('grid_spacing', 0.5)
            )
            for result in results:
                print(result)

    return 0


if __name__ == "__main__":
    main()
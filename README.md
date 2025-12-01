# Quantum Molecular Dynamics (QMD) Package

A high-performance quantum molecular dynamics simulation package for proteins with integrated visualization and analysis tools, featuring comprehensive electron density analysis and 3D visualization capabilities.

## Features

### Core MD Capabilities
- **Quantum Chemistry Integration**: PySCF interface for ab initio MD
- **Born-Oppenheimer Dynamics**: Velocity Verlet integrator
- **Protein Structure Handling**: PDB file support with MDAnalysis
- **Trajectory Analysis**: Energy plots, RMSD, temperature monitoring
- **Multiple DFT Methods**: B3LYP, PBE, PBE0, and more
- **Flexible Basis Sets**: From minimal to high-accuracy

### Advanced Visualization & Analysis
- **3D Electron Density Visualization**: Isosurfaces, volume rendering, interactive plots
- **Molecular Orbital Analysis**: HOMO/LUMO visualization with phase coloring
- **Density Comparison Tools**: Aligned molecular density comparisons
- **Interactive Visualizations**: Plotly-based rotatable, zoomable 3D plots
- **Cube File I/O**: Import/export standard quantum chemistry formats
- **Molecular Alignment**: Kabsch algorithm for optimal superposition

## Installation

```bash
# Clone repository
git clone <repository_url>
cd md

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

```python
from qmd import System, BornOppenheimerIntegrator
from qmd.quantum import PySCFCalculator

# Load protein structure
system = System("protein.pdb")

# Set up quantum calculator
calc = PySCFCalculator(method="B3LYP", basis="6-31G*")

# Run MD simulation
integrator = BornOppenheimerIntegrator(calc, timestep=0.5)
trajectory = integrator.run(system, steps=100)
```

## Examples

### Basic MD Simulation
See `examples/basic_md_simulation.py` for a complete water molecule simulation.

### Electron Density Analysis
```python
from qmd.utils import MolecularAligner, ElectronDensityCalculator
from qmd.visualization.density_plots import compare_densities

# Align molecules for comparison
aligner = MolecularAligner()
aligned_coords, R, rmsd = aligner.kabsch_align(coords1, coords2)

# Calculate densities on identical grids
calc = ElectronDensityCalculator(pyscf_calculator)
density1, grid_coords = calc.calculate_density_grid(system1)
density2, _ = calc.calculate_density_grid(system2)

# Compare and visualize
fig = compare_densities(density1, density2, grid_coords)
```

### 3D Visualization
```python
from qmd.visualization.density_3d import create_isosurface_matplotlib
from qmd.visualization.interactive_density import create_interactive_isosurface
from qmd.visualization.orbital_viz import MolecularOrbitalVisualizer

# 3D density isosurface
fig = create_isosurface_matplotlib(density, grid_coords)

# Interactive visualization
fig = create_interactive_isosurface(density, grid_coords)
fig.show()  # Opens in browser

# Molecular orbitals
orbital_viz = MolecularOrbitalVisualizer(calculator)
homo_fig, lumo_fig = orbital_viz.plot_frontier_orbitals(system)
```

### Cube File I/O
```python
from qmd.utils import save_density_cube, load_density_cube

# Export for VMD, GaussView, etc.
save_density_cube(system, density, grid_coords, "molecule.cube")

# Import from other calculations
atoms, density, coords = load_density_cube("external.cube")
```

## Complete Examples

- `examples/basic_md_simulation.py` - Water molecule MD simulation
- `examples/density_comparison_simple.py` - Simple density comparison
- `examples/electron_density_comparison.py` - Comprehensive density analysis
- `examples/density_3d_visualization.py` - Complete 3D visualization demo

## Requirements

### Core Dependencies
- Python ≥ 3.8
- NumPy, SciPy
- PySCF (quantum chemistry)
- ASE (atomic simulation environment)
- MDAnalysis (protein structures)
- Matplotlib (plotting)

### Additional Dependencies for 3D Visualization
```bash
pip install scikit-image plotly
```

- **scikit-image**: For marching cubes isosurface generation
- **plotly**: For interactive 3D visualizations

## Module Structure

```
qmd/
├── core/                    # Core MD functionality
│   ├── system.py           # System class, PDB handling
│   ├── atoms.py            # Atom and Molecule classes
│   └── integrator.py       # Born-Oppenheimer integrator
├── quantum/                 # Quantum chemistry interfaces
│   ├── calculator.py       # Abstract base class
│   ├── pyscf_interface.py  # PySCF implementation
│   └── dft_methods.py      # Method definitions
├── analysis/               # Analysis tools
│   ├── density_utils.py    # Density calculation & alignment
│   └── trajectory.py       # Trajectory analysis
├── visualization/          # Visualization modules
│   ├── viewer.py           # Basic molecular viewer
│   ├── density_plots.py    # 2D density plots
│   ├── density_3d.py       # 3D isosurfaces (matplotlib)
│   ├── interactive_density.py # Interactive plots (plotly)
│   └── orbital_viz.py      # Molecular orbital visualization
└── io/                     # File I/O
    └── cube_files.py       # Gaussian cube file support
```

## Advanced Capabilities

### Electron Density Analysis
- **Quantum-mechanical densities**: Calculate electron densities on 3D grids using PySCF
- **Molecular alignment**: Kabsch algorithm for optimal structural superposition
- **Density comparison**: Statistical analysis and visualization of density differences
- **Grid-based calculations**: Customizable resolution and boundaries

### 3D Visualization Options
1. **Matplotlib 3D**
   - Static isosurfaces with marching cubes
   - Multiple cross-sectional planes
   - Dual molecule comparisons

2. **Interactive Plotly**
   - Rotatable, zoomable isosurfaces
   - Volume rendering with transparency
   - Side-by-side comparisons
   - Animation support for trajectories

3. **Molecular Orbitals**
   - Individual orbital visualization
   - HOMO/LUMO analysis
   - Phase coloring (red/blue for ±)
   - Both static and interactive modes

### File Format Support
- **Gaussian cube files**: Standard quantum chemistry format
- **VMD DX format**: For molecular visualization software
- **PDB structures**: Protein and small molecule support
- **ASE compatibility**: Seamless integration with Atomic Simulation Environment

### Analysis Tools
- **RMSD calculations**: Before and after alignment
- **Density statistics**: Mean, maximum, correlation coefficients
- **Trajectory analysis**: Energy evolution, temperature monitoring
- **Comparison metrics**: Quantitative similarity measures

## Getting Started

1. **Install QMD**:
   ```bash
   git clone <repository_url>
   cd md
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Install visualization dependencies**:
   ```bash
   pip install scikit-image plotly
   ```

3. **Run examples**:
   ```bash
   # Basic MD simulation
   python examples/basic_md_simulation.py

   # Density comparison
   python examples/density_comparison_simple.py

   # Full 3D visualization demo
   python examples/density_3d_visualization.py
   ```

## License

MIT License
# Scalar Modeling and Environmental Simulation

Welcome to a multidimensional exploration of scalar frameworks, quantum mechanics, environmental data integration, and recursive systems. This repository is the foundation for a living simulation project that reimagines environmental monitoring through unified scalar modeling.

## 🧭 Project Overview

This initiative seeks to:
- Build and model **information pockets** for testing information retention
- Develop a **scalar modeling framework** integrating quantum, environmental, and human cycles
- Synthesize **transforms, fractals, and recursive structures** into cohesive modeling tools

## 📐 Core Components

- `scalar_simulation.py`: Core simulation engine for information persistence modeling
- `2025.11.29_RAW_Notebook.ipynb`: Interactive Jupyter notebook with visualization tools
- `observer_profiles.json`: **NEW!** Standardized configuration profiles representing different observer vantage points and experimental perspectives
- `profile_loader.py`: Utility module for loading and executing observer profiles
- `requirements.txt`: Python dependencies for running the simulation

### Planned Components
- `resonance_map/`: Spatial-temporal mapping of scalar behaviors across dimensions
- `fractal_transform.py`: Engine for recursive modeling of environmental fluctuations
- `simulation_core/`: Modular code representing dynamic virtual terrain features

## 🌍 Philosophical Foundations

This project is informed by:
- **Path integral formalism** and least-action principles in quantum systems
- The conceptual unity of **environmental field data** and recursive memory traces
- The search for **scalar harmony** across biological, physical, and social domains

## 🚧 Development Status

This repository is in its **active development stage**. Key accomplishments:
- ✅ Core simulation engine implemented (`scalar_simulation.py`)
- ✅ Interactive Jupyter notebook with visualization (`2025.11.29_RAW_Notebook.ipynb`)
- ✅ Vectorized batch simulation for efficient computation
- ✅ **Scalar modeling actively progressing in Google Colab** with interactive parameter exploration

Key goals:
- Formalize scalar data structures
- Prototype environmental model renderings
- Define recursive logic gates for system interactions

## 🚀 Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Running the Simulation

```python
from scalar_simulation import run_simulation_chunked, compute_mi_from_counts, compute_tau
import numpy as np

# Parameters
N = 21  # Number of nodes
M = 1000  # Ensemble size
T = 60  # Time steps
source_j = 10  # Source node index
beta = 1.5  # Inverse temperature
theta_arr = np.ones(N)

# Create coupling array with pocket
c_pocket = np.ones(N - 1)
c_pocket[7:13] = 10.0  # Strong pocket coupling

# Run simulation
counts = run_simulation_chunked(N, c_pocket, beta, theta_arr, M, T, source_j)
I = compute_mi_from_counts(counts, M)
tau = compute_tau(I, threshold=1e-3)
```

### Using Observer Profiles

Observer profiles provide standardized experimental configurations for reproducible research:

```python
from profile_loader import run_profile, list_profiles, compare_profiles

# List all available profiles
profiles = list_profiles()
print(f"Available profiles: {profiles}")

# Run a single profile
result = run_profile("single_pocket_strong", verbose=True)
print(f"Mean persistence time: {result['tau'].mean()}")

# Compare multiple perspectives
results = compare_profiles([
    "baseline_homogeneous",
    "single_pocket_weak", 
    "single_pocket_strong"
])

# Access results for analysis
for name, data in results.items():
    print(f"{name}: mean tau = {data['tau'].mean():.2f}")
```

**Available Profile Categories:**
- `baseline`: Homogeneous systems for control experiments
- `single_pocket` / `multi_pocket`: Localized information retention structures
- `thermodynamic`: Temperature/noise regime variations
- `spatial`: Edge effects and asymmetric propagation
- `temporal`: Continuous driving and steady-state dynamics
- `scale`: Large-system behavior and hierarchical structures
- `gradient`: Smooth spatial transitions

### Interactive Notebook

Open `2025.11.29_RAW_Notebook.ipynb` in Jupyter or Google Colab for interactive exploration with sliders.

**Scalar modeling is actively being developed in Google Colab**, enabling:
- Real-time parameter tuning and visualization
- Interactive exploration of coupling strengths, temperature parameters, and network topologies
- Dynamic visualization of information persistence across node networks

## 🔭 Future Directions

- Incorporate distributed sensor inputs
- Enable real-time simulation feedback loops
- Build collaborative models across observers and domains


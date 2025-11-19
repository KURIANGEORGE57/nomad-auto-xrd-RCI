## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/FAIRmat-NFDI/nomad-auto-xrd.git
cd nomad-auto-xrd

# Create virtual environment (Python 3.10, 3.11, or 3.12)
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with ARCO support
pip install -e '.[dev]'
```

### Minimal ARCO Example

```python
import numpy as np
from nomad_auto_xrd.lib import compute_arco_features

# Your XRD data (2θ angles and intensities)
two_theta = np.linspace(10, 80, 1024)  # degrees
intensity = your_xrd_data  # measured intensities

# Compute ARCO features
features = compute_arco_features(
    two_theta=two_theta,
    intensity=intensity,
    Qmax=40,     # Max rational denominator
    alpha=0.5    # Bandwidth scale
)

# View results
print(f"RCI (periodicity): {features['rci']:.4f}")
print(f"ARCO fingerprint: {len(features['arco_print'])} features")
print(f"Top rational: {features['top_rationals'][0]}")
```

**Expected Output:**
```
RCI (periodicity): 0.4521
ARCO fingerprint: 1576 features
Top rational: {'frequency': 0.0833, 'power': 0.0234, 'denominator': 12}
```

### Run Tests

```bash
# Quick smoke tests (ARCO only, ~10 seconds)
pytest tests/test_arco.py -v

# All tests including pipeline tests (~5 minutes)
RUN_PIPELINE_TESTS=true pytest -v

# With coverage
pytest --cov=src tests/
```

### Explore Examples

```bash
# Run the ARCO demo notebook
jupyter notebook notebooks/arco_xrd_demo.ipynb

# Or try the quickstart example
python examples/arco_quickstart.py
```

See the [ARCO Documentation](#arco-documentation) section below for detailed usage.

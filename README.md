# nomad-auto-xrd

[![Python Tests](https://github.com/FAIRmat-NFDI/nomad-auto-xrd/actions/workflows/actions.yml/badge.svg)](https://github.com/FAIRmat-NFDI/nomad-auto-xrd/actions/workflows/actions.yml)
[![ARCO Tests](https://github.com/FAIRmat-NFDI/nomad-auto-xrd/actions/workflows/arco-tests.yml/badge.svg)](https://github.com/FAIRmat-NFDI/nomad-auto-xrd/actions/workflows/arco-tests.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Nomad example template

This `nomad` plugin was generated with `Cookiecutter` along with `@nomad`'s [`cookiecutter-nomad-plugin`](https://github.com/FAIRmat-NFDI/cookiecutter-nomad-plugin) template.

## Features

### ARCO (Arcogram with Rational Coherence Index)

This plugin includes **ARCO**, an advanced algorithm for analyzing periodicity in XRD diffraction patterns. ARCO converts 1-D intensity profiles into interpretable fingerprints by integrating spectral power around rational frequency anchors.

**Key features:**
- **RCI (Rational Coherence Index)**: Scalar metric (0-1) quantifying periodic structure strength
- **ARCO-print**: Fixed-length feature vector for ML applications and similarity search
- **ARCO-3D**: Position-resolved arcogram for localizing structural changes
- **Top Rationals**: Interpretable frequency components corresponding to lattice periodicities

**Applications:**
- Phase identification via similarity-based retrieval
- Quality control and crystallinity assessment
- Structure optimization in inverse design
- Peak spacing analysis for characteristic periodicities

See the [ARCO Documentation](#arco-documentation) section below for detailed usage instructions.


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
RUN_SMOKE_ONLY=true pytest tests/test_arco.py -m smoke -v

# All ARCO tests (~2 minutes)
pytest tests/test_arco.py -v

# All tests including pipeline tests (~5 minutes)
RUN_PIPELINE_TESTS=true pytest -v
```

### Explore Examples

```bash
# Run the ARCO demo notebook
jupyter notebook notebooks/arco_xrd_demo.ipynb

# Or try the quickstart example
python examples/arco_quickstart.py
```


## Development

If you want to develop locally this plugin, clone the project and in the plugin folder, create a virtual environment (you can use Python 3.9, 3.10, or 3.11):
```sh
git clone https://github.com/foo/nomad-auto-xrd.git
cd nomad-auto-xrd
python3.11 -m venv .pyenv
. .pyenv/bin/activate
```

Make sure to have `pip` upgraded:
```sh
pip install --upgrade pip
```

We recommend installing `uv` for fast pip installation of the packages:
```sh
pip install uv
```

Install the `nomad-lab` package:
```sh
uv pip install '.[dev]' --index-url https://gitlab.mpcdf.mpg.de/api/v4/projects/2187/packages/pypi/simple
```

**Note!**
Until we have an official pypi NOMAD release with the plugins functionality make
sure to include NOMAD's internal package registry (via `--index-url` in the above command).

The plugin is still under development. If you would like to contribute, install the package in editable mode (with the added `-e` flag):
```sh
uv pip install -e '.[dev]' --index-url https://gitlab.mpcdf.mpg.de/api/v4/projects/2187/packages/pypi/simple
```


### Run the tests

You can run locally the tests:
```sh
python -m pytest -sv tests
```

where the `-s` and `-v` options toggle the output verbosity.

Our CI/CD pipeline produces a more comprehensive test report using the `pytest-cov` package. You can generate a local coverage report:
```sh
uv pip install pytest-cov
python -m pytest --cov=src tests
```

By default, the tests related to training and inference of the models are
skipped. If you want to execute them, set the environment variable
`RUN_PIPELINE_TESTS` before running the tests.
```sh
export RUN_PIPELINE_TESTS=true
```

### Run linting and auto-formatting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting the code. Ruff auto-formatting is also a part of the GitHub workflow actions. You can run locally:
```sh
ruff check .
ruff format . --check
```


### Debugging

For interactive debugging of the tests, use `pytest` with the `--pdb` flag. We recommend using an IDE for debugging, e.g., _VSCode_. If that is the case, add the following snippet to your `.vscode/launch.json`:
```json
{
  "configurations": [
      {
        "name": "<descriptive tag>",
        "type": "debugpy",
        "request": "launch",
        "cwd": "${workspaceFolder}",
        "program": "${workspaceFolder}/.pyenv/bin/pytest",
        "justMyCode": true,
        "env": {
            "_PYTEST_RAISE": "1"
        },
        "args": [
            "-sv",
            "--pdb",
            "<path-to-plugin-tests>",
        ]
    }
  ]
}
```

where `<path-to-plugin-tests>` must be changed to the local path to the test module to be debugged.

The settings configuration file `.vscode/settings.json` automatically applies the linting and formatting upon saving the modified file.


### Documentation on Github pages

To view the documentation locally, install the related packages using:
```sh
uv pip install -r requirements_docs.txt
```

Run the documentation server:
```sh
mkdocs serve
```


## Adding this plugin to NOMAD

Currently, NOMAD has two distinct flavors that are relevant depending on your role as an user:
1. [A NOMAD Oasis](#adding-this-plugin-in-your-nomad-oasis): any user with a NOMAD Oasis instance.
2. [Local NOMAD installation and the source code of NOMAD](#adding-this-plugin-in-your-local-nomad-installation-and-the-source-code-of-nomad): internal developers.

### Adding this plugin in your NOMAD Oasis

Read the [NOMAD plugin documentation](https://nomad-lab.eu/prod/v1/staging/docs/howto/oasis/plugins_install.html) for all details on how to deploy the plugin on your NOMAD instance.

### Adding this plugin in your local NOMAD installation and the source code of NOMAD

Modify the text file under `/nomad/default_plugins.txt` and add:
```sh
<other-content-in-default_plugins.txt>
nomad-auto-xrd==x.y.z
```
where `x.y.z` represents the released version of this plugin.

Then, go to your NOMAD folder, activate your NOMAD virtual environment and run:
```sh
deactivate
cd <route-to-NOMAD-folder>/nomad
source .pyenv/bin/activate
./scripts/setup_dev_env.sh
```

Alternatively and only valid for your local NOMAD installation, you can modify `nomad.yaml` to include this plugin, see [NOMAD Oasis - Install plugins](https://nomad-lab.eu/prod/v1/staging/docs/howto/oasis/plugins_install.html).


### Build the python package

The `pyproject.toml` file contains everything that is necessary to turn the project
into a pip installable python package. Run the python build tool to create a package distribution:

```sh
pip install build
python -m build --sdist
```

You can install the package with pip:

```sh
pip install dist/nomad-auto-xrd-0.1.0
```

Read more about python packages, `pyproject.toml`, and how to upload packages to PyPI
on the [PyPI documentation](https://packaging.python.org/en/latest/tutorials/packaging-projects/).


### Template update

We use cruft to update the project based on template changes. A `cruft-update.yml` is included in Github workflows to automatically check for updates and create pull requests to apply updates. Follow the [instructions](https://github.blog/changelog/2022-05-03-github-actions-prevent-github-actions-from-creating-and-approving-pull-requests/) on how to enable Github Actions to create pull requests.

To run the check for updates locally, follow the instructions on [`cruft` website](https://cruft.github.io/cruft/#updating-a-project).


## Main contributors
| Name | E-mail     |
|------|------------|
| Pepe Márquez | [jose.marquez@physik.hu-berlin.de](mailto:jose.marquez@physik.hu-berlin.de)


## ARCO Documentation

### Overview

ARCO (Arcogram with Rational Coherence Index) is a signal processing algorithm that analyzes periodicity in XRD patterns by computing spectral power around **rational frequency anchors** (e.g., 1/2, 1/3, 1/7, etc.). These rationals correspond to small-integer periodicities commonly found in crystalline materials.

### Quick Start

```python
from nomad_auto_xrd.lib import compute_arco_features

# Compute ARCO features from XRD pattern
features = compute_arco_features(
    two_theta=two_theta_array,
    intensity=intensity_array,
    Qmax=40,           # Max denominator for rational anchors
    alpha=0.5,         # Bandwidth scale factor
    major_q=20         # Threshold for "major" rationals
)

print(f"RCI (periodicity score): {features['rci']:.4f}")
print(f"ARCO fingerprint dimension: {len(features['arco_print'])}")
print(f"Top rational frequencies: {features['top_rationals'][:3]}")
```

### Key Outputs

| Output | Type | Description |
|--------|------|-------------|
| **RCI** | float | Rational Coherence Index (0-1). Higher = stronger periodicity |
| **arco_print** | array | Fixed-length feature vector for ML/similarity |
| **top_rationals** | list | Top-k dominant rational frequencies |
| **arco_3d** | array | Position-resolved periodicity map |

### Parameter Guidelines for XRD

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `Qmax` | 30-60 | Maximum denominator for rational anchors. Higher captures longer periodicities |
| `window_sizes` | [128, 256] | Sliding window sizes for multi-scale analysis |
| `alpha` | 0.3-1.0 | Bandwidth scale (δ = α/q²). Lower = narrower frequency bands |
| `major_q` | 10-20 | Denominator threshold for RCI computation |

### Integration with XRD Analysis

To compute ARCO features during analysis:

```python
from nomad_auto_xrd.common.arco_integration import attach_arco_to_result

# After running XRD analysis
result = analyzer.eval(analysis_inputs)

# Attach ARCO features
result = attach_arco_to_result(
    result,
    analysis_inputs,
    enable_arco=True,
    Qmax=40,
    alpha=0.5
)

# Access ARCO results
for idx, arco in enumerate(result.arco_features):
    print(f"Pattern {idx}: RCI = {arco['rci']:.4f}")
```

### Rational Frequency Interpretation

Common rational frequencies and their physical meanings:

| Rational | Frequency | Period | Physical Interpretation |
|----------|-----------|--------|------------------------|
| 1/2 | 0.500 | 2 | Alternating pattern / β-strand-like |
| 1/3 | 0.333 | 3 | Collagen-like repeat |
| 1/4 | 0.250 | 4 | Quarter repeats |
| 1/7 | 0.143 | 7 | Heptad repeat / coiled-coil |
| 1/10 | 0.100 | 10 | Long-range periodicity |
| 5/18 | 0.278 | ~3.6 | α-helix-like (protein analogy) |

### Demo Notebook

See `notebooks/arco_xrd_demo.ipynb` for a comprehensive demonstration including:
- Generating rational anchors
- Analyzing synthetic XRD patterns (uniform vs random peaks)
- Computing RCI and ARCO-prints
- Visualizing arc power spectra
- Multi-track analysis (intensity + derivative)
- Position-resolved ARCO-3D heatmaps
- Statistical validation with z-scores

### Advanced Usage

#### Multi-track Analysis

Use both intensity and derivative tracks for richer features:

```python
from nomad_auto_xrd.lib import XRDArcoAnalyzer

analyzer = XRDArcoAnalyzer(
    Qmax=40,
    window_sizes=[128, 256],
    alpha=0.5,
    use_derivative=True  # Include derivative track
)

result = analyzer.analyze_pattern(two_theta, intensity)
```

#### Similarity Computation

Compare ARCO fingerprints for pattern matching:

```python
from nomad_auto_xrd.lib import XRDArcoAnalyzer

analyzer = XRDArcoAnalyzer()

# Compute fingerprints
result1 = analyzer.analyze_pattern(two_theta1, intensity1)
result2 = analyzer.analyze_pattern(two_theta2, intensity2)

# Compute similarity (L1 distance, lower = more similar)
similarity = analyzer.compute_similarity(
    result1['arco_print'],
    result2['arco_print']
)

print(f"L1 distance: {similarity:.4f}")
```

#### Statistical Validation

Test significance against null model:

```python
from nomad_auto_xrd.lib import ARCO, generate_anchors

anchors = generate_anchors(Qmax=40)
arco = ARCO(anchors, window_sizes=[128])

# Compute z-score vs shuffled null
zscore = arco.null_model_zscore(
    intensity,
    n_shuffles=50,
    preserve_composition=True
)

print(f"Z-score: {zscore:.2f}")
if zscore > 3:
    print("Highly significant periodicity (z > 3)")
```

### Testing

Run ARCO-specific tests:

```bash
# Install test dependencies
pip install numpy pytest

# Run ARCO unit tests
PYTHONPATH=src python -m pytest tests/test_arco.py -v

# Run specific test categories
pytest tests/test_arco.py::TestSinePattern -v
pytest tests/test_arco.py::TestXRDIntegration -v
```

### API Reference

#### `generate_anchors(Qmax: int) -> List[float]`
Generate rational frequency anchors up to denominator Qmax.

#### `XRDArcoAnalyzer`
High-level interface for XRD pattern analysis.
- `__init__(Qmax, window_sizes, alpha, major_q, use_derivative)`
- `analyze_pattern(two_theta, intensity) -> Dict`
- `compute_similarity(arco_print1, arco_print2) -> float`

#### `ARCO`
Core ARCO algorithm implementation.
- `compute_rci(tracks, major_q) -> float`
- `compute_arco_print(tracks) -> ndarray`
- `compute_arco_3d(tracks, finest_window) -> ndarray`
- `compute_track_arcs(track_signal, window_size) -> ndarray`
- `null_model_zscore(track, n_shuffles) -> float`

#### `compute_arco_features(two_theta, intensity, ...) -> Dict`
Convenience function for one-shot ARCO analysis.

### Performance

- **Computation time**: ~50-200ms per pattern (2048 points, Qmax=40, 2 window sizes)
- **Memory**: ARCO-print dimensions = n_tracks × n_window_sizes × n_anchors
  - Example: 2 tracks × 2 windows × 140 anchors = 560 dimensions
- **Parallelization**: Vectorized with NumPy; GPU acceleration possible via PyTorch

### Citation

If you use ARCO in your research, please cite:

```
ARCO (Arcogram with Rational Coherence Index)
Implementation in nomad-auto-xrd plugin
https://github.com/FAIRmat-NFDI/nomad-auto-xrd
```

### References

- Rational (Diophantine) pooling provides semantically meaningful frequency decomposition
- Gaussian weighting around rational anchors with bandwidth δ = α/q²
- RCI measures fraction of spectral energy in low-denominator rationals
- Applications to protein sequences, ECG analysis, and now XRD patterns

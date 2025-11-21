# nomad-auto-xrd

A NOMAD plugin for automatic XRD pattern analysis with integrated **ARCO/RCI** (Arcogram with Rational Coherence Index) for advanced periodicity detection and fingerprinting.

This `nomad` plugin was generated with `Cookiecutter` along with `@nomad`'s [`cookiecutter-nomad-plugin`](https://github.com/FAIRmat-NFDI/cookiecutter-nomad-plugin) template.

## Features

- ✅ Automatic XRD phase identification using deep learning models
- ✅ **ARCO/RCI analysis** for quantifying crystallinity and periodicity
- ✅ Multi-pattern batch analysis with visualization
- ✅ Integration with NOMAD data infrastructure
- ✅ Temporal workflow support for scalable processing
- ✅ W&B logging for model training

## ARCO/RCI: Novel Periodicity Analysis

This repository implements **ARCO** (Arcogram with Rational Coherence) and **RCI** (Rational Coherence Index), a novel method for analyzing periodic patterns in XRD diffraction profiles.

### What is ARCO/RCI?

ARCO converts 1D XRD intensity profiles into interpretable fingerprints by:
1. Computing local power spectra using sliding windows
2. Integrating spectral power around **rational frequency anchors** (Farey sequence)
3. Producing:
   - **ARCO-print**: Fixed-length feature vector for ML and similarity search
   - **RCI**: Scalar index [0,1] quantifying crystallinity/periodicity

### Why Rational Frequencies?

Rational frequencies (e.g., 1/3, 1/7, 1/20) correspond to integer periodicities in:
- **Regular lattice spacing** → periodic diffraction peaks
- **Peak spacing relationships** → crystalline order
- **Long-range structural patterns** → material quality

**Benefits:**
- **Interpretable**: Each anchor maps to a physical period
- **Robust**: Resistant to noise and peak jitter (Gaussian weighting)
- **Compact**: Fixed-length fingerprint (vs. variable-length raw patterns)

### Quick Start with ARCO

```python
from nomad_auto_xrd.common.arco_rci import ARCO, generate_anchors

# Generate rational anchors (Qmax=40 for XRD)
anchors = generate_anchors(Qmax=40)

# Initialize analyzer
arco = ARCO(anchors, window_sizes=[128, 256], alpha=1.0)

# Compute features
arco_print = arco.compute_arco_print({'intensity': intensity_array})
rci = arco.compute_rci({'intensity': intensity_array}, major_q=20)

print(f"RCI (crystallinity): {rci:.4f}")  # Higher = more crystalline
print(f"ARCO-print shape: {arco_print.shape}")  # Feature vector for ML
```

### RCI Interpretation for XRD

| RCI Range | Interpretation |
|-----------|----------------|
| 0.7 - 1.0 | Highly crystalline / ordered |
| 0.4 - 0.7 | Partially crystalline |
| 0.2 - 0.4 | Weakly ordered |
| 0.0 - 0.2 | Amorphous / disordered |

### Documentation

- **Comprehensive Guide**: [`docs/explanation/arco_rci_guide.md`](docs/explanation/arco_rci_guide.md)
- **XRD Demo Notebook**: [`notebooks/arco_xrd_demo.ipynb`](notebooks/arco_xrd_demo.ipynb)
- **Sequence Demo Notebook**: [`notebooks/arco_sequence_demo.ipynb`](notebooks/arco_sequence_demo.ipynb)
- **API Documentation**: See guide for full API reference

### Integration in Analysis Pipeline

ARCO/RCI is automatically computed during XRD analysis:

```python
from nomad_auto_xrd.common.analysis import XRDAutoAnalyzer

analyzer = XRDAutoAnalyzer(working_directory, settings, enable_arco=True)
results = analyzer.eval(analysis_inputs)

# Access ARCO features
for rci, arco_print in zip(results.rci_values, results.arco_prints):
    print(f"Pattern RCI: {rci:.4f}")
    # Use arco_print for similarity search, ML, etc.
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

# ARCO/RCI User Guide

## Overview

**ARCO** (Arcogram with Rational Coherence) and **RCI** (Rational Coherence Index) are novel signal processing methods for analyzing periodic patterns in 1D signals, particularly X-ray diffraction (XRD) intensity profiles.

### Core Concept

ARCO converts 1D signals into fixed-length, interpretable fingerprints by:
1. Computing local power spectra (STFT-style sliding windows with Hann tapering)
2. Integrating spectral power around **rational-frequency anchors** (Farey/Diophantine set) rather than arbitrary linear bins
3. Producing:
   - **ARCO-print**: Fixed-length feature vector for ML/similarity search
   - **RCI**: Scalar index [0,1] measuring fraction of energy on major rationals
   - **ARCO-3D**: Position-resolved arcogram for localization

### Why Rational Anchors?

Low-denominator rationals (e.g., 1/3, 1/7, 1/20) correspond to small-integer periodicities seen in:
- **XRD patterns**: Regular lattice spacing → periodic peak positions
- **Biological sequences**: Heptad repeats (coiled-coils), α-helix (3.6 residues/turn)
- **ECG signals**: Beat harmonics and rhythm patterns

Aggregating energy at these rationals provides:
- **Semantic meaning**: Each anchor corresponds to a physical period
- **Robustness**: Resistant to frequency jitter and noise (Gaussian weighting)
- **Interpretability**: Direct mapping from features to periodicities

---

## Installation & Setup

### Requirements

```python
numpy>=1.20
```

### Import

```python
from nomad_auto_xrd.common.arco_rci import ARCO, generate_anchors, get_anchor_interpretation_table
```

---

## Quick Start

### 1. Generate Rational Anchors

```python
# For XRD: use Qmax=40 (longer periodicities)
anchors = generate_anchors(Qmax=40)
print(f"Generated {len(anchors)} anchors")  # ~174 anchors
```

### 2. Initialize ARCO Analyzer

```python
arco = ARCO(
    anchors=anchors,
    window_sizes=[128, 256],  # Multi-scale analysis
    hop_fraction=0.25,        # 75% overlap
    alpha=1.0,                # Bandwidth = alpha / q^2
    apply_hann=True           # Hann window tapering
)
```

### 3. Compute Features

```python
# Single-track analysis (XRD intensity)
intensity = [...] # your XRD intensity array

# ARCO-print: fixed-length fingerprint
arco_print = arco.compute_arco_print({'intensity': intensity})

# RCI: rational coherence index [0, 1]
rci = arco.compute_rci({'intensity': intensity}, major_q=20)

print(f"ARCO-print shape: {arco_print.shape}")  # (n_anchors * n_window_sizes,)
print(f"RCI: {rci:.4f}")  # Higher = more periodic
```

---

## API Reference

### `generate_anchors(Qmax: int) -> List[float]`

Generate Farey sequence of rational frequencies.

**Parameters:**
- `Qmax`: Maximum denominator (e.g., 11 for sequences, 40 for XRD)

**Returns:**
- List of rational anchor frequencies in (0, 0.5)

**Example:**
```python
anchors = generate_anchors(Qmax=20)
# Returns: [1/20, 1/19, ..., 9/19, 9/20]
```

---

### `class ARCO`

Main analyzer class for ARCO/RCI computation.

#### `__init__(anchors, window_sizes, hop_fraction, alpha, apply_hann, track_names)`

**Parameters:**
- `anchors` *(List[float])*: Rational frequency anchors from `generate_anchors()`
- `window_sizes` *(List[int])*: Window sizes for multi-scale analysis (default: [31, 63])
- `hop_fraction` *(float)*: Hop as fraction of window size (default: 0.25)
- `alpha` *(float)*: Bandwidth scaling factor, δ = α/q² (default: 1.0)
- `apply_hann` *(bool)*: Apply Hann window tapering (default: True)
- `track_names` *(List[str], optional)*: Names for multi-track analysis

#### `compute_arco_print(tracks: Dict[str, np.ndarray]) -> np.ndarray`

Compute ARCO-print: fixed-length fingerprint vector.

**Parameters:**
- `tracks`: Dictionary mapping track names to 1D signal arrays

**Returns:**
- 1D array of shape `(n_tracks * n_anchors * n_window_sizes,)`

**Example:**
```python
tracks = {'intensity': intensity_array}
arco_print = arco.compute_arco_print(tracks)
```

#### `compute_rci(tracks: Dict[str, np.ndarray], major_q: int = 11) -> float`

Compute Rational Coherence Index (RCI).

**Parameters:**
- `tracks`: Dictionary of signal arrays
- `major_q`: Max denominator for "major" rational arcs (default: 11)

**Returns:**
- RCI value in [0, 1]

**Interpretation:**
- **High RCI (>0.5)**: Strong periodicity
- **Medium RCI (0.2-0.5)**: Moderate periodicity
- **Low RCI (<0.2)**: Weak/no periodicity

**Example:**
```python
rci = arco.compute_rci({'intensity': intensity}, major_q=20)
```

#### `compute_arco_3d(tracks, finest_window) -> np.ndarray`

Compute position-resolved ARCO (ARCO-3D).

**Parameters:**
- `tracks`: Dictionary of signal arrays
- `finest_window`: Window size for position resolution

**Returns:**
- 3D array of shape `(n_tracks, n_windows, n_anchors)`

**Use case:** Visualize how periodicity varies across the signal

#### `null_model_zscore(track, n_shuffles, preserve_composition) -> Tuple[float, float]`

Compute z-score vs shuffled null distribution.

**Parameters:**
- `track`: 1D signal array
- `n_shuffles`: Number of shuffles (default: 50)
- `preserve_composition`: Preserve value histogram (default: True)

**Returns:**
- `(z_score, p_value)`

**Interpretation:**
- **z > 2**: Significant periodicity (p < 0.05)
- **z < 2**: Not significantly different from random

---

## Parameter Guidelines

### Domain-Specific Recommendations

| Domain | Qmax | Window Sizes | Alpha | Major_q |
|--------|------|-------------|-------|---------|
| **XRD patterns** | 40-60 | [128, 256] | 1.0-2.0 | 20 |
| **Protein sequences** | 11-20 | [31, 63] | 1.0 | 11 |
| **ECG signals** | 30 | [500, 1000]* | 1.0-2.0 | 20 |
| **General** | 20 | [32, 64, 128] | 1.0 | 11 |

\* For ECG at 250-500 Hz sampling

### Parameter Effects

#### `Qmax` (Anchor Density)
- **Low (11)**: Fewer anchors, captures only strong low-period patterns
- **Medium (20)**: Balanced for most applications
- **High (40-60)**: Dense sampling, captures longer periodicities (XRD)

#### `alpha` (Bandwidth Scaling)
- **Low (0.5)**: Narrow bands, high frequency resolution, requires strong signal
- **Medium (1.0)**: Recommended default
- **High (2.0)**: Wide bands, more robust to noise but lower resolution

#### `window_sizes` (Multi-scale)
- Small windows: Capture short-period variations
- Large windows: Capture long-period trends
- Multiple sizes: Robustness across scales

#### `major_q` (RCI Threshold)
- Determines which rationals count as "major" for RCI
- **XRD**: Use 20 (longer periods matter)
- **Sequences**: Use 11 (biological repeats 3-11)

---

## Interpretation Guide

### Common Rational Anchors

| Rational | Frequency | Period | Typical Meaning (XRD) | Typical Meaning (Bio) |
|----------|-----------|--------|----------------------|----------------------|
| 1/2 | 0.500 | 2 | Alternating peaks | β-strand |
| 1/3 | 0.333 | 3 | Triplet spacing | Collagen (Gly-X-Y) |
| 1/7 | 0.143 | 7 | - | Heptad (coiled-coil) |
| 1/10 | 0.100 | 10 | Decagonal | - |
| 1/20 | 0.050 | 20 | Long-range order | - |
| 5/18 | 0.278 | ~3.6 | - | α-helix pitch |

### RCI Interpretation for XRD

```python
if rci > 0.7:
    print("Highly crystalline / ordered")
elif rci > 0.4:
    print("Partially crystalline")
elif rci > 0.2:
    print("Weakly ordered")
else:
    print("Amorphous / disordered")
```

---

## Integration with XRD Analysis

ARCO/RCI is automatically computed in the XRD analysis pipeline when `enable_arco=True` (default).

### Accessing ARCO Features

```python
from nomad_auto_xrd.common.analysis import XRDAutoAnalyzer

analyzer = XRDAutoAnalyzer(
    working_directory='/path/to/work',
    analysis_settings=settings,
    enable_arco=True  # Default
)

results = analyzer.eval(analysis_inputs)

# Access ARCO features
for arco_print, rci in zip(results.arco_prints, results.rci_values):
    print(f"RCI: {rci:.4f}")
    print(f"ARCO-print length: {len(arco_print)}")
```

### Use Cases

1. **Similarity Search**: Use `arco_print` as feature vector in FAISS/vector DB
2. **Crystallinity Quantification**: Use `rci` as crystallinity metric
3. **Phase Classification**: Train ML model on `arco_print` features
4. **Quality Control**: Flag patterns with anomalous RCI values
5. **Evolutionary Optimization**: Add RCI as constraint/objective in EvolvMorph

---

## Advanced Usage

### Multi-Track Analysis

```python
# Example: Intensity + Derivative
intensity = np.array([...])
derivative = np.gradient(intensity)

tracks = {
    'intensity': intensity,
    'derivative': derivative
}

arco_print = arco.compute_arco_print(tracks)
rci = arco.compute_rci(tracks)
```

### ARCO-3D Visualization

```python
import matplotlib.pyplot as plt

arco_3d = arco.compute_arco_3d({'intensity': intensity}, finest_window=128)

plt.imshow(arco_3d[0].T, aspect='auto', cmap='viridis')
plt.xlabel('Window Position')
plt.ylabel('Rational Anchor')
plt.title('ARCO-3D: Position-Resolved Periodicity')
plt.colorbar(label='Arc Power')
plt.show()
```

### Statistical Validation

```python
z_score, p_value = arco.null_model_zscore(
    intensity,
    n_shuffles=100,
    preserve_composition=True
)

if z_score > 2.0:
    print(f"Significant periodicity detected (z={z_score:.2f}, p={p_value:.4f})")
else:
    print(f"No significant periodicity (z={z_score:.2f})")
```

---

## Performance Considerations

### Computational Cost

- **FFT per window**: O(W log W)
- **Total for signal length N**: O((N/H) * W log W) where H = hop size
- **Memory**: ARCO-print is ~100-500 floats (tiny compared to raw signal)

### Optimization Tips

1. **Use larger hop** (`hop_fraction=0.5`) for faster computation (less overlap)
2. **Limit window sizes** to 1-2 scales if speed is critical
3. **Precompute anchors** once and reuse across signals
4. **Batch processing**: Process multiple patterns in parallel

### Typical Performance

- **Single XRD pattern** (2048 points): ~50-100ms
- **100 patterns**: ~5-10 seconds
- **Memory per pattern**: <1 MB

---

## Examples

See the notebooks directory for complete examples:
- `notebooks/arco_xrd_demo.ipynb`: XRD pattern analysis
- `notebooks/arco_sequence_demo.ipynb`: Protein sequence analysis

---

## Troubleshooting

### Issue: RCI is always low

**Solution:**
- Check signal length: Should be at least 2× largest window size
- Verify signal has actual periodicity (visual inspection)
- Try increasing `alpha` for more bandwidth
- Check if signal needs resampling to uniform grid

### Issue: ARCO-print has NaN values

**Solution:**
- Signal may have constant regions (zero std)
- Add small noise: `signal += 1e-6 * np.random.randn(len(signal))`
- Check for invalid values (inf, nan) in input

### Issue: Computation is slow

**Solution:**
- Reduce `Qmax` (fewer anchors)
- Increase `hop_fraction` (less overlap)
- Use fewer/smaller window sizes

---

## Citation

If you use ARCO/RCI in your research, please cite:

```bibtex
@software{nomad_auto_xrd_rci,
  title = {NOMAD Auto-XRD with ARCO/RCI},
  author = {FAIRmat Consortium},
  year = {2025},
  url = {https://github.com/FAIRmat-NFDI/nomad-auto-xrd}
}
```

---

## References

- Farey sequence: Mathematical foundation for rational frequency sampling
- STFT: Short-Time Fourier Transform for windowed spectral analysis
- Diophantine approximation: Theory of rational approximations to real numbers

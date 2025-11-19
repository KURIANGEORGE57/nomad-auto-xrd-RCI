# ARCO Examples

This directory contains example scripts and notebooks demonstrating ARCO usage.

## Contents

### 📄 Scripts

#### `arco_quickstart.py`
Complete quickstart example demonstrating:
- Synthetic XRD pattern generation
- ARCO feature computation
- Result interpretation
- Visualization

**Run:**
```bash
python examples/arco_quickstart.py
```

**Output:**
- Console summary with RCI values and top rationals
- `arco_quickstart_output.png` visualization (if matplotlib available)

---

### 📓 Notebooks

#### `arco_example.ipynb`
Interactive notebook for quick pattern analysis:
- Load XRD data from CSV
- Compute ARCO features
- Visualize fingerprints
- Try your own data

**Run:**
```bash
jupyter notebook examples/arco_example.ipynb
```

**For comprehensive tutorial**, see `notebooks/arco_xrd_demo.ipynb` in the main directory.

---

### 📊 Sample Data

#### `data/sample_xrd_pattern.csv`
Synthetic XRD pattern with periodic peaks:
- **Format:** 2-column CSV (two_theta, intensity)
- **Range:** 10° - 80° 2θ
- **Points:** 141
- **Features:** Multiple Gaussian peaks with periodic spacing

**Load in Python:**
```python
import pandas as pd
df = pd.read_csv('examples/data/sample_xrd_pattern.csv')
two_theta = df['two_theta'].values
intensity = df['intensity'].values
```

---

## Quick Examples

### 1. Basic Usage

```python
from nomad_auto_xrd.lib import compute_arco_features
import numpy as np

# Your data
two_theta = np.linspace(10, 80, 1024)
intensity = your_xrd_intensity

# Compute
features = compute_arco_features(two_theta, intensity, Qmax=40)
print(f"RCI: {features['rci']:.4f}")
```

### 2. Compare Two Patterns

```python
from nomad_auto_xrd.lib import XRDArcoAnalyzer

analyzer = XRDArcoAnalyzer(Qmax=40)

# Compute fingerprints
result1 = analyzer.analyze_pattern(two_theta1, intensity1)
result2 = analyzer.analyze_pattern(two_theta2, intensity2)

# Compare
similarity = analyzer.compute_similarity(
    result1['arco_print'],
    result2['arco_print']
)
print(f"Similarity (L1 distance): {similarity:.2f}")
```

### 3. Statistical Validation

```python
from nomad_auto_xrd.lib import ARCO, generate_anchors

anchors = generate_anchors(Qmax=40)
arco = ARCO(anchors, window_sizes=[128])

# Compute z-score
zscore = arco.null_model_zscore(
    intensity,
    n_shuffles=50
)

if zscore > 3:
    print("Significant periodicity detected!")
```

---

## Parameter Guidelines

| Parameter | Example Values | When to Use |
|-----------|----------------|-------------|
| **Qmax** | 30, 40, 60 | Higher for long-range periodicities |
| **alpha** | 0.3, 0.5, 1.0 | Lower for sharp peaks, higher for broad |
| **major_q** | 10, 20, 30 | Typically Qmax/2 for balanced discrimination |
| **window_sizes** | [128, 256] | Based on pattern resolution |

---

## Expected Results

### High RCI (> 0.7)
- Strong periodic structure
- Crystalline material
- Uniform peak spacing

### Medium RCI (0.4 - 0.7)
- Moderate periodicity
- Semi-crystalline
- Some structural order

### Low RCI (< 0.4)
- Weak periodicity
- Amorphous material
- Random/disordered structure

---

## Troubleshooting

**Issue:** RCI always close to 1.0
- **Solution:** Reduce `major_q` parameter (try major_q ≈ Qmax/2)

**Issue:** RCI doesn't discriminate patterns
- **Solution:** Adjust `alpha` (try 0.3 for sharp peaks, 1.0 for broad features)

**Issue:** Top rationals don't make sense
- **Solution:** Check if pattern is truly periodic; try different window_sizes

**Issue:** Very long computation time
- **Solution:** Reduce Qmax or use fewer window_sizes

---

## Next Steps

1. ✅ Run `arco_quickstart.py` to see basic usage
2. 📓 Open `arco_example.ipynb` for interactive exploration
3. 📚 Read `notebooks/arco_xrd_demo.ipynb` for comprehensive tutorial
4. 📖 See main `README.md` for full documentation
5. 🧪 Run tests: `pytest tests/test_arco.py -v`

---

## Questions?

- **Documentation:** See `README.md` [ARCO Documentation](#arco-documentation) section
- **Validation Report:** See `ARCO_VALIDATION_REPORT.md`
- **Issues:** https://github.com/FAIRmat-NFDI/nomad-auto-xrd/issues

#!/usr/bin/env python3
"""
ARCO Quickstart Example
=======================

This script demonstrates basic ARCO usage on synthetic XRD data.
Run: python examples/arco_quickstart.py
"""

import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from nomad_auto_xrd.lib import compute_arco_features, XRDArcoAnalyzer

print("="*70)
print(" ARCO Quickstart Example")
print("="*70)

# 1. Generate synthetic XRD pattern
print("\n[1] Generating synthetic XRD pattern...")

two_theta = np.linspace(10, 80, 2048)  # 2θ angles (degrees)

# Create pattern with uniform peak spacing (periodic)
intensity_periodic = np.zeros_like(two_theta)
for peak_pos in np.arange(15, 75, 5):  # Peaks every 5 degrees
    intensity_periodic += 100 * np.exp(-((two_theta - peak_pos)**2) / 0.8)
intensity_periodic += 20 + 5 * np.random.randn(len(two_theta))

# Create pattern with random peak spacing (non-periodic)
intensity_random = np.zeros_like(two_theta)
for peak_pos in np.random.uniform(15, 75, 12):
    intensity_random += 100 * np.exp(-((two_theta - peak_pos)**2) / 0.8)
intensity_random += 20 + 5 * np.random.randn(len(two_theta))

print(f"  ✓ Created 2 synthetic patterns ({len(two_theta)} points each)")

# 2. Compute ARCO features
print("\n[2] Computing ARCO features...")

features_periodic = compute_arco_features(
    two_theta=two_theta,
    intensity=intensity_periodic,
    Qmax=40,
    alpha=0.5,
    major_q=20
)

features_random = compute_arco_features(
    two_theta=two_theta,
    intensity=intensity_random,
    Qmax=40,
    alpha=0.5,
    major_q=20
)

print(f"  ✓ Computed ARCO fingerprints")

# 3. Display results
print("\n[3] Results:")
print(f"\n  Periodic pattern:")
print(f"    RCI (periodicity score): {features_periodic['rci']:.4f}")
print(f"    ARCO-print dimension:    {len(features_periodic['arco_print'])}")
print(f"    Top 3 rationals:")
for i, r in enumerate(features_periodic['top_rationals'][:3], 1):
    print(f"      {i}. freq={r['frequency']:.4f}, power={r['power']:.4f}, q={r['denominator']}")

print(f"\n  Random pattern:")
print(f"    RCI (periodicity score): {features_random['rci']:.4f}")
print(f"    ARCO-print dimension:    {len(features_random['arco_print'])}")
print(f"    Top 3 rationals:")
for i, r in enumerate(features_random['top_rationals'][:3], 1):
    print(f"      {i}. freq={r['frequency']:.4f}, power={r['power']:.4f}, q={r['denominator']}")

# 4. Comparison
print(f"\n[4] Comparison:")
rci_diff = features_periodic['rci'] - features_random['rci']
print(f"    RCI difference: {rci_diff:+.4f}")

if abs(rci_diff) > 0.05:
    if rci_diff > 0:
        print(f"    → Periodic pattern has HIGHER RCI (expected for uniform peaks)")
    else:
        print(f"    → Random pattern has HIGHER RCI (unexpected!)")
else:
    print(f"    → RCI values are similar (both patterns have comparable periodicity)")

# 5. Fingerprint similarity
from nomad_auto_xrd.lib.arco_analysis import XRDArcoAnalyzer

analyzer = XRDArcoAnalyzer(Qmax=40, alpha=0.5)
similarity = analyzer.compute_similarity(
    features_periodic['arco_print'],
    features_random['arco_print']
)

print(f"\n[5] ARCO Fingerprint Similarity:")
print(f"    L1 distance: {similarity:.2f}")
print(f"    (Lower = more similar; 0 = identical)")

# 6. Visualization (optional - requires matplotlib)
print(f"\n[6] Creating visualization...")

try:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot XRD patterns
    axes[0].plot(two_theta, intensity_periodic, linewidth=0.8, label='Periodic peaks')
    axes[0].plot(two_theta, intensity_random, linewidth=0.8, alpha=0.7, label='Random peaks')
    axes[0].set_xlabel('2θ (degrees)')
    axes[0].set_ylabel('Intensity (a.u.)')
    axes[0].set_title('XRD Patterns')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot ARCO fingerprints (first 200 features for visibility)
    axes[1].plot(features_periodic['arco_print'][:200], linewidth=0.8, label=f'Periodic (RCI={features_periodic["rci"]:.3f})')
    axes[1].plot(features_random['arco_print'][:200], linewidth=0.8, alpha=0.7, label=f'Random (RCI={features_random["rci"]:.3f})')
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('Arc Power')
    axes[1].set_title('ARCO Fingerprints (first 200 features)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), 'arco_quickstart_output.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    ✓ Saved visualization to: {output_path}")

    # Try to display (works in Jupyter or if display available)
    try:
        plt.show()
    except:
        pass

except ImportError:
    print(f"    ! matplotlib not available, skipping visualization")

# 7. Summary
print("\n" + "="*70)
print(" Summary")
print("="*70)
print(f"""
This example demonstrated:
  ✓ Computing ARCO features from XRD patterns
  ✓ Interpreting RCI (Rational Coherence Index)
  ✓ Extracting top rational frequencies
  ✓ Comparing ARCO fingerprints

Key Takeaways:
  • RCI measures periodicity strength (0-1, higher = more periodic)
  • ARCO-print provides fixed-length features for ML/similarity
  • Top rationals show dominant frequency components
  • Works on any 1-D signal (XRD, time series, etc.)

Next Steps:
  • Try on your own XRD data
  • Tune parameters (Qmax, alpha, major_q) for your application
  • Use ARCO-prints for similarity search or classification
  • See notebooks/arco_xrd_demo.ipynb for detailed tutorial
""")

print("="*70)
print("✓ Quickstart completed successfully!")
print("="*70)

"""
ARCO integration helpers for XRD analysis pipeline.

This module provides utilities to compute ARCO features from XRD patterns
and integrate them into the analysis workflow.
"""

from typing import Dict, Optional

import numpy as np

from nomad_auto_xrd.lib.arco_rci import ARCO, compute_derivative_track, generate_anchors


class XRDArcoAnalyzer:
    """
    Wrapper for ARCO analysis of XRD patterns.

    This class provides a simplified interface for applying ARCO to XRD
    intensity patterns, with sensible defaults for XRD analysis.

    Parameters:
        Qmax: Maximum denominator for rational anchors (30-60 for XRD).
        window_sizes: List of window sizes for multi-scale analysis.
        alpha: Bandwidth scale factor.
        major_q: Denominator threshold for "major" rationals in RCI.
        use_derivative: Whether to include derivative track.

    Example:
        >>> analyzer = XRDArcoAnalyzer()
        >>> result = analyzer.analyze_pattern(two_theta, intensity)
        >>> print(f"RCI: {result['rci']:.4f}")
    """

    def __init__(
        self,
        Qmax: int = 40,
        window_sizes: Optional[list[int]] = None,
        alpha: float = 0.5,
        major_q: int = 20,
        use_derivative: bool = True,
    ):
        """Initialize XRD ARCO analyzer."""
        self.Qmax = Qmax
        self.window_sizes = window_sizes or [128, 256]
        self.alpha = alpha
        self.major_q = major_q
        self.use_derivative = use_derivative

        # Generate anchors
        self.anchors = generate_anchors(Qmax=Qmax)

        # Create ARCO instance
        self.arco = ARCO(
            anchors=self.anchors,
            window_sizes=self.window_sizes,
            hop_fraction=0.25,
            alpha=alpha,
            apply_hann=True,
        )

    def analyze_pattern(
        self, two_theta: np.ndarray, intensity: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Analyze an XRD pattern and compute ARCO features.

        Args:
            two_theta: 2-theta values (can be non-uniform, will be resampled).
            intensity: Intensity values.

        Returns:
            Dictionary containing:
                - 'rci': Rational Coherence Index (float)
                - 'arco_print': ARCO fingerprint vector
                - 'arco_3d': Position-resolved arcogram (optional)
                - 'top_rationals': Top 10 rational frequencies
                - 'arc_powers': Average arc powers for all anchors
        """
        # Resample to uniform grid if needed
        if len(two_theta) < 1024:
            # Interpolate to at least 1024 points
            two_theta_uniform = np.linspace(
                two_theta.min(), two_theta.max(), 2048
            )
            intensity_uniform = np.interp(two_theta_uniform, two_theta, intensity)
        else:
            intensity_uniform = intensity

        # Prepare tracks
        tracks = {'intensity': intensity_uniform}

        if self.use_derivative:
            deriv = compute_derivative_track(intensity_uniform)
            tracks['derivative'] = deriv

        # Compute RCI
        rci = self.arco.compute_rci(tracks, major_q=self.major_q)

        # Compute ARCO-print
        arco_print = self.arco.compute_arco_print(tracks)

        # Compute average arc powers for interpretation
        arc_matrix = self.arco.compute_track_arcs(
            intensity_uniform, window_size=self.window_sizes[0]
        )
        avg_arc_powers = np.mean(arc_matrix, axis=0)

        # Get top rationals
        top_rationals = self.arco.get_top_rationals(avg_arc_powers, top_k=10)

        # Optionally compute ARCO-3D for position-resolved analysis
        arco_3d = self.arco.compute_arco_3d(tracks, finest_window=self.window_sizes[0])

        return {
            'rci': float(rci),
            'arco_print': arco_print,
            'arco_3d': arco_3d,
            'top_rationals': top_rationals,
            'arc_powers': avg_arc_powers,
            'anchors': self.anchors,
        }

    def compute_similarity(
        self, arco_print1: np.ndarray, arco_print2: np.ndarray
    ) -> float:
        """
        Compute similarity between two ARCO fingerprints.

        Args:
            arco_print1: First ARCO fingerprint.
            arco_print2: Second ARCO fingerprint.

        Returns:
            Similarity score (L1 distance, lower = more similar).
        """
        return float(np.sum(np.abs(arco_print1 - arco_print2)))


def compute_arco_features(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    Qmax: int = 40,
    window_sizes: Optional[list[int]] = None,
    alpha: float = 0.5,
    major_q: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to compute ARCO features from XRD pattern.

    Args:
        two_theta: 2-theta values.
        intensity: Intensity values.
        Qmax: Maximum denominator for rational anchors.
        window_sizes: Window sizes for multi-scale analysis.
        alpha: Bandwidth scale factor.
        major_q: Threshold for major rationals.

    Returns:
        Dictionary with ARCO features (rci, arco_print, etc.).

    Example:
        >>> features = compute_arco_features(two_theta, intensity)
        >>> print(f"RCI: {features['rci']:.4f}")
        >>> print(f"ARCO-print shape: {features['arco_print'].shape}")
    """
    analyzer = XRDArcoAnalyzer(
        Qmax=Qmax,
        window_sizes=window_sizes,
        alpha=alpha,
        major_q=major_q,
        use_derivative=True,
    )

    return analyzer.analyze_pattern(two_theta, intensity)

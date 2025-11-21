"""
ARCO (Arcogram with Rational Coherence pooling) and RCI implementation.

This module implements ARCO fingerprinting and RCI (Rational Coherence Index)
for analyzing periodic patterns in 1D signals, particularly XRD diffraction patterns.

Key concepts:
- ARCO converts 1D signals into fixed-length fingerprints by computing power spectra
  and integrating around rational-frequency anchors (Farey/Diophantine set)
- RCI measures the fraction of spectral energy concentrated on major rational arcs
- Uses Gaussian weighting around rational anchors for smooth spectral decomposition
"""

import math
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np


def generate_anchors(Qmax: int = 11) -> List[float]:
    """
    Generate rational frequency anchors (Farey sequence).

    Creates a set of rational numbers a/q where:
    - 0 < a/q < 0.5 (normalized frequency, Nyquist = 0.5)
    - gcd(a, q) = 1 (reduced fractions only)
    - q <= Qmax

    Args:
        Qmax: Maximum denominator for rational anchors

    Returns:
        Sorted list of rational anchor frequencies as floats

    Example:
        >>> anchors = generate_anchors(Qmax=11)
        >>> print(anchors[:5])
        [0.09090909090909091, 0.1, 0.1111111111111111, 0.125, 0.14285714285714285]
    """
    anchors = set()
    for q in range(2, Qmax + 1):
        for a in range(1, q):
            if math.gcd(a, q) == 1:
                r = a / q
                # Only keep frequencies in (0, 0.5) range
                if 0 < r < 0.5:
                    anchors.add(Fraction(a, q))

    # Convert to sorted list of floats
    anchors_list = sorted([float(frac) for frac in anchors])
    return anchors_list


def _estimate_denominator(r: float, max_denominator: int = 100) -> int:
    """
    Estimate the denominator of a rational number from its float representation.

    Args:
        r: Float value to estimate denominator for
        max_denominator: Maximum denominator to consider

    Returns:
        Estimated denominator
    """
    frac = Fraction(r).limit_denominator(max_denominator)
    return frac.denominator


class ARCO:
    """
    ARCO (Arcogram with Rational Coherence) analyzer.

    Computes spectral fingerprints by integrating power around rational frequency
    anchors using sliding windows and Gaussian weighting.

    Attributes:
        anchors: List of rational frequency anchors
        window_sizes: List of window sizes for multi-scale analysis
        hop_fraction: Fraction of window to hop (0.25 = 75% overlap)
        alpha: Bandwidth scale factor (bandwidth = alpha / q^2)
        apply_hann: Whether to apply Hann window tapering
        track_names: Names of tracks for multi-track analysis
    """

    def __init__(
        self,
        anchors: List[float],
        window_sizes: List[int] = [31, 63],
        hop_fraction: float = 0.25,
        alpha: float = 1.0,
        apply_hann: bool = True,
        track_names: List[str] = None,
    ):
        """
        Initialize ARCO analyzer.

        Args:
            anchors: List of rational frequency anchors from generate_anchors()
            window_sizes: Window sizes for multi-scale analysis
            hop_fraction: Hop size as fraction of window (0.25 = 75% overlap)
            alpha: Bandwidth scaling (delta = alpha / q^2)
            apply_hann: Apply Hann window tapering
            track_names: Optional names for tracks
        """
        self.anchors = anchors
        self.window_sizes = window_sizes
        self.hop_fraction = hop_fraction
        self.alpha = alpha
        self.apply_hann = apply_hann
        self.track_names = track_names

    def _compute_window_power(
        self, signal_window: np.ndarray, anchor_denominators: List[int]
    ) -> np.ndarray:
        """
        Compute arc powers for a single window.

        Args:
            signal_window: 1D signal segment
            anchor_denominators: Denominators for each anchor (for bandwidth calc)

        Returns:
            Array of arc powers, shape (n_anchors,)
        """
        W = len(signal_window)

        # 1. Detrend and normalize
        s = signal_window - np.mean(signal_window)
        std = np.std(s)
        if std > 1e-12:
            s = s / std

        # 2. Apply Hann window if requested
        if self.apply_hann:
            hann = np.hanning(W)
            s = s * hann
            window_energy = np.sum(hann * hann)
        else:
            window_energy = W

        # 3. Compute FFT and power spectrum
        freqs = np.fft.rfftfreq(W, d=1.0)  # Normalized frequency
        fft_vals = np.fft.rfft(s)
        power = np.abs(fft_vals) ** 2

        # Correct for window energy
        power = power / (window_energy + 1e-12)

        # Normalize so power sums to 1
        power_sum = power.sum()
        if power_sum > 1e-12:
            power = power / power_sum

        # 4. Integrate around each rational anchor with Gaussian weighting
        arc_powers = []
        for r, q in zip(self.anchors, anchor_denominators):
            # Compute bandwidth: delta = alpha / q^2
            delta = self.alpha / (q ** 2)

            # Convert delta to sigma (FWHM to sigma conversion)
            sigma = delta / 2.3548

            # Gaussian weights centered at frequency r
            weights = np.exp(-0.5 * ((freqs - r) / (sigma + 1e-12)) ** 2)

            # Integrate: sum of power * weights
            arc_power = np.sum(power * weights)
            arc_powers.append(arc_power)

        return np.array(arc_powers)

    def compute_track_arcs(
        self, track_signal: np.ndarray, window_size: int
    ) -> np.ndarray:
        """
        Compute arc powers across all windows for a single track.

        Args:
            track_signal: 1D signal array
            window_size: Window size for STFT-style analysis

        Returns:
            Array of arc powers, shape (n_windows, n_anchors)
        """
        N = len(track_signal)
        hop = int(window_size * self.hop_fraction)

        # Pre-compute denominators for all anchors
        anchor_denominators = [
            _estimate_denominator(r, max_denominator=100) for r in self.anchors
        ]

        # Sliding window analysis
        arc_powers_list = []
        for start in range(0, N - window_size + 1, hop):
            window = track_signal[start : start + window_size]
            arc_powers = self._compute_window_power(window, anchor_denominators)
            arc_powers_list.append(arc_powers)

        if not arc_powers_list:
            # Signal too short, compute single window
            arc_powers = self._compute_window_power(track_signal, anchor_denominators)
            arc_powers_list.append(arc_powers)

        return np.array(arc_powers_list)

    def compute_arco_print(self, tracks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute ARCO-print: fixed-length fingerprint vector.

        Computes arc powers for all tracks and window sizes, then averages
        across windows and flattens to create a fixed-length feature vector.

        Args:
            tracks: Dictionary mapping track names to 1D signal arrays

        Returns:
            1D ARCO-print vector of shape (n_tracks * n_anchors * n_window_sizes,)
        """
        fingerprint_components = []

        for track_name, track_signal in tracks.items():
            for window_size in self.window_sizes:
                # Compute arc powers for this track and window size
                arc_powers = self.compute_track_arcs(track_signal, window_size)

                # Average across windows to get fixed-length representation
                mean_arc_powers = np.mean(arc_powers, axis=0)
                fingerprint_components.append(mean_arc_powers)

        # Flatten into single vector
        arco_print = np.concatenate(fingerprint_components)
        return arco_print

    def compute_arco_3d(
        self, tracks: Dict[str, np.ndarray], finest_window: int
    ) -> np.ndarray:
        """
        Compute position-resolved ARCO (ARCO-3D).

        Returns per-window arc powers for localization and visualization.

        Args:
            tracks: Dictionary mapping track names to 1D signal arrays
            finest_window: Window size to use for position resolution

        Returns:
            3D array of shape (n_tracks, n_windows, n_anchors)
        """
        arco_3d = []

        for track_name, track_signal in tracks.items():
            arc_powers = self.compute_track_arcs(track_signal, finest_window)
            arco_3d.append(arc_powers)

        return np.array(arco_3d)

    def compute_rci(
        self, tracks: Dict[str, np.ndarray], major_q: int = 11
    ) -> float:
        """
        Compute Rational Coherence Index (RCI).

        RCI is the fraction of total spectral energy concentrated on major
        rational arcs (anchors with denominator <= major_q). Bounded [0, 1].

        Args:
            tracks: Dictionary mapping track names to 1D signal arrays
            major_q: Maximum denominator for "major" rational arcs

        Returns:
            RCI value in [0, 1]
        """
        # Identify major anchor indices
        anchor_denominators = [
            _estimate_denominator(r, max_denominator=100) for r in self.anchors
        ]
        major_indices = [i for i, q in enumerate(anchor_denominators) if q <= major_q]

        if not major_indices:
            return 0.0

        total_arc_power = 0.0
        n_windows_total = 0

        for track_signal in tracks.values():
            # Use first window size for RCI computation
            window_size = self.window_sizes[0]
            arc_powers = self.compute_track_arcs(track_signal, window_size)

            # Sum power from major arcs across all windows
            major_arc_powers = arc_powers[:, major_indices]
            total_arc_power += np.sum(major_arc_powers)
            n_windows_total += arc_powers.shape[0]

        # Average RCI across all tracks and windows
        if n_windows_total > 0:
            rci = total_arc_power / (n_windows_total * len(tracks))
        else:
            rci = 0.0

        # Cap at 1.0 due to potential Gaussian overlap
        rci = min(1.0, rci)

        return rci

    def null_model_zscore(
        self,
        track: np.ndarray,
        n_shuffles: int = 50,
        preserve_composition: bool = True,
    ) -> Tuple[float, float]:
        """
        Compute z-score of RCI against null model via shuffling.

        Args:
            track: 1D signal array
            n_shuffles: Number of shuffles for null distribution
            preserve_composition: Preserve value histogram when shuffling

        Returns:
            Tuple of (z_score, p_value_estimate)
            z_score: (RCI_real - mean_null) / std_null
            p_value: Fraction of null RCI values >= real RCI
        """
        # Compute real RCI
        real_rci = self.compute_rci({'track': track})

        # Generate null distribution
        null_rcis = []
        for _ in range(n_shuffles):
            if preserve_composition:
                # Shuffle preserving value histogram
                shuffled = np.random.permutation(track)
            else:
                # Complete randomization
                shuffled = np.random.randn(len(track))

            null_rci = self.compute_rci({'track': shuffled})
            null_rcis.append(null_rci)

        null_rcis = np.array(null_rcis)

        # Compute z-score
        mean_null = np.mean(null_rcis)
        std_null = np.std(null_rcis)

        if std_null > 1e-12:
            z_score = (real_rci - mean_null) / std_null
        else:
            z_score = 0.0

        # Estimate p-value (one-tailed: how many null >= real)
        p_value = np.mean(null_rcis >= real_rci)

        return z_score, p_value


def get_anchor_interpretation_table() -> Dict[float, Dict[str, str]]:
    """
    Get interpretation table for common rational anchors.

    Returns:
        Dictionary mapping anchor frequencies to their interpretations
    """
    return {
        1/2: {
            'period': '2',
            'typical_meaning': 'Period-2 → alternating pattern / β-strand-like',
        },
        1/3: {
            'period': '3',
            'typical_meaning': 'Period-3 → collagen-like repeat',
        },
        1/7: {
            'period': '7',
            'typical_meaning': 'Heptad → coiled-coil',
        },
        5/18: {
            'period': '~3.6',
            'typical_meaning': 'α-helix (~3.6 residues per turn)',
        },
        1/20: {
            'period': '20',
            'typical_meaning': 'Long beat / long-distance periodicity',
        },
    }

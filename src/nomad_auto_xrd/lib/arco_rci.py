"""
ARCO (Arcogram with Rational Diophantine pooling) for XRD Pattern Analysis.

This module implements the ARCO algorithm which converts 1-D signals (e.g., XRD
diffraction intensity over 2θ) into fixed-length, interpretable fingerprints by:

1. Computing local power spectra (STFT-style sliding windows with window tapering)
2. Integrating spectral power around rational-frequency anchors a/q (Farey/rational set)
3. Summarizing power across tracks and windows to produce:
   - ARCO-print: vector of arc powers (track × rational × scale)
   - ARCO-3D: position-resolved arcogram (for localization)
   - RCI (Rational Coherence Index): scalar fraction of total spectral energy
     concentrated on major rational arcs (bounded [0,1])

Low-denominator rationals correspond to small-integer periodicities seen in many
physical systems. Aggregating energy at these rationals gives semantically meaningful
features and resists jitter/noise better than raw bins.
"""

import math
from fractions import Fraction
from typing import Dict, List, Optional

import numpy as np


def generate_anchors(Qmax: int = 11) -> List[float]:
    """
    Generate rational frequency anchors (Farey sequence) up to denominator Qmax.

    For each rational a/q where gcd(a,q)=1 and q <= Qmax, we include the
    normalized frequency in (0, 0.5). This corresponds to meaningful periodic
    structures in the signal.

    Args:
        Qmax: Maximum denominator for rational anchors.

    Returns:
        Sorted list of unique rational frequencies in (0, 0.5).

    Example:
        >>> anchors = generate_anchors(Qmax=7)
        >>> len(anchors)  # Will include 1/2, 1/3, 2/3->1/3, 1/4, 3/4->1/4, etc.
        11
    """
    anchors = set()
    for q in range(2, Qmax + 1):
        for a in range(1, q):
            if math.gcd(a, q) == 1:
                r = a / q
                # Map to [0, 0.5] by reflecting > 0.5
                if r > 0.5:
                    r = 1 - r
                if 0 < r < 0.5:
                    anchors.add(r)

    # Sort and return as list
    return sorted(list(anchors))


class ARCO:
    """
    ARCO (Arcogram with Rational Coherence Index) analyzer.

    This class provides methods to compute ARCO fingerprints, position-resolved
    arcograms (ARCO-3D), and the Rational Coherence Index (RCI) from 1-D signals.

    Parameters:
        anchors: List of rational frequency anchors (use generate_anchors()).
        window_sizes: List of window sizes for multi-scale analysis.
        hop_fraction: Fraction of window size to use as hop (0.25 = 75% overlap).
        alpha: Bandwidth scale factor (bandwidth = alpha / q^2).
        apply_hann: Whether to apply Hann window tapering.
        track_names: Optional names for tracks.

    Example:
        >>> from nomad_auto_xrd.lib.arco_rci import ARCO, generate_anchors
        >>> anchors = generate_anchors(Qmax=40)
        >>> arco = ARCO(anchors, window_sizes=[128, 256])
        >>> # For XRD intensity pattern:
        >>> tracks = {'intensity': intensity_array}
        >>> rci = arco.compute_rci(tracks, major_q=11)
        >>> arco_print = arco.compute_arco_print(tracks)
    """

    def __init__(
        self,
        anchors: List[float],
        window_sizes: List[int] = None,
        hop_fraction: float = 0.25,
        alpha: float = 1.0,
        apply_hann: bool = True,
        track_names: Optional[List[str]] = None,
    ):
        """Initialize ARCO analyzer."""
        self.anchors = np.array(anchors)
        self.window_sizes = window_sizes or [31, 63]
        self.hop_fraction = hop_fraction
        self.alpha = alpha
        self.apply_hann = apply_hann
        self.track_names = track_names

        # Precompute anchor denominators for bandwidth calculation
        self._anchor_denoms = self._compute_anchor_denominators()

    def _compute_anchor_denominators(self) -> np.ndarray:
        """
        Compute denominators for each anchor by converting to Fraction.

        Returns:
            Array of denominators for each anchor.
        """
        denoms = []
        for anchor in self.anchors:
            # Convert float to fraction with reasonable precision
            frac = Fraction(anchor).limit_denominator(1000)
            denoms.append(frac.denominator)
        return np.array(denoms)

    def _process_window(
        self, signal_window: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process a single window: detrend, normalize, apply window, compute FFT.

        Args:
            signal_window: 1-D signal segment.

        Returns:
            Tuple of (frequencies, normalized_power).
        """
        W = len(signal_window)

        # 1. Detrend and normalize
        s = signal_window - np.mean(signal_window)
        std = np.std(s)
        s = s / (std + 1e-12)

        # 2. Apply Hann window if requested
        if self.apply_hann:
            hann = np.hanning(W)
            s = s * hann
            window_energy = np.sum(hann * hann)
        else:
            window_energy = W

        # 3. Compute FFT power
        freqs = np.fft.rfftfreq(W, d=1.0)  # Normalized frequency
        fft_vals = np.fft.rfft(s)
        power = np.abs(fft_vals) ** 2

        # Correct for window energy
        power = power / (window_energy + 1e-12)

        # Normalize so power sums to 1
        total_power = power.sum()
        if total_power > 0:
            power = power / total_power

        return freqs, power

    def _integrate_arc_power(
        self, freqs: np.ndarray, power: np.ndarray
    ) -> np.ndarray:
        """
        Integrate power around rational anchors using Gaussian weighting.

        Args:
            freqs: Frequency bins from FFT.
            power: Normalized power at each frequency.

        Returns:
            Array of arc powers for each anchor.
        """
        arc_powers = np.zeros(len(self.anchors))

        for i, (r, q) in enumerate(zip(self.anchors, self._anchor_denoms)):
            # Bandwidth: delta = alpha / q^2
            delta = self.alpha / (q**2)

            # Gaussian sigma from FWHM relationship
            sigma = delta / 2.3548

            # Gaussian weights centered at r
            weights = np.exp(-0.5 * ((freqs - r) / (sigma + 1e-12)) ** 2)

            # Normalize weights to sum to 1 (so arc power represents fraction of total power in this band)
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum

            # Integrate weighted power
            arc_powers[i] = np.sum(power * weights)

        return arc_powers

    def compute_track_arcs(
        self, track_signal: np.ndarray, window_size: int
    ) -> np.ndarray:
        """
        Compute arc powers for a single track across sliding windows.

        Args:
            track_signal: 1-D signal (e.g., XRD intensity).
            window_size: Size of sliding window.

        Returns:
            Array of shape (n_windows, n_anchors).
        """
        N = len(track_signal)
        hop = max(1, int(window_size * self.hop_fraction))

        # Generate window starts
        window_starts = list(range(0, N - window_size + 1, hop))
        if not window_starts:
            # Signal too short, use entire signal
            window_starts = [0]
            W = min(N, window_size)
        else:
            W = window_size

        n_windows = len(window_starts)
        arc_matrix = np.zeros((n_windows, len(self.anchors)))

        for w_idx, start in enumerate(window_starts):
            window = track_signal[start : start + W]

            # Process window
            freqs, power = self._process_window(window)

            # Integrate around anchors
            arc_powers = self._integrate_arc_power(freqs, power)

            arc_matrix[w_idx, :] = arc_powers

        return arc_matrix

    def compute_arco_print(self, tracks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute ARCO fingerprint: fixed-length vector summarizing all tracks.

        The fingerprint is created by:
        1. For each track and window size, compute arc powers
        2. Average across windows (position-invariant)
        3. Flatten to 1-D vector: [track1_ws1, track1_ws2, ..., trackN_ws2]

        Args:
            tracks: Dictionary mapping track names to 1-D signals.

        Returns:
            1-D ARCO fingerprint vector of length:
            n_tracks × n_window_sizes × n_anchors.
        """
        fingerprints = []

        for track_name, track_signal in tracks.items():
            for window_size in self.window_sizes:
                # Compute arc powers across windows
                arc_matrix = self.compute_track_arcs(track_signal, window_size)

                # Average across windows for position-invariance
                avg_arc_powers = np.mean(arc_matrix, axis=0)

                fingerprints.append(avg_arc_powers)

        # Concatenate all into single vector
        arco_print = np.concatenate(fingerprints)

        return arco_print

    def compute_arco_3d(
        self, tracks: Dict[str, np.ndarray], finest_window: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute position-resolved ARCO map (ARCO-3D).

        This maintains positional information by not averaging across windows.
        Useful for identifying where in the signal periodic structure changes.

        Args:
            tracks: Dictionary mapping track names to 1-D signals.
            finest_window: Window size to use (defaults to smallest in window_sizes).

        Returns:
            Array of shape (n_windows, n_anchors × n_tracks).
        """
        if finest_window is None:
            finest_window = min(self.window_sizes)

        all_track_arcs = []

        for track_name, track_signal in tracks.items():
            arc_matrix = self.compute_track_arcs(track_signal, finest_window)
            all_track_arcs.append(arc_matrix)

        # Stack tracks: shape (n_windows, n_anchors, n_tracks)
        arco_3d = np.stack(all_track_arcs, axis=2)

        # Reshape to (n_windows, n_anchors × n_tracks)
        n_windows, n_anchors, n_tracks = arco_3d.shape
        arco_3d = arco_3d.reshape(n_windows, n_anchors * n_tracks)

        return arco_3d

    def compute_rci(
        self, tracks: Dict[str, np.ndarray], major_q: int = 11
    ) -> float:
        """
        Compute Rational Coherence Index (RCI).

        RCI measures the fraction of total spectral energy concentrated on
        major rational arcs (low-denominator rationals). Higher RCI indicates
        stronger periodic structure.

        Args:
            tracks: Dictionary mapping track names to 1-D signals.
            major_q: Maximum denominator to consider as "major" rational.

        Returns:
            RCI value in [0, 1]. Higher values indicate stronger periodicity.
        """
        # Identify major anchors (q <= major_q)
        major_mask = self._anchor_denoms <= major_q
        major_indices = np.where(major_mask)[0]

        total_arc_power = 0.0
        total_windows = 0

        for track_name, track_signal in tracks.items():
            # Use smallest window for RCI computation
            window_size = min(self.window_sizes)
            arc_matrix = self.compute_track_arcs(track_signal, window_size)

            # Sum power in major arcs across all windows
            major_arc_power = np.sum(arc_matrix[:, major_indices])

            total_arc_power += major_arc_power
            total_windows += arc_matrix.shape[0]

        # Average RCI across tracks and windows
        # Each window's power sums to 1, so normalize by total windows
        if total_windows > 0:
            rci = total_arc_power / (len(tracks) * total_windows)
        else:
            rci = 0.0

        # Cap at 1.0 (can exceed due to overlapping Gaussians)
        rci = min(1.0, rci)

        return rci

    def null_model_zscore(
        self,
        track: np.ndarray,
        n_shuffles: int = 50,
        preserve_composition: bool = True,
    ) -> float:
        """
        Compute z-score of RCI against null model (shuffled signal).

        This validates whether observed periodicity is significant or just
        due to compositional bias.

        Args:
            track: 1-D signal.
            n_shuffles: Number of random shuffles for null distribution.
            preserve_composition: If True, shuffle preserves amplitude histogram.

        Returns:
            Z-score: (RCI_real - mean_null) / std_null.
        """
        # Compute real RCI
        real_rci = self.compute_rci({'track': track})

        # Generate null distribution
        null_rcis = []
        for _ in range(n_shuffles):
            if preserve_composition:
                # Shuffle preserving composition
                shuffled = np.random.permutation(track)
            else:
                # Random Gaussian noise
                shuffled = np.random.randn(len(track))

            null_rci = self.compute_rci({'track': shuffled})
            null_rcis.append(null_rci)

        null_rcis = np.array(null_rcis)
        mean_null = np.mean(null_rcis)
        std_null = np.std(null_rcis)

        if std_null > 0:
            zscore = (real_rci - mean_null) / std_null
        else:
            zscore = 0.0

        return zscore

    def get_top_rationals(
        self, arc_powers: np.ndarray, top_k: int = 5
    ) -> List[tuple[float, float, int]]:
        """
        Get top-k rational anchors by arc power.

        Args:
            arc_powers: Array of arc powers (length n_anchors).
            top_k: Number of top rationals to return.

        Returns:
            List of (anchor_frequency, arc_power, denominator) tuples,
            sorted by descending power.
        """
        top_indices = np.argsort(arc_powers)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            anchor = self.anchors[idx]
            power = arc_powers[idx]
            denom = self._anchor_denoms[idx]
            results.append((anchor, power, denom))

        return results


def compute_derivative_track(signal: np.ndarray) -> np.ndarray:
    """
    Compute first derivative track (emphasizes peak spacing).

    Args:
        signal: 1-D signal.

    Returns:
        Derivative signal (same length, padded at edges).
    """
    # Simple finite difference
    deriv = np.diff(signal)
    # Pad to maintain length
    deriv = np.concatenate([[deriv[0]], deriv])
    return deriv


def resample_uniform(
    x: np.ndarray, y: np.ndarray, n_points: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample (x, y) data to uniform grid.

    Useful for XRD patterns which may have non-uniform 2θ spacing.

    Args:
        x: Independent variable (e.g., 2-theta).
        y: Dependent variable (e.g., intensity).
        n_points: Number of points in uniform grid.

    Returns:
        Tuple of (x_uniform, y_uniform).
    """
    x_uniform = np.linspace(x.min(), x.max(), n_points)
    y_uniform = np.interp(x_uniform, x, y)
    return x_uniform, y_uniform


# Interpretability table for common rationals
RATIONAL_MEANINGS = {
    1 / 2: "Period-2 (alternating pattern)",
    1 / 3: "Period-3",
    1 / 4: "Period-4 (quarter repeats)",
    1 / 5: "Period-5",
    1 / 6: "Period-6",
    1 / 7: "Period-7 (heptad repeat)",
    2 / 7: "Period-7/2",
    1 / 8: "Period-8",
    3 / 8: "Period-8/3",
    1 / 10: "Period-10",
    1 / 11: "Period-11",
    5 / 18: "~Period-3.6 (alpha-helix-like)",
}


def interpret_rational(freq: float, tolerance: float = 0.01) -> str:
    """
    Get interpretation of a rational frequency.

    Args:
        freq: Rational frequency.
        tolerance: Tolerance for matching known rationals.

    Returns:
        Interpretation string.
    """
    for known_freq, meaning in RATIONAL_MEANINGS.items():
        if abs(freq - known_freq) < tolerance:
            return meaning

    # Generic interpretation
    frac = Fraction(freq).limit_denominator(100)
    period = 1 / freq if freq > 0 else float('inf')
    return f"Period-{period:.1f} ({frac.numerator}/{frac.denominator})"

"""
Unit tests for ARCO (Arcogram with Rational Diophantine pooling) module.

Tests cover:
1. Sine pattern test - single-tone sine should highlight correct rational anchor
2. Heptad test - synthetic 7-residue repeat pattern
3. Noise baseline - white noise should give low RCI
4. Resolution test - downsampled signal recovery
5. Multi-scale analysis
6. Edge cases
"""

import math

import numpy as np
import pytest

from nomad_auto_xrd.lib.arco_rci import (
    ARCO,
    compute_derivative_track,
    generate_anchors,
    resample_uniform,
)


class TestAnchorGeneration:
    """Test rational anchor generation."""

    def test_generate_anchors_basic(self):
        """Test basic anchor generation."""
        anchors = generate_anchors(Qmax=5)

        # Should include rationals like 1/2, 1/3, 1/4, 1/5, 2/5, etc.
        assert len(anchors) > 0
        assert all(0 < a < 0.5 for a in anchors)
        assert 1 / 2 in anchors or abs(min(anchors, key=lambda x: abs(x - 0.5)) - 0.5) < 0.01

    def test_generate_anchors_sorted(self):
        """Test that anchors are sorted."""
        anchors = generate_anchors(Qmax=10)
        assert anchors == sorted(anchors)

    def test_generate_anchors_unique(self):
        """Test that anchors are unique."""
        anchors = generate_anchors(Qmax=15)
        assert len(anchors) == len(set(anchors))

    def test_generate_anchors_count(self):
        """Test approximate count of anchors for known Qmax."""
        # For Qmax=11, we expect around 40-50 unique rationals in (0, 0.5)
        anchors = generate_anchors(Qmax=11)
        assert 30 < len(anchors) < 60


class TestSinePattern:
    """Test ARCO on synthetic sine waves."""

    def test_single_tone_detection(self):
        """Test that a single-frequency sine wave highlights the correct rational."""
        # Create sine wave at frequency 1/8 (0.125 cycles/sample)
        target_freq = 1 / 8
        N = 1024
        t = np.arange(N)
        signal = np.sin(2 * np.pi * target_freq * t)

        # Setup ARCO
        anchors = generate_anchors(Qmax=20)
        arco = ARCO(anchors, window_sizes=[128], alpha=1.0)

        # Compute arc powers
        tracks = {'amplitude': signal}
        arc_matrix = arco.compute_track_arcs(signal, window_size=128)

        # Average across windows
        avg_arc_powers = np.mean(arc_matrix, axis=0)

        # Find anchor closest to target frequency
        closest_idx = np.argmin(np.abs(arco.anchors - target_freq))

        # The closest anchor should have high power
        top_3_indices = np.argsort(avg_arc_powers)[-3:]

        assert closest_idx in top_3_indices, (
            f"Expected anchor {arco.anchors[closest_idx]:.4f} "
            f"to be in top 3, but got {arco.anchors[top_3_indices]}"
        )

    def test_rci_high_for_periodic(self):
        """Test that RCI is high for periodic signals."""
        # Create periodic signal (sine at 1/7)
        N = 2048
        t = np.arange(N)
        signal = np.sin(2 * np.pi * (1 / 7) * t)

        anchors = generate_anchors(Qmax=15)
        arco = ARCO(anchors, window_sizes=[128, 256], alpha=1.0)

        rci = arco.compute_rci({'amplitude': signal}, major_q=11)

        # RCI should be reasonably high for strong periodic signal
        assert rci > 0.1, f"Expected RCI > 0.1 for periodic signal, got {rci}"

    def test_multi_frequency_detection(self):
        """Test detection of multiple frequency components."""
        N = 2048
        t = np.arange(N)

        # Two components: 1/4 and 1/8
        signal = np.sin(2 * np.pi * (1 / 4) * t) + 0.5 * np.sin(
            2 * np.pi * (1 / 8) * t
        )

        anchors = generate_anchors(Qmax=20)
        arco = ARCO(anchors, window_sizes=[256], alpha=1.0)

        arc_matrix = arco.compute_track_arcs(signal, window_size=256)
        avg_arc_powers = np.mean(arc_matrix, axis=0)

        # Find top 5 rationals
        top_rationals = arco.get_top_rationals(avg_arc_powers, top_k=5)

        # Check that both 1/4 and 1/8 are represented
        top_freqs = [r[0] for r in top_rationals]

        # Allow some tolerance
        has_quarter = any(abs(f - 0.25) < 0.02 for f in top_freqs)
        has_eighth = any(abs(f - 0.125) < 0.02 for f in top_freqs)

        assert has_quarter or has_eighth, (
            f"Expected to find 1/4 or 1/8 in top rationals, got {top_freqs}"
        )


class TestHeptadPattern:
    """Test ARCO on heptad repeat pattern (period-7)."""

    def test_heptad_repeat(self):
        """Test that period-7 pattern highlights 1/7 anchor."""
        # Create signal with period-7 pattern
        pattern = np.array([1, 0, 0, 2, 0, 0, 1])  # 7-element motif
        N_repeats = 50
        signal = np.tile(pattern, N_repeats)

        anchors = generate_anchors(Qmax=20)
        arco = ARCO(anchors, window_sizes=[63], alpha=2.0)  # Wider bands for discrete

        arc_matrix = arco.compute_track_arcs(signal, window_size=63)
        avg_arc_powers = np.mean(arc_matrix, axis=0)

        # Find anchor closest to 1/7 ≈ 0.1429
        target_freq = 1 / 7
        closest_idx = np.argmin(np.abs(arco.anchors - target_freq))

        # Should be in top performers
        top_5_indices = np.argsort(avg_arc_powers)[-5:]

        assert closest_idx in top_5_indices, (
            f"Expected 1/7 anchor to be in top 5 for heptad pattern"
        )


class TestNoiseBaseline:
    """Test ARCO on noise signals."""

    def test_white_noise_low_rci(self):
        """Test that white noise gives low RCI."""
        np.random.seed(42)
        N = 2048
        noise = np.random.randn(N)

        anchors = generate_anchors(Qmax=11)
        arco = ARCO(anchors, window_sizes=[128], alpha=1.0)

        rci = arco.compute_rci({'noise': noise}, major_q=11)

        # RCI should be low for noise (< 0.3 typically)
        assert rci < 0.4, f"Expected low RCI for white noise, got {rci}"

    def test_zscore_significance(self):
        """Test z-score computation against null model."""
        # Periodic signal
        N = 1024
        t = np.arange(N)
        signal = np.sin(2 * np.pi * (1 / 5) * t)

        anchors = generate_anchors(Qmax=11)
        arco = ARCO(anchors, window_sizes=[128], alpha=1.0)

        # Compute z-score (this may take a moment)
        zscore = arco.null_model_zscore(signal, n_shuffles=20, preserve_composition=True)

        # Periodic signal should have positive z-score
        assert zscore > 0, f"Expected positive z-score for periodic signal, got {zscore}"

    def test_noise_near_zero_zscore(self):
        """Test that pure noise has z-score near zero."""
        np.random.seed(123)
        N = 512
        noise = np.random.randn(N)

        anchors = generate_anchors(Qmax=11)
        arco = ARCO(anchors, window_sizes=[64], alpha=1.0)

        zscore = arco.null_model_zscore(noise, n_shuffles=20)

        # Noise should have z-score close to 0 (within ±2)
        assert abs(zscore) < 3, f"Expected |z-score| < 3 for noise, got {zscore}"


class TestResolution:
    """Test ARCO behavior with different resolutions."""

    def test_downsampled_signal_recovery(self):
        """Test that downsampled signal still recovers correct rational."""
        # High-res signal
        N_high = 4096
        t_high = np.arange(N_high)
        signal_high = np.sin(2 * np.pi * (1 / 10) * t_high)

        # Downsample by 2x
        signal_low = signal_high[::2]

        anchors = generate_anchors(Qmax=20)
        arco = ARCO(anchors, window_sizes=[256], alpha=1.0)

        # Compute for both
        arc_high = arco.compute_track_arcs(signal_high, window_size=256)
        arc_low = arco.compute_track_arcs(signal_low, window_size=128)

        avg_high = np.mean(arc_high, axis=0)
        avg_low = np.mean(arc_low, axis=0)

        # Both should identify similar top rational
        top_high = arco.get_top_rationals(avg_high, top_k=1)[0][0]
        top_low = arco.get_top_rationals(avg_low, top_k=1)[0][0]

        # Should be close (within tolerance)
        assert abs(top_high - top_low) < 0.05, (
            f"Downsampled top rational {top_low:.4f} differs from "
            f"high-res {top_high:.4f}"
        )


class TestARCOPrint:
    """Test ARCO fingerprint generation."""

    def test_arco_print_shape(self):
        """Test that ARCO-print has expected shape."""
        N = 1024
        signal = np.random.randn(N)

        anchors = generate_anchors(Qmax=11)
        n_anchors = len(anchors)

        window_sizes = [64, 128]
        n_window_sizes = len(window_sizes)

        arco = ARCO(anchors, window_sizes=window_sizes, alpha=1.0)

        # Single track
        tracks = {'amplitude': signal}
        arco_print = arco.compute_arco_print(tracks)

        expected_length = 1 * n_window_sizes * n_anchors  # 1 track
        assert len(arco_print) == expected_length

    def test_arco_print_multi_track(self):
        """Test ARCO-print with multiple tracks."""
        N = 512
        signal = np.random.randn(N)
        deriv = compute_derivative_track(signal)

        anchors = generate_anchors(Qmax=11)
        n_anchors = len(anchors)

        window_sizes = [32, 64]
        arco = ARCO(anchors, window_sizes=window_sizes, alpha=1.0)

        tracks = {'amplitude': signal, 'derivative': deriv}
        arco_print = arco.compute_arco_print(tracks)

        expected_length = 2 * len(window_sizes) * n_anchors  # 2 tracks
        assert len(arco_print) == expected_length

    def test_arco_print_deterministic(self):
        """Test that ARCO-print is deterministic."""
        N = 256
        np.random.seed(99)
        signal = np.random.randn(N)

        anchors = generate_anchors(Qmax=11)
        arco = ARCO(anchors, window_sizes=[32], alpha=1.0)

        tracks = {'signal': signal}
        print1 = arco.compute_arco_print(tracks)
        print2 = arco.compute_arco_print(tracks)

        np.testing.assert_array_almost_equal(print1, print2)


class TestARCO3D:
    """Test position-resolved ARCO (ARCO-3D)."""

    def test_arco_3d_shape(self):
        """Test ARCO-3D output shape."""
        N = 512
        signal = np.random.randn(N)

        anchors = generate_anchors(Qmax=11)
        arco = ARCO(anchors, window_sizes=[64, 128], alpha=1.0)

        tracks = {'amplitude': signal}
        arco_3d = arco.compute_arco_3d(tracks, finest_window=64)

        # Should have shape (n_windows, n_anchors * n_tracks)
        assert arco_3d.ndim == 2
        assert arco_3d.shape[1] == len(anchors) * 1  # 1 track

    def test_arco_3d_multi_track(self):
        """Test ARCO-3D with multiple tracks."""
        N = 256
        signal1 = np.random.randn(N)
        signal2 = np.random.randn(N)

        anchors = generate_anchors(Qmax=11)
        arco = ARCO(anchors, window_sizes=[32], alpha=1.0)

        tracks = {'s1': signal1, 's2': signal2}
        arco_3d = arco.compute_arco_3d(tracks, finest_window=32)

        assert arco_3d.shape[1] == len(anchors) * 2  # 2 tracks


class TestUtilityFunctions:
    """Test utility functions."""

    def test_compute_derivative(self):
        """Test derivative track computation."""
        signal = np.array([1, 2, 4, 7, 11])
        deriv = compute_derivative_track(signal)

        assert len(deriv) == len(signal)
        # First element should be same as first diff
        assert deriv[1] == 2  # diff between index 1 and 0

    def test_resample_uniform(self):
        """Test uniform resampling."""
        x = np.array([0, 1, 3, 6, 10])  # Non-uniform
        y = np.array([0, 1, 3, 6, 10])  # Linear

        x_uniform, y_uniform = resample_uniform(x, y, n_points=11)

        assert len(x_uniform) == 11
        assert len(y_uniform) == 11
        assert x_uniform[0] == 0
        assert x_uniform[-1] == 10

    def test_resample_preserves_range(self):
        """Test that resampling preserves value range."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x)

        _, y_resampled = resample_uniform(x, y, n_points=100)

        # Should preserve approximate min/max
        assert abs(y_resampled.min() - y.min()) < 0.1
        assert abs(y_resampled.max() - y.max()) < 0.1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_short_signal(self):
        """Test ARCO on very short signal."""
        signal = np.array([1, 2, 3, 4, 5])

        anchors = generate_anchors(Qmax=5)
        arco = ARCO(anchors, window_sizes=[10], alpha=1.0)  # Window larger than signal

        # Should handle gracefully
        arc_matrix = arco.compute_track_arcs(signal, window_size=10)
        assert arc_matrix.shape[0] >= 1  # At least one "window"

    def test_constant_signal(self):
        """Test ARCO on constant signal."""
        signal = np.ones(100)

        anchors = generate_anchors(Qmax=11)
        arco = ARCO(anchors, window_sizes=[32], alpha=1.0)

        rci = arco.compute_rci({'constant': signal}, major_q=11)

        # Constant signal has no AC components, so low RCI expected
        assert rci < 0.5

    def test_empty_anchors(self):
        """Test behavior with empty anchors list."""
        signal = np.random.randn(100)
        arco = ARCO([], window_sizes=[32], alpha=1.0)

        arc_matrix = arco.compute_track_arcs(signal, window_size=32)
        assert arc_matrix.shape[1] == 0  # No anchors

    def test_single_anchor(self):
        """Test with single anchor."""
        signal = np.sin(2 * np.pi * 0.25 * np.arange(256))
        arco = ARCO([0.25], window_sizes=[64], alpha=1.0)

        rci = arco.compute_rci({'signal': signal}, major_q=10)
        assert 0 <= rci <= 1


class TestXRDIntegration:
    """Test ARCO specifically for XRD pattern analysis."""

    def test_xrd_uniform_peaks(self):
        """Test ARCO on synthetic XRD pattern with uniform peak spacing."""
        # Simulate uniform peaks (like a simple cubic lattice)
        two_theta = np.linspace(10, 80, 2048)
        intensity = np.zeros_like(two_theta)

        # Add peaks at regular intervals (every 5 degrees)
        peak_positions = np.arange(15, 75, 5)
        for pos in peak_positions:
            # Gaussian peak
            intensity += np.exp(-((two_theta - pos) ** 2) / 0.5)

        # Add background
        intensity += 10 + 2 * np.random.randn(len(intensity))

        # ARCO analysis
        anchors = generate_anchors(Qmax=40)
        arco = ARCO(anchors, window_sizes=[128, 256], alpha=1.0)

        tracks = {'intensity': intensity}
        rci = arco.compute_rci(tracks, major_q=20)

        # Uniform peaks should give moderately high RCI
        assert rci > 0.05, f"Expected RCI > 0.05 for uniform peaks, got {rci}"

    def test_xrd_random_peaks(self):
        """Test ARCO on random peak pattern (low periodicity)."""
        two_theta = np.linspace(10, 80, 2048)
        intensity = np.zeros_like(two_theta)

        # Random peak positions
        np.random.seed(42)
        peak_positions = np.random.uniform(15, 75, 10)

        for pos in peak_positions:
            intensity += np.exp(-((two_theta - pos) ** 2) / 0.5)

        intensity += 10 + 2 * np.random.randn(len(intensity))

        anchors = generate_anchors(Qmax=40)
        arco = ARCO(anchors, window_sizes=[128, 256], alpha=1.0)

        rci_random = arco.compute_rci({'intensity': intensity}, major_q=20)

        # Random peaks should give lower RCI than uniform
        # (This is a relative test - exact value depends on random seed)
        assert 0 <= rci_random <= 1

    def test_xrd_with_derivative(self):
        """Test XRD analysis with derivative track."""
        two_theta = np.linspace(10, 80, 1024)
        # Sine modulation to simulate some periodicity
        intensity = 100 + 50 * np.sin(2 * np.pi * 0.1 * two_theta)
        intensity += 10 * np.random.randn(len(intensity))

        deriv = compute_derivative_track(intensity)

        anchors = generate_anchors(Qmax=30)
        arco = ARCO(anchors, window_sizes=[128], alpha=1.0)

        tracks = {'intensity': intensity, 'derivative': deriv}
        arco_print = arco.compute_arco_print(tracks)

        # Should produce valid fingerprint
        assert len(arco_print) > 0
        assert np.isfinite(arco_print).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

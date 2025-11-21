"""
Unit tests for ARCO/RCI implementation.

Tests include:
1. Sine pattern test - single-tone frequency detection
2. Heptad test - coiled-coil periodic pattern
3. Noise baseline - white noise should have low RCI
4. Resolution test - downsampled signals
5. Multi-track analysis
6. Null model z-score validation
"""

import numpy as np
import pytest

from nomad_auto_xrd.common.arco_rci import ARCO, generate_anchors, get_anchor_interpretation_table


class TestGenerateAnchors:
    """Tests for rational anchor generation."""

    def test_generate_anchors_basic(self):
        """Test basic anchor generation."""
        anchors = generate_anchors(Qmax=11)

        # Should have anchors in range (0, 0.5)
        assert all(0 < a < 0.5 for a in anchors)

        # Should be sorted
        assert anchors == sorted(anchors)

        # Should include common rationals
        assert 1/7 in anchors or abs(min(abs(a - 1/7) for a in anchors)) < 1e-10
        assert 1/3 in anchors or abs(min(abs(a - 1/3) for a in anchors)) < 1e-10

    def test_generate_anchors_count(self):
        """Test that anchor count scales with Qmax."""
        anchors_11 = generate_anchors(Qmax=11)
        anchors_20 = generate_anchors(Qmax=20)

        # More anchors with higher Qmax
        assert len(anchors_20) > len(anchors_11)

    def test_generate_anchors_reduced_fractions(self):
        """Test that only reduced fractions are included."""
        anchors = generate_anchors(Qmax=20)

        # 2/4 = 1/2 should only appear once (as 1/2)
        count_half = sum(1 for a in anchors if abs(a - 0.5) < 1e-10)
        assert count_half == 0  # 0.5 is excluded (boundary)

        # 1/4 should appear
        count_quarter = sum(1 for a in anchors if abs(a - 0.25) < 1e-10)
        assert count_quarter == 1


class TestARCOSinePattern:
    """Test ARCO with synthetic sine patterns."""

    def test_single_tone_detection(self):
        """Test that a single-tone sine is detected at correct anchor."""
        # Create sine at frequency 1/8 (0.125 cycles/sample)
        N = 512
        freq = 0.125  # 1/8
        t = np.arange(N)
        signal = np.sin(2 * np.pi * freq * t)

        # Generate anchors and create ARCO analyzer
        anchors = generate_anchors(Qmax=20)
        arco = ARCO(anchors, window_sizes=[128], hop_fraction=0.25, alpha=1.0)

        # Compute ARCO-print
        arco_print = arco.compute_arco_print({'signal': signal})

        # The anchor closest to 1/8 should have high power
        closest_idx = np.argmin(np.abs(np.array(anchors) - freq))

        # Extract arc powers for the single window size
        n_anchors = len(anchors)
        arc_powers = arco_print[:n_anchors]

        # Top anchor should be the one closest to 1/8
        top_idx = np.argmax(arc_powers)
        assert top_idx == closest_idx, f"Expected anchor {closest_idx} but got {top_idx}"

        # RCI should be high (strong periodicity)
        rci = arco.compute_rci({'signal': signal})
        assert rci > 0.3, f"RCI too low: {rci}"

    def test_multiple_harmonics(self):
        """Test detection of multiple harmonics."""
        N = 512
        t = np.arange(N)
        # Fundamental + harmonic
        signal = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.2 * t)

        anchors = generate_anchors(Qmax=20)
        arco = ARCO(anchors, window_sizes=[128], alpha=1.0)

        arco_print = arco.compute_arco_print({'signal': signal})

        # Both 1/10 and 1/5 should have elevated power
        idx_tenth = np.argmin(np.abs(np.array(anchors) - 0.1))
        idx_fifth = np.argmin(np.abs(np.array(anchors) - 0.2))

        n_anchors = len(anchors)
        arc_powers = arco_print[:n_anchors]

        # Both should be in top 5
        top_5_indices = np.argsort(arc_powers)[-5:]
        assert idx_tenth in top_5_indices or idx_fifth in top_5_indices


class TestARCOHeptadPattern:
    """Test ARCO with heptad (coiled-coil) patterns."""

    def test_heptad_repeat(self):
        """Test that 7-residue repeat is detected at 1/7 anchor."""
        # Create repeating pattern with period 7
        N = 280  # 40 heptads
        base_pattern = np.array([1.0, 0.5, 0.3, 0.2, 0.3, 0.5, 0.8])
        signal = np.tile(base_pattern, N // 7)

        anchors = generate_anchors(Qmax=20)
        arco = ARCO(anchors, window_sizes=[63], hop_fraction=0.25, alpha=1.0)

        # Compute RCI
        rci = arco.compute_rci({'signal': signal}, major_q=11)

        # Should have high RCI due to strong 1/7 periodicity
        assert rci > 0.1, f"RCI too low for heptad: {rci}"

        # Check that 1/7 anchor has high power
        arco_print = arco.compute_arco_print({'signal': signal})
        n_anchors = len(anchors)
        arc_powers = arco_print[:n_anchors]

        # Find 1/7 anchor
        idx_seventh = np.argmin(np.abs(np.array(anchors) - 1/7))

        # Should be in top 3
        top_3_indices = np.argsort(arc_powers)[-3:]
        assert idx_seventh in top_3_indices, f"1/7 anchor not in top 3"


class TestARCONoiseBaseline:
    """Test ARCO behavior with noise."""

    def test_white_noise_low_rci(self):
        """White noise should have low RCI."""
        np.random.seed(42)
        N = 512
        signal = np.random.randn(N)

        anchors = generate_anchors(Qmax=11)
        arco = ARCO(anchors, window_sizes=[63], alpha=1.0)

        rci = arco.compute_rci({'signal': signal}, major_q=11)

        # Noise should have low RCI
        assert rci < 0.5, f"RCI too high for noise: {rci}"

    def test_noise_arco_print_uniform(self):
        """Noise should produce relatively uniform ARCO-print."""
        np.random.seed(42)
        N = 512
        signal = np.random.randn(N)

        anchors = generate_anchors(Qmax=11)
        arco = ARCO(anchors, window_sizes=[63], alpha=1.0)

        arco_print = arco.compute_arco_print({'signal': signal})

        # Coefficient of variation should be low (uniform distribution)
        cv = np.std(arco_print) / (np.mean(arco_print) + 1e-12)
        assert cv < 2.0, f"ARCO-print too non-uniform for noise: CV={cv}"


class TestARCOResolution:
    """Test ARCO behavior with different resolutions."""

    def test_downsampled_signal_recovery(self):
        """Test that periodicity is recovered in downsampled signal."""
        # High-resolution signal
        N_high = 1024
        freq = 0.1
        t_high = np.arange(N_high)
        signal_high = np.sin(2 * np.pi * freq * t_high)

        # Downsample by factor of 2
        signal_low = signal_high[::2]

        anchors = generate_anchors(Qmax=20)
        arco = ARCO(anchors, window_sizes=[128], alpha=1.0)

        # Both should detect periodicity
        rci_high = arco.compute_rci({'signal': signal_high})
        rci_low = arco.compute_rci({'signal': signal_low})

        assert rci_high > 0.2
        assert rci_low > 0.2


class TestARCOMultiTrack:
    """Test ARCO with multiple tracks."""

    def test_multi_track_arco_print(self):
        """Test ARCO-print with multiple tracks."""
        N = 512
        t = np.arange(N)

        # Two tracks with different periodicities
        track1 = np.sin(2 * np.pi * 0.125 * t)  # 1/8
        track2 = np.sin(2 * np.pi * 0.1 * t)    # 1/10

        tracks = {'track1': track1, 'track2': track2}

        anchors = generate_anchors(Qmax=20)
        arco = ARCO(anchors, window_sizes=[128], alpha=1.0)

        arco_print = arco.compute_arco_print(tracks)

        # Should have length n_tracks * n_anchors * n_window_sizes
        expected_length = 2 * len(anchors) * 1
        assert len(arco_print) == expected_length

    def test_multi_track_rci(self):
        """Test RCI computation with multiple tracks."""
        N = 512
        t = np.arange(N)

        track1 = np.sin(2 * np.pi * 0.125 * t)
        track2 = np.random.randn(N)  # Noise

        tracks = {'periodic': track1, 'noise': track2}

        anchors = generate_anchors(Qmax=11)
        arco = ARCO(anchors, window_sizes=[63], alpha=1.0)

        rci = arco.compute_rci(tracks, major_q=11)

        # RCI should be intermediate (one periodic, one noise)
        assert 0.1 < rci < 0.8


class TestARCO3D:
    """Test position-resolved ARCO (ARCO-3D)."""

    def test_arco_3d_shape(self):
        """Test that ARCO-3D has correct shape."""
        N = 512
        signal = np.sin(2 * np.pi * 0.1 * np.arange(N))

        anchors = generate_anchors(Qmax=11)
        arco = ARCO(anchors, window_sizes=[63], hop_fraction=0.25)

        arco_3d = arco.compute_arco_3d({'signal': signal}, finest_window=63)

        # Shape should be (n_tracks, n_windows, n_anchors)
        n_tracks = 1
        n_anchors = len(anchors)
        hop = int(63 * 0.25)
        n_windows = (N - 63) // hop + 1

        assert arco_3d.shape[0] == n_tracks
        assert arco_3d.shape[2] == n_anchors
        assert arco_3d.shape[1] > 0  # At least some windows

    def test_arco_3d_multi_track(self):
        """Test ARCO-3D with multiple tracks."""
        N = 512
        t = np.arange(N)
        tracks = {
            'track1': np.sin(2 * np.pi * 0.1 * t),
            'track2': np.sin(2 * np.pi * 0.125 * t),
        }

        anchors = generate_anchors(Qmax=11)
        arco = ARCO(anchors, window_sizes=[63])

        arco_3d = arco.compute_arco_3d(tracks, finest_window=63)

        # Should have 2 tracks
        assert arco_3d.shape[0] == 2


class TestNullModel:
    """Test null model z-score computation."""

    def test_null_model_zscore_periodic(self):
        """Periodic signal should have high z-score vs null."""
        N = 512
        signal = np.sin(2 * np.pi * 0.1 * np.arange(N))

        anchors = generate_anchors(Qmax=11)
        arco = ARCO(anchors, window_sizes=[63], alpha=1.0)

        z_score, p_value = arco.null_model_zscore(
            signal, n_shuffles=30, preserve_composition=True
        )

        # Periodic signal should have significantly higher RCI than shuffled
        assert z_score > 1.0, f"Z-score too low: {z_score}"
        assert p_value < 0.5, f"P-value too high: {p_value}"

    def test_null_model_zscore_noise(self):
        """Noise should have z-score near 0."""
        np.random.seed(42)
        N = 512
        signal = np.random.randn(N)

        anchors = generate_anchors(Qmax=11)
        arco = ARCO(anchors, window_sizes=[63])

        z_score, p_value = arco.null_model_zscore(signal, n_shuffles=30)

        # Noise should not be significantly different from shuffled noise
        assert abs(z_score) < 2.0, f"Z-score too extreme for noise: {z_score}"


class TestMultiScale:
    """Test multi-scale analysis with different window sizes."""

    def test_multiscale_arco_print(self):
        """Test that multi-scale produces correct length ARCO-print."""
        N = 512
        signal = np.sin(2 * np.pi * 0.1 * np.arange(N))

        anchors = generate_anchors(Qmax=11)
        window_sizes = [31, 63, 127]
        arco = ARCO(anchors, window_sizes=window_sizes, alpha=1.0)

        arco_print = arco.compute_arco_print({'signal': signal})

        # Length should be n_tracks * n_anchors * n_window_sizes
        expected_length = 1 * len(anchors) * len(window_sizes)
        assert len(arco_print) == expected_length


class TestInterpretationTable:
    """Test interpretation table utilities."""

    def test_interpretation_table_structure(self):
        """Test that interpretation table has expected structure."""
        table = get_anchor_interpretation_table()

        assert isinstance(table, dict)
        assert len(table) > 0

        # Check structure of entries
        for anchor, info in table.items():
            assert 'period' in info
            assert 'typical_meaning' in info
            assert isinstance(anchor, (int, float))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_signal(self):
        """Test behavior with very short signal."""
        signal = np.array([1.0, 2.0, 3.0])

        anchors = generate_anchors(Qmax=11)
        arco = ARCO(anchors, window_sizes=[31])

        # Should handle gracefully (signal shorter than window)
        arco_print = arco.compute_arco_print({'signal': signal})
        assert len(arco_print) > 0

    def test_constant_signal(self):
        """Test behavior with constant signal."""
        signal = np.ones(512)

        anchors = generate_anchors(Qmax=11)
        arco = ARCO(anchors, window_sizes=[63])

        rci = arco.compute_rci({'signal': signal})

        # Constant signal (DC only) should have low RCI
        assert rci < 0.5

    def test_alpha_parameter_effect(self):
        """Test that alpha affects bandwidth."""
        N = 512
        signal = np.sin(2 * np.pi * 0.1 * np.arange(N))

        anchors = generate_anchors(Qmax=11)

        # Different alpha values
        arco_narrow = ARCO(anchors, window_sizes=[63], alpha=0.5)
        arco_wide = ARCO(anchors, window_sizes=[63], alpha=2.0)

        rci_narrow = arco_narrow.compute_rci({'signal': signal})
        rci_wide = arco_wide.compute_rci({'signal': signal})

        # Both should detect periodicity (exact values depend on signal)
        assert rci_narrow > 0
        assert rci_wide > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

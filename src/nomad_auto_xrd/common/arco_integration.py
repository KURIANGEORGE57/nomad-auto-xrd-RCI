"""
Integration utilities for ARCO analysis in XRD pipeline.

This module provides helper functions to compute ARCO features during XRD analysis
and attach them to AnalysisResult objects.
"""

from typing import Optional

import numpy as np

from nomad_auto_xrd.common.models import AnalysisInput, AnalysisResult


def compute_arco_features_for_analysis(
    analysis_inputs: list[AnalysisInput],
    Qmax: int = 40,
    window_sizes: Optional[list[int]] = None,
    alpha: float = 0.5,
    major_q: int = 20,
    enable_arco: bool = True,
) -> list[dict]:
    """
    Compute ARCO features for a list of XRD patterns.

    This function can be called during XRD analysis to compute ARCO fingerprints
    and RCI values for each pattern.

    Args:
        analysis_inputs: List of AnalysisInput objects containing XRD patterns.
        Qmax: Maximum denominator for rational anchors (30-60 recommended for XRD).
        window_sizes: Window sizes for multi-scale analysis (default: [128, 256]).
        alpha: Bandwidth scale factor (default: 0.5).
        major_q: Threshold denominator for "major" rationals (default: 20).
        enable_arco: Whether to compute ARCO features (default: True).

    Returns:
        List of dictionaries containing ARCO features for each pattern.
        Each dictionary contains:
            - 'rci': Rational Coherence Index (float)
            - 'arco_print': ARCO fingerprint vector (serialized as list)
            - 'top_rationals': Top 5 rational frequencies
            - 'anchors_used': Number of rational anchors

    Example:
        >>> arco_features = compute_arco_features_for_analysis(
        ...     analysis_inputs,
        ...     Qmax=40,
        ...     alpha=0.5
        ... )
        >>> for idx, features in enumerate(arco_features):
        ...     print(f"Pattern {idx}: RCI = {features['rci']:.4f}")
    """
    if not enable_arco:
        return []

    try:
        from nomad_auto_xrd.lib.arco_analysis import XRDArcoAnalyzer
    except ImportError:
        # ARCO library not available
        return []

    # Initialize ARCO analyzer
    analyzer = XRDArcoAnalyzer(
        Qmax=Qmax,
        window_sizes=window_sizes,
        alpha=alpha,
        major_q=major_q,
        use_derivative=True,
    )

    arco_features_list = []

    for analysis_input in analysis_inputs:
        two_theta = np.array(analysis_input.two_theta)
        intensity = np.array(analysis_input.intensity)

        try:
            # Compute ARCO features
            result = analyzer.analyze_pattern(two_theta, intensity)

            # Serialize for storage (convert numpy arrays to lists)
            features = {
                'rci': float(result['rci']),
                'arco_print': result['arco_print'].tolist(),
                'top_rationals': [
                    {
                        'frequency': float(freq),
                        'power': float(power),
                        'denominator': int(denom),
                    }
                    for freq, power, denom in result['top_rationals'][:5]
                ],
                'anchors_used': len(analyzer.anchors),
                'window_sizes': analyzer.window_sizes,
                'alpha': analyzer.alpha,
                'major_q': analyzer.major_q,
            }

            arco_features_list.append(features)

        except Exception as e:
            # If ARCO computation fails, append None
            arco_features_list.append({
                'error': str(e),
                'rci': None,
            })

    return arco_features_list


def attach_arco_to_result(
    result: AnalysisResult,
    analysis_inputs: list[AnalysisInput],
    enable_arco: bool = True,
    **arco_kwargs,
) -> AnalysisResult:
    """
    Compute ARCO features and attach them to an AnalysisResult.

    This is a convenience function that computes ARCO features and adds them
    to an existing AnalysisResult object.

    Args:
        result: AnalysisResult object to attach ARCO features to.
        analysis_inputs: List of AnalysisInput objects.
        enable_arco: Whether to compute ARCO (default: True).
        **arco_kwargs: Additional keyword arguments for ARCO computation.

    Returns:
        AnalysisResult with arco_features populated.

    Example:
        >>> result = analyzer.eval(analysis_inputs)
        >>> result = attach_arco_to_result(result, analysis_inputs, enable_arco=True)
        >>> print(result.arco_features[0]['rci'])
    """
    if enable_arco:
        arco_features = compute_arco_features_for_analysis(
            analysis_inputs, enable_arco=True, **arco_kwargs
        )
        result.arco_features = arco_features

    return result


# Integration example for XRDAutoAnalyzer
# ----------------------------------------
# To integrate ARCO into the existing XRDAutoAnalyzer.eval() method,
# add the following lines after analysis is complete:
#
# # Import ARCO integration
# from nomad_auto_xrd.common.arco_integration import attach_arco_to_result
#
# # In XRDAutoAnalyzer.eval(), after all_results is populated:
# all_results = attach_arco_to_result(
#     all_results,
#     analysis_inputs,
#     enable_arco=True,  # Can be controlled by analysis settings
#     Qmax=40,
#     alpha=0.5,
#     major_q=20
# )
#
# This will add ARCO features to the result without modifying
# existing functionality.

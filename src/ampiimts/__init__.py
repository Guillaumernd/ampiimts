"""Convenience imports for the :mod:`ampiimts` package.

This module exposes the most commonly used functions so they can be
imported directly from :mod:`ampiimts`.
"""

from .pre_processed import (
    interpolate,
    _compute_aswn,
    aswn_with_trend,
    normalization,
    pre_processed,
    missing_values,
    define_m_using_clustering,
)
from .plotting import (
    plot_multiple_dfs,
    plot_all_variables_multiple_dfs,
    plot_patterns_and_discords,
)
from .matrix_profile import matrix_profile
from .motif_pattern import discover_patterns_stumpy_mixed

__all__ = [
    "interpolate",
    "_compute_aswn",
    "aswn_with_trend",
    "normalization",
    "pre_processed",
    "missing_values",
    "define_m_using_clustering",
    "plot_multiple_dfs",
    "plot_all_variables_multiple_dfs",
    "plot_patterns_and_discords",
    "matrix_profile",
    "discover_patterns_stumpy_mixed",
]


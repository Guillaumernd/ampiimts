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
    define_m_using_clustering,
    synchronize_on_common_grid,
)

from .plotting import (
    plot_patterns_and_discords,
    plot_multidim_patterns_and_discords,
    plot_motif_overlays,
    plot_all_motif_overlays,
    plot_all_patterns_and_discords,
)

from .matrix_profile import matrix_profile

from .motif_pattern import (
    discover_patterns_stumpy_mixed,
    discover_patterns_mstump_mixed,
)

from .ampiimts import (
    ampiimts
)

__all__ = [
    "interpolate",
    "synchronize_on_common_grid",
    "_compute_aswn",
    "aswn_with_trend",
    "normalization",
    "pre_processed",
    "define_m_using_clustering",
    "plot_patterns_and_discords",
    "matrix_profile",
    "plot_multidim_patterns_and_discords",
    "discover_patterns_stumpy_mixed",
    "discover_patterns_mstump_mixed",
    "plot_motif_overlays",
    "plot_all_motif_overlays",
    "plot_all_patterns_and_discords",
    "ampiimts",
]

"""Expose frequently used functions at the package root.

Importing :mod:`ampiimts` makes the main helpers available directly
without referencing their submodules.
"""

from .pre_processed import (
    interpolate,
    _compute_aswn,
    aswn_with_trend,
    normalization,
    pre_processed,
    define_m,
    synchronize_on_common_grid,
    interpolate_all_columns_by_similarity,
)

from .plotting import (
    plot_multidim_patterns_and_discords,
    plot_motif_overlays,
    plot_all_motif_overlays,
    plot_all_patterns_and_discords,
)

from .matrix_profile import (
    matrix_profile,
    matrix_profile_process,
)

from .motif_pattern import (
    discover_patterns_stumpy_mixed,
    discover_patterns_mstump_mixed,
    find_discords,
    compute_univariate_matrix_profiles,
)

from .ampiimts import (
    ampiimts,
    process
)

__all__ = [
    "interpolate",
    "synchronize_on_common_grid",
    "_compute_aswn",
    "aswn_with_trend",
    "normalization",
    "pre_processed",
    "define_m",
    "interpolate_all_columns_by_similarity",
    "matrix_profile",
    "plot_multidim_patterns_and_discords",
    "discover_patterns_stumpy_mixed",
    "discover_patterns_mstump_mixed",
    "plot_motif_overlays",
    "plot_all_motif_overlays",
    "plot_all_patterns_and_discords",
    "ampiimts",
    "process",
    "matrix_profile_process",
    "find_discords",
    "compute_univariate_matrix_profiles",
]

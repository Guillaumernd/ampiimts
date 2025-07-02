"""High level pipeline for motif and discord discovery.

The function reads one or several time series, preprocesses them and
computes the matrix profile in order to detect motifs and discords.
"""
from typing import Tuple, Union, List, Dict, Any
from .matrix_profile import (
    matrix_profile,
)
from .pre_processed import (
    pre_processed,
)
from .plotting import (
    plot_all_patterns_and_discords,
    plot_all_motif_overlays,
)
import os
import pandas as pd

def ampiimts(
    data: Union[pd.DataFrame, List[pd.DataFrame]],
    gap_multiplier: float = 15,
    min_std: float = 1e-2,
    min_valid_ratio: float = 0.8,
    alpha: float = 0.65,
    window_size: str = None,
    sort_by_variables: bool = False,
    cluster: bool = True,
    top_k_cluster: int = 4,
    visualize: bool = True,
    max_motifs: int = 5,
    discord_top_pct: float = 0.04,
    max_matches: int = 10,
    motif: bool = False,
    max_len: int = None,
) -> Tuple[
    Union[pd.DataFrame, List[pd.DataFrame]],
    Union[pd.DataFrame, List[pd.DataFrame]],
    Union[Dict[str, Any], List[Dict[str, Any]]]
]:
    """Complete motif and discord analysis pipeline.

    Parameters
    ----------
    data : pandas.DataFrame or list of DataFrame
        Input dataset(s) or a path to CSV files.
    gap_multiplier : float, optional
        Gap multiplier used during interpolation.
    min_std : float, optional
        Minimum allowed standard deviation for normalization.
    min_valid_ratio : float, optional
        Minimum fraction of valid values in a window.
    alpha : float, optional
        Weight of the trend component in ASWN normalization.
    window_size : str or None, optional
        Sliding window size; if ``None`` it is inferred.
    sort_by_variables : bool, optional
        Sort variables by variance prior to clustering.
    cluster : bool, optional
        Whether to cluster variables before computing the profile.
    top_k_cluster : int, optional
        Maximum number of clusters retained.
    visualize : bool, optional
        If ``True`` plots of the results are shown.
    max_motifs : int, optional
        Maximum number of motifs to return.
    discord_top_pct : float, optional
        Fraction of highest profile values considered discords.
    max_matches : int, optional
        Maximum number of motif matches returned.
    motif : bool, optional
        Whether to extract motifs in addition to discords.
    max_len : int or None, optional
        Maximum number of rows loaded from each file.

    Returns
    -------
    tuple
        ``(interpolated, normalized, result)`` containing processed
        dataframes and the matrix profile result.
    """

    if os.path.isdir(data):  # data is a directory
        pds = []
        with os.scandir(data) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.csv'):
                    try:
                        df = pd.read_csv(os.path.join(data, entry.name))
                        if max_len  is None:
                            max_len = len(df)
                        pds.append(df.iloc[:max_len])  # keep only the first rows
                    except Exception:
                        continue
    else:
        if max_len is None:
            max_len = len(data)
        pds = data.iloc[:max_len]


    # --- Merge all files and preprocess (interpolation, normalization, optional clustering) ---
    pds_interpolated, pds_normalized = pre_processed(
        pds,
        gap_multiplier=gap_multiplier,
        min_std=min_std,
        min_valid_ratio=min_valid_ratio,
        alpha=alpha,
        window_size=window_size,
        sort_by_variables=sort_by_variables,
        cluster=cluster,
        top_k_cluster=top_k_cluster,
    )

    # --- Compute the matrix profile ---
    matrix_profile_result = matrix_profile(
        pds_normalized,
        n_jobs=4,
        max_motifs=max_motifs,
        discord_top_pct=discord_top_pct,
        max_matches=max_matches,
        cluster=cluster,
        motif=motif
    )

    if visualize:
        # --- Visualization ---
        plot_all_patterns_and_discords(pds_interpolated, matrix_profile_result)
        plot_all_motif_overlays(pds_interpolated, matrix_profile_result)
    
    return pds_interpolated, pds_normalized, matrix_profile_result

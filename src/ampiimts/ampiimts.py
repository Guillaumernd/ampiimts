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

def process(
    pds: Union[pd.DataFrame, List[pd.DataFrame]],
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
    group_size: int = 6,
    display_info: bool = False,
    most_stable_only: bool = False,
) -> Tuple[
    Union[pd.DataFrame, List[pd.DataFrame]],
    Union[pd.DataFrame, List[pd.DataFrame]],
    Union[Dict[str, Any], List[Dict[str, Any]]]
]:
    """Process one or several DataFrames and return analysis results.

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
    group_size : int, optional
        Target group size for hierarchical clustering.
    display_info : bool, optional
        Display informations about data.
    most_stable_only : bool, optional
        ``True`` to extract the most stable sensor
        
    Returns
    -------
    tuple
        ``(interpolated, normalized, result)`` containing processed
        dataframes and the matrix profile result.
    """

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
        group_size=group_size,
        display_info=display_info,
    )

    matrix_profile_result = matrix_profile(
        pds_normalized,
        n_jobs=4,
        max_motifs=max_motifs,
        discord_top_pct=discord_top_pct,
        max_matches=max_matches,
        cluster=cluster,
        motif=motif,
        most_stable_only=most_stable_only,
    )

    if visualize:
        plot_all_patterns_and_discords(pds_interpolated, matrix_profile_result)
        plot_all_motif_overlays(pds_interpolated, matrix_profile_result)

    return pds_interpolated, pds_normalized, matrix_profile_result


def ampiimts(
    data: Union[str, pd.DataFrame, List[pd.DataFrame]],
    gap_multiplier: float = 15,
    min_std: float = 1e-2,
    min_valid_ratio: float = 0.8,
    alpha: float = 0.65,
    window_size: str = None,
    sort_by_variables: bool = False,
    cluster: bool = True,
    top_k_cluster: int = 4,
    visualize: bool = True,
    max_motifs: int = 10,
    discord_top_pct: float = 0.04,
    max_matches: int = 30,
    motif: bool = False,
    max_len: int = None,
    group_size: int = None,
    display_info: bool = False,
    most_stable_only: bool = False,
) -> Tuple[
    Union[pd.DataFrame, List[pd.DataFrame]],
    Union[pd.DataFrame, List[pd.DataFrame]],
    Union[Dict[str, Any], List[Dict[str, Any]]]
]:
    """Complete motif and discord analysis pipeline."""

    if isinstance(data, str) and os.path.isdir(data):
        pds = []
        with os.scandir(data) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.csv'):
                    try:
                        df = pd.read_csv(os.path.join(data, entry.name))
                        pds.append(df.iloc[:max_len] if max_len else df)
                    except Exception:
                        continue
        return process(
            pds,
            gap_multiplier,
            min_std,
            min_valid_ratio,
            alpha,
            window_size,
            sort_by_variables,
            cluster,
            top_k_cluster,
            visualize,
            max_motifs,
            discord_top_pct,
            max_matches,
            motif,
            group_size,
            display_info,
            most_stable_only,
        )

    elif isinstance(data, pd.DataFrame):
        df = data.iloc[:max_len] if max_len else data
        return process(
            df,
            gap_multiplier,
            min_std,
            min_valid_ratio,
            alpha,
            window_size,
            sort_by_variables,
            cluster,
            top_k_cluster,
            visualize,
            max_motifs,
            discord_top_pct,
            max_matches,
            motif,
            group_size,
            display_info,
            most_stable_only
        )

    else:
        raise TypeError(
            "[ERROR] `data` must be a path to a folder, a DataFrame or a list of DataFrames."
        )

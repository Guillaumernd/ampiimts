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
    interpolate_all_columns_by_similarity,
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
    smart_interpolation: bool = True,
    printunidimensional: bool = False,
    only_heat_map: bool = True,
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
    smart_interpolation : bool, optional
        interpolation with matrix_profile via other
        similar sensors
    printunidimensional : bool, optional
        See unidimensional matri_profil
    only_heat_map : bool, optional
        only print heatmap

    Returns
    -------
    tuple
        ``(interpolated, normalized, result)`` containing processed
        dataframes and the matrix profile result.
    """
    # Si DataFrame unidimensionnel → désactive automatiquement les traitements multivariés
    if isinstance(pds, pd.DataFrame):
        numeric_cols = pds.select_dtypes(include=[float, int]).columns
        if len(numeric_cols) == 1:
            cluster = False
            smart_interpolation = False
            most_stable_only = False
        
    pds_interpolated, pds_normalized, pds_normalized_without_nan = pre_processed(
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
        smart_interpolation=smart_interpolation
    )

    if pds_normalized is None:
        return None, None, None

    smart_interpolation, most_stable_only = (False, False) if not cluster else (
        smart_interpolation, most_stable_only)
    most_stable_only2 = True if most_stable_only and smart_interpolation else False
    most_stable_only = True if not smart_interpolation and most_stable_only else False

    matrix_profile_result = matrix_profile(
        pds_normalized,
        n_jobs=4,
        max_motifs=max_motifs,
        discord_top_pct=discord_top_pct,
        max_matches=max_matches,
        cluster=cluster,
        motif=motif,
        most_stable_only=most_stable_only,
        smart_interpolation=smart_interpolation,
        printunidimensional=printunidimensional,
    )
    if matrix_profile_result is None or matrix_profile_result == []:
        return pds_interpolated, pds_normalized, None
    if smart_interpolation:
        pds_normalized = interpolate_all_columns_by_similarity(
            pds=pds_normalized_without_nan,
            matrix_profiles=matrix_profile_result,
        )
            
        matrix_profile_result = matrix_profile(
            pds_normalized,
            n_jobs=4,
            max_motifs=max_motifs,
            discord_top_pct=discord_top_pct,
            max_matches=max_matches,
            cluster=cluster,
            motif=motif,
            most_stable_only=most_stable_only2,
            smart_interpolation=False,
            printunidimensional=printunidimensional,
        )

    if visualize:
        plot_all_patterns_and_discords(pds_interpolated, matrix_profile_result, only_heat_map=only_heat_map)
        if motif:
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
    group_size: int = 16,
    display_info: bool = False,
    most_stable_only: bool = False,
    smart_interpolation: bool = False,
    printunidimensional: bool = False,
    only_heat_map: bool = True,
) -> Tuple[
    Union[pd.DataFrame, List[pd.DataFrame]],
    Union[pd.DataFrame, List[pd.DataFrame]],
    Union[Dict[str, Any], List[Dict[str, Any]]]
]:
    """
    Analyze time series data using motif/discord discovery and matrix profile.

    This function processes one or multiple time series (either univariate or multivariate),
    performing preprocessing, optional clustering, and extracting motifs and/or discords using
    matrix profile analysis. It supports visualization and multiple data input formats.

    Parameters
    ----------
    data : str or pandas.DataFrame or list of DataFrame
        Either a path to a CSV file or a directory of CSVs, a single pandas DataFrame,
        or a list of DataFrames. Each DataFrame must be indexed by a time column.
    gap_multiplier : float, default=15
        Multiplier for defining acceptable gaps in time during interpolation.
    min_std : float, default=1e-2
        Minimum standard deviation threshold to consider a signal valid for analysis.
    min_valid_ratio : float, default=0.8
        Minimum required ratio of valid (non-NaN) points in a rolling window.
    alpha : float, default=0.65
        Trend contribution weight for ASWN (Adaptive Sliding Window Normalization).
    window_size : str or None, optional
        If provided, sets the sliding window size. If None, it will be estimated automatically.
    sort_by_variables : bool, default=False
        Whether to sort variables by variance before clustering or profile computation.
    cluster : bool, default=True
        If True, cluster sensors into groups before computing matrix profiles (recommended for multivariate).
    top_k_cluster : int, default=4
        Number of top clusters to retain when clustering is enabled.
    visualize : bool, default=True
        If True, display visual plots including motifs, discords, and heatmaps.
    max_motifs : int, default=10
        Maximum number of motifs to return per time series or group.
    discord_top_pct : float, default=0.04
        Top percentage of discord scores to consider as anomalies.
    max_matches : int, default=30
        Maximum number of motif matches returned per motif.
    motif : bool, default=False
        Whether to perform motif search in addition to discord detection.
    max_len : int or None, optional
        Maximum number of rows to read per DataFrame. If None, all data is loaded.
    group_size : int, default=16
        Maximum group size for clustering sensors when `cluster=True`.
    display_info : bool, default=False
        Print detailed info about dataset dimensions, gaps, and preprocessing steps.
    most_stable_only : bool, default=False
        If True, retain only the most stable sensor per group (based on matrix profile stability).
    smart_interpolation : bool, default=False
        Use structure-aware interpolation by leveraging similar sensors via matrix profile.
    printunidimensional : bool, default=False
        If True, output the matrix profile of each univariate signal to the console.
    only_heat_map : bool, default=True
        If True, suppresses motif/discord plots and shows only the heatmap visualization.

    Returns
    -------
    interpolated : pandas.DataFrame or list of DataFrame
        Interpolated version(s) of the input time series after resampling and cleaning.
    normalized : pandas.DataFrame or list of DataFrame
        Normalized version(s) of the input time series, post ASWN and preprocessing.
    result : dict or list of dict
        Dictionary (or list of dictionaries) containing the matrix profile results including:
        - detected motifs
        - detected discords
        - best window sizes
        - clustering metadata (if applicable)

    Raises
    ------
    TypeError
        If the input `data` is not a valid path, DataFrame, or list of DataFrames.

    Notes
    -----
    This function wraps preprocessing, normalization, clustering, and matrix profile logic
    into a single unified API for ease of use. Visualization is optional and can be turned off
    for headless environments.

    Examples
    --------
    >>> from ampiimts import ampiimts
    >>> result = ampiimts("data/my_timeseries.csv", motif=True)

    >>> df = pd.read_csv("my.csv", parse_dates=["timestamp"])
    >>> interpolated, normalized, results = ampiimts(df, cluster=False)
    """
    if isinstance(data, str) and os.path.isfile(data):
        df = pd.read_csv(data)
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
            smart_interpolation,
            printunidimensional,
            only_heat_map,
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
            most_stable_only,
            smart_interpolation,
            printunidimensional,
            only_heat_map,
        )
    elif isinstance(data, list) and all(isinstance(d, pd.DataFrame) for d in data):
        pds = [df.iloc[:max_len] if max_len else df for df in data]
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
            smart_interpolation,
            printunidimensional,
            only_heat_map,
        )
    else:
        raise TypeError(
            "[ERROR] `data` must be a path to a folder, a DataFrame or a list of DataFrames."
        )

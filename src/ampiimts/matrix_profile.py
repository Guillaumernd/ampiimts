"""
Module `matrix_profile`: High-level pipeline for motif and anomaly analysis.

This module provides a main interface to analyze patterns and discords in time series,
handling multivariate, noisy, or irregular data with adaptive strategies.

Main Features:
--------------
- Automates the selection and execution of motif/discord detection using `motif_pattern.py`.
- Applies MDL (Minimum Description Length) to select relevant subspaces in multivariate series.
- Integrates preprocessing: interpolation, normalization, clustering, and alignment.
- Supports processing of single or multiple DataFrames.
- Allows visualization and export of the most dominant motifs or anomalies.

Difference with `motif_pattern.py`:
-----------------------------------
- `matrix_profile.py` is a **complete pipeline** that prepares the data, selects parameters,
  runs detection, and post-processes results.
- `motif_pattern.py` provides the **core low-level algorithms** for pattern matching.

Use this module when you want a ready-to-use, fully automated motif/discord detection workflow.
"""

from typing import List, Optional, Union
import pandas as pd
import os
os.environ["NUMBA_NUM_THREADS"] = "8"

from .motif_pattern import (
    discover_patterns_stumpy_mixed,
    discover_patterns_mstump_mixed,
)


def matrix_profile_process(
    df: pd.DataFrame,
    window_size: Optional[int] = None,
    max_motifs: int = 3,
    discord_top_pct: float = 0.04,
    max_matches: int = 10,
    cluster: bool = False,
    motif: bool = False,
    min_mdl_ratio: float = 0.25,
    most_stable_only: bool = False,
    smart_interpolation: bool = True,
    printunidimensional: bool = False,
    group_size: int = 5,
) -> dict:
    """Compute motif and discord information for one DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing only numeric columns.
    window_size : int, optional
        Sliding window size. If ``None`` the value stored in ``df.attrs['m']``
        is used.
    max_motifs : int
        Maximum number of motifs to return.
    discord_top_pct : float
        Percentage of highest profile values considered discords.
    max_matches : int
        Maximum number of motif matches to retrieve.
    cluster : bool
        Whether the dataframe represents a clustered signal.
    motif : bool
        Whether motifs should be extracted in addition to discords.
    min_mdl_ratio : float
        Minimum ratio used when selecting dimensions with MDL.
    most_stable_only : bool, optional
        ``True`` to extract the most stable sensor
    smart_interpolation : bool, optional
        interpolation with matrix_profile via other
        similar sensors
    printunidimensional : bool, optional
        See unidimensional matri_profil
    group_size : int, optional
        limit dimensions with MDL algorithm when
        cluster == False

    Returns
    -------
    dict
        Dictionary containing patterns, discords and the matrix profile.
    """

    # Work on a copy to avoid mutating the caller's DataFrame
    df = df.copy()

    # Use the window size stored in the DataFrame metadata when not given
    window_size = df.attrs["m"][1]
    
    # ==== UNIVARIATE ====
    if df.shape[1] == 1:
        # Use the univariate motif discovery helper
        return discover_patterns_stumpy_mixed(
            df,
            window_size,
            max_motifs=max_motifs,
            discord_top_pct=discord_top_pct,
            max_matches=max_matches,
        )
    # ==== MULTIVARIATE ====
    else:
        return discover_patterns_mstump_mixed(
            df,
            window_size,
            max_motifs=max_motifs,
            discord_top_pct=discord_top_pct,
            max_matches=max_matches,
            cluster=cluster,
            motif=motif,
            min_mdl_ratio=min_mdl_ratio,
            most_stable_only=most_stable_only,
            smart_interpolation=smart_interpolation,
            printunidimensional=printunidimensional,
            group_size=group_size,
        )

def matrix_profile(
    data: Union[pd.DataFrame, List[pd.DataFrame], List[List[pd.DataFrame]]],
    n_jobs: int = 4,
    max_motifs: int = 5,
    discord_top_pct: float = 0.04,
    max_matches: int = 10,
    cluster:bool = False,
    motif:bool =False,
    min_mdl_ratio: float = 0.25,
    most_stable_only: bool = False,
    smart_interpolation: bool = True,
    printunidimensional: bool = False,
    group_size: int = 5,
) -> Union[dict, List[dict], List[List[dict]]]:
    """Compute matrix profiles for one or many DataFrames.

    Parameters
    ----------
    data : DataFrame or list
        Either a single ``pandas.DataFrame`` or a list of DataFrames.
    n_jobs : int
        Number of parallel jobs used when processing multiple DataFrames.
    max_motifs : int
        Maximum number of motifs per profile.
    discord_top_pct : float
        Fraction of the highest profile values considered discords.
    max_matches : int
        Maximum number of matches returned per motif.
    cluster : bool 
        Indicates whether the data came from clustering.
    motif : bool
        If ``True`` motifs are extracted in addition to discords.
    min_mdl_ratio : float
        Minimum MDL ratio when selecting dimensions.
    most_stable_only : bool, optional
        ``True`` to extract the most stable sensor
    smart_interpolation : bool, optional
        interpolation with matrix_profile via other
        similar sensors
    printunidimensional : bool, optional
        See unidimensional matri_profil
    group_size : int, optional
        limit dimensions with MDL algorithm when
        cluster == False

    Returns
    -------
    dict or list
        Matrix profile information matching the structure of ``data``.
    """

    if isinstance(data, pd.DataFrame):
        # Single DataFrame case
        df = data
        window_size = df.attrs["m"][1]

        return matrix_profile_process(
            df,
            window_size=window_size,
            max_motifs=max_motifs,
            motif=motif,
            discord_top_pct=discord_top_pct,
            max_matches=max_matches,
            cluster=cluster,
            most_stable_only=most_stable_only,
            smart_interpolation=smart_interpolation,
            printunidimensional=printunidimensional,
            group_size=group_size,
        )

    else:
        pds = data
        # Flat list of DataFrames
        return [
            matrix_profile_process(
                df,
                max_motifs=max_motifs,
                discord_top_pct=discord_top_pct,
                max_matches=max_matches,
                cluster=cluster,
                motif=motif,
                min_mdl_ratio=min_mdl_ratio,
                most_stable_only=most_stable_only,
                smart_interpolation=smart_interpolation,
                printunidimensional=printunidimensional,
                group_size=group_size,
            )
            for df in pds
        ]
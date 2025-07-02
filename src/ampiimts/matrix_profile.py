"""Utilities for computing matrix profiles."""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import stumpy as sp
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

    Returns
    -------
    dict
        Dictionary containing patterns, discords and the matrix profile.
    """

    # Work on a copy to avoid mutating the caller's DataFrame
    df = df.copy()

    # Remove a timestamp column if present; only numeric data is required
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    # Use the window size stored in the DataFrame metadata when not given
    window_size = df.attrs["m"]
 
    # Validate that all columns contain numeric data
    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
        raise ValueError(
            "All columns must be numeric for matrix profile computation."
        )
    
    # ==== UNIVARIATE ====
    if df.shape[1] == 1:
        try:
                
            # Use the univariate motif discovery helper
            return discover_patterns_stumpy_mixed(
                df,
                window_size,
                max_motifs=max_motifs,
                discord_top_pct=discord_top_pct,
                max_matches=max_matches,
            )
        except ValueError as e:
            print(f"[MatrixProfile Warning] failed for window {window_size}: {e}")
            return None



    # ==== MULTIVARIATE ====
    try:
        return discover_patterns_mstump_mixed(
            df,
            window_size,
            max_motifs=max_motifs,
            discord_top_pct=discord_top_pct,
            max_matches=max_matches,
            cluster=cluster,
            motif=motif,
            min_mdl_ratio=min_mdl_ratio,
        )
    except ValueError as e:
        print(f"[MatrixProfile Warning] failed for window {window_size}: {e}")
        return None

def matrix_profile(
    data: Union[pd.DataFrame, List[pd.DataFrame], List[List[pd.DataFrame]]],
    n_jobs: int = 4,
    max_motifs: int = 5,
    discord_top_pct: float = 0.04,
    max_matches: int = 10,
    cluster:bool = False,
    motif:bool =False,
    min_mdl_ratio: float = 0.25,
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

    Returns
    -------
    dict or list
        Matrix profile information matching the structure of ``data``.
    """
        
    if data is None or (isinstance(data, list) and all(x is None for x in data)):
        return None

    if isinstance(data, pd.DataFrame):
        # Single DataFrame case
        df = data
        window_size = df.attrs["m"]

        return matrix_profile_process(
            df,
            window_size=window_size,
            max_motifs=max_motifs,
            discord_top_pct=discord_top_pct,
            max_matches=max_matches,
            cluster=cluster,
        )

    elif isinstance(data, list) and all(isinstance(x, pd.DataFrame) for x in data):
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
            )
            for df in pds
        ]

    else:
        raise TypeError("df must be a DataFrame, a list of DataFrames, or a list of lists of DataFrames")

"""Matrix profile computation helpers."""

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
    df_o: pd.DataFrame,
    window_size: Optional[int] = None,
    column: str | None = None,
    max_motifs: int = 3,
    discord_top_pct: float = 0.04,
    max_matches: int = 10,
) -> dict:
    """Return matrix profile related data for a single DataFrame."""

    # Copy to avoid mutating the original data
    df = df_o.copy()

    # Drop timestamp column if present, computation works on numeric values only
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    # Retrieve window size from attributes if not provided
    if window_size is None and "m" in df.attrs:
        window_size = df.attrs["m"]
    if window_size is None:
        raise RuntimeError("Critical error: window_size not provided")

    # Ensure all columns are numeric
    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
        raise ValueError(
            "All columns must be numeric for matrix profile computation."
        )

    # ==== UNIVARIATE ====
    if df.shape[1] == 1:
        # Delegate to motif discovery helper for one variable
        return discover_patterns_stumpy_mixed(
            df,
            window_size,
            max_motifs=max_motifs,
            discord_top_pct=discord_top_pct,
            max_matches=max_matches,
        )

    # ==== MULTIVARIATE ====
    return discover_patterns_mstump_mixed(
        df,
        window_size,
        max_motifs=max_motifs,
        discord_top_pct=discord_top_pct,
        max_matches=max_matches,
    )


def matrix_profile(
    df_o: Union[pd.DataFrame, List[pd.DataFrame]],
    window_size: Optional[int] = None,
    n_jobs: int = 4,
    column: str | None = None,
    max_motifs: int = 5,
    discord_top_pct: float = 0.04,
    max_matches: int = 10,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """Compute the matrix profile for one or several DataFrames.

    Parameters
    ----------
    df_o : DataFrame or list of DataFrame
        Time series data to process.
    window_size : int, optional
        Length of the subsequences. If ``None`` the value stored in
        ``df_o.attrs['m']`` is used when available.
    n_jobs : int
        Number of parallel workers when ``df_o`` is a list.
    column : str, optional
        Column name for univariate input.
    max_motifs : int
        Maximum number of motifs to detect.
    discord_top_pct : float
        Fraction of discords to return.
    max_matches : int
        Maximum matches per motif.

    Returns
    -------
    DataFrame or list of DataFrame
        The computed matrix profile(s).
    """
    if not (
        isinstance(df_o, pd.DataFrame) or
        (isinstance(
            df_o, list) and all(isinstance(x, pd.DataFrame) for x in df_o))
    ):
        raise TypeError("df must be a pd.DataFrame or a list of pd.DataFrame")
    if isinstance(df_o, list):
        result = []
        [result.append(matrix_profile_process(df, window_size=window_size)) for df in df_o]
        return result


    # Automatically pick the sole column for univariate series
    if column is None and len(df_o.columns) == 1:
        column = df_o.columns[0]

    # Use the preprocessing attribute if no window size is provided
    if window_size is None and "m" in df_o.attrs:
        window_size = df_o.attrs["m"]

    df_profile = matrix_profile_process(
        df_o,
        window_size=window_size,
        column=column,
        max_motifs=max_motifs,
        discord_top_pct=discord_top_pct,
        max_matches=max_matches,
    )

    return df_profile

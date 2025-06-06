"""Matrix profile computation helpers."""

from typing import List, Optional, Union

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import stumpy as sp

os.environ["NUMBA_NUM_THREADS"] = "8"

from .motif_pattern import discover_patterns_stumpy_mixed


def matrix_profile_process(
    df_o: pd.DataFrame,
    window_size: Optional[int] = None,
    column: str | None = None,
    max_motifs: int = 3,
    top_percent_discords: float = 0.02,
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
            top_percent_discords=top_percent_discords,
            max_matches=max_matches,
        )

    # ==== MULTIVARIATE ====
    # Compute multivariate matrix profile
    P, I = sp.mstump(df, m=window_size, normalize=False, discords=False)
    motif_distances, motif_indices, motif_subspaces, motif_mdls = sp.mmotifs(
        df,
        P,
        I,
        max_motifs=3,
        max_matches=5,
        normalize=False,
    )

    # Convert STUMPY outputs to DataFrames aligned with the original index
    profile_len = df.shape[0] - window_size + 1
    center_indices = np.arange(profile_len) + window_size // 2
    center_indices = center_indices[center_indices < len(df)]

    df_profile = pd.DataFrame(
        P.T,
        columns=[f"value_{col}" for col in df.columns],
    )
    df_profile.index = df.index[center_indices]

    df_index = pd.DataFrame(
        I.T,
        columns=[f"index_{col}" for col in df.columns],
    )
    df_index.index = df.index[center_indices]

    # Group all outputs in a single dictionary
    return {
        "profile": df_profile,
        "profile_index": df_index,
        "motif_distances": motif_distances,
        "motif_indices": motif_indices,
        "motif_subspaces": motif_subspaces,
        "motif_mdls": motif_mdls,
        "window_size": window_size,
    }




def matrix_profile(
    df_o: Union[pd.DataFrame, List[pd.DataFrame]],
    window_size: Optional[int] = None,
    n_jobs: int = 4,
    column: str | None = None,
    max_motifs: int = 3,
    top_percent_discords: float = 0.02,
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
    top_percent_discords : float
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
        # Cas multivarié, plusieurs DataFrames en batch
        # On utilise joblib pour paralléliser
        def mp_func(df):
            return matrix_profile_process(df, window_size=window_size)
        result = Parallel(n_jobs=n_jobs)(delayed(mp_func)(df) for df in df_o)
        # Option : si tous tes dfs ont plusieurs colonnes, détecte et passe par mstump
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
        top_percent_discords=top_percent_discords,
        max_matches=max_matches,
    )

    return df_profile

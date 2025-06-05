from typing import Optional, Union, List
from tslearn.metrics import dtw_path, dtw
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import stumpy as sp
import faiss
import os 
os.environ["NUMBA_NUM_THREADS"] = "8"
import matplotlib.pyplot as plt
from .motif_pipeline import discover_univariate_patterns, discover_multivariate_patterns
from .plotting import plot_matrix_profiles
from .motif_pattern import discover_patterns_stumpy_mixed


def matrix_profile_process(
        df_o: pd.DataFrame, window_size: Optional[int] = None, column=None, 
        max_motifs=3, top_percent_discords=0.02, margin_discord=0,
        max_matches=10):
    df = df_o.copy()
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    if window_size is None and 'm' in df.attrs:
        window_size = df.attrs['m']
    if window_size is None:
        raise RuntimeError("Critical error: window_size not provided")
    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
        raise ValueError("All columns must be numeric for matrix profile computation.")
    # ==== UNIVARIATE ====
    if df.shape[1] == 1:
        return discover_patterns_stumpy_mixed(
            df, window_size, max_motifs=max_motifs, 
            top_percent_discords=top_percent_discords, margin_discord=margin_discord,
            max_matches=max_matches)
    # ==== MULTIVARIATE ====
    else:
        P, I = sp.mstump(df, m=window_size, normalize=False, discords=False)
        motif_distances, motif_indices, motif_subspaces, motif_mdls = sp.mmotifs(
            df,
            P,
            I,
            max_motifs=3,
            max_matches=5,
            normalize=False
        )
        profile_len = df.shape[0] - window_size + 1
        center_indices = np.arange(profile_len) + window_size // 2
        center_indices = center_indices[center_indices < len(df)]

        df_profile = pd.DataFrame(P.T, columns=[f'value_{col}' for col in df.columns])
        df_profile.index = df.index[center_indices]

        df_index = pd.DataFrame(I.T, columns=[f'index_{col}' for col in df.columns])
        df_index.index = df.index[center_indices]

        # Tout rassembler dans un dict pour clarté
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
    column: str = None, 
    max_motifs=3,
    top_percent_discords=0.02,
    margin_discord=0,
    max_matches=10, 
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Compute the matrix profile for a DataFrame or a list of DataFrames
    using the specified window size.
    Uses joblib to parallelize the computation on lists for fast batch mode.

    Args:
        df_o (pd.DataFrame or list of pd.DataFrame): Time series to process.
        window_size (int, optional): Subsequence length.
        n_jobs (int): Number of parallel workers if list input.

    Returns:
        pd.DataFrame or list of pd.DataFrame: Matrix profiles.
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


    if column is None and len(df_o.columns) == 1:
        column = df_o.columns[0]
    if window_size is None and 'm' in df_o.attrs:
        window_size = df_o.attrs['m']
    df_profile = matrix_profile_process(df_o, window_size=window_size,
            column=column, max_motifs=max_motifs, 
            top_percent_discords=top_percent_discords,
            margin_discord=margin_discord,
            max_matches=max_matches)
    # result = discover_univariate_patterns(
    #     df_o, df_profile, window_size, column)
    return df_profile
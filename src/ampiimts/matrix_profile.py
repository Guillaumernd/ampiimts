from typing import Optional, Union, List
import numpy as np
import pandas as pd
import stumpy as sp
from joblib import Parallel, delayed


def multi_aamp_with_nan_parallel(
    df: pd.DataFrame,
    window_size: int,
    n_jobs: int = -1
):
    """
    Matrix Profile multivarié non normalisé (façon 'aamp'), NaN-friendly,
    joblib-parallélisé.
    Args:
        df (pd.DataFrame): Preprocessed time series (n_timepoints, n_features).
        window_size (int): Taille de la fenêtre temporelle.
        n_jobs (int): Nombre de jobs parallèles (-1 = all cores)
    Returns:
        pd.DataFrame: Matrix profile multivarié.
    """
    X = df.values
    n, k = X.shape
    profile_len = n - window_size + 1

    # 1. Profil aamp univarié pour chaque colonne, EN PARALLÈLE !
    def compute_aamp(col):
        aamp_result = sp.aamp(X[:, col], window_size)
        return aamp_result[:, 0], aamp_result[:, 1]

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_aamp)(col) for col in range(k)
    )
    uni_profiles, uni_indices = zip(*results)
    uni_profiles = np.stack(uni_profiles, axis=0).astype(np.float64)
    uni_indices = np.stack(uni_indices, axis=0).astype(np.int64)

    # 2. Agrégation euclidienne
    mx_profile = np.sqrt(np.sum(uni_profiles ** 2, axis=0))
    # Pour l'indice, tu peux prendre le min sur toutes colonnes
    # (indice du vrai match global)
    min_indices = np.argmin(uni_profiles, axis=0)
    mx_profile_idx = np.array([
        uni_indices[min_indices[i], i] for i in range(profile_len)
    ])

    # 4. DataFrame résultat (index centré sur la fenêtre)
    center_indices = np.arange(profile_len) + window_size // 2
    center_indices = center_indices[center_indices < n]
    df_profile = pd.DataFrame({
        "value": mx_profile[:len(center_indices)],
        "index_1": np.arange(len(center_indices)),
        "index_2": mx_profile_idx[:len(center_indices)],
    }, index=df.index[center_indices])
    return df_profile


def matrix_profile_process(
        df_o: pd.DataFrame, window_size: Optional[int] = None) -> pd.DataFrame:
    """
    Compute the matrix profile for a (multi-)column time series DataFrame
    using the specified window size (unnormalized, supports NaN).
    Uses stumpy.aamp for univariate, stumpy.maamp for multivariate.
    The result index is centered within the window.
    """
    df = df_o.copy()
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    if window_size is None and 'm' in df.attrs:
        window_size = df.attrs['m']
    if window_size is None:
        raise RuntimeError("Critical error: window_size not provided")
    # Ensure all columns are numeric
    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
        raise ValueError(
            "All columns must be numeric for matrix profile computation."
        )
    # Compute matrix profile using stumpy
    if df.shape[1] == 1:
        mx_profile = sp.aamp(df.values.ravel(), window_size)
    else:
        mx_profile = multi_aamp_with_nan_parallel(
            df, window_size=window_size, n_jobs=-1)
    df_profile = pd.DataFrame(
        mx_profile, columns=['value', 'index_1', 'index_2', 'index_3']
    )
    profile_len = len(df_o) - window_size + 1
    # Center timestamp index on matrix profile
    center_indices = np.arange(profile_len) + window_size // 2
    center_indices = center_indices[center_indices < len(df_o)]
    df_profile = df_profile.iloc[:len(center_indices)]
    df_profile.index = df_o.index[center_indices]
    return df_profile


def matrix_profile(
    df_o: Union[pd.DataFrame, List[pd.DataFrame]],
    window_size: Optional[int] = None,
    n_jobs: int = -1
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
        # Use joblib for parallel processing
        matrix_profiles = Parallel(n_jobs=n_jobs)(
            delayed(matrix_profile_process)(df, window_size=window_size)
            for df in df_o
        )
        return matrix_profiles
    return matrix_profile_process(df_o, window_size=window_size)

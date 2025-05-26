import numpy as np
import pandas as pd
import stumpy as sp
from typing import Optional

def matrix_profile(df_o: pd.DataFrame, window_size: Optional[int] = None):
    """
    Compute the matrix profile for a (multi-)column time series DataFrame using the specified window size.
    Uses aamp for univariate (no normalization) and maamp for multivariate (no normalization).
    Handles NaN values natively as supported by stumpy.

    Args:
        df_o (pd.DataFrame): Input DataFrame with time series columns (must be numeric).
        window_size (int, optional): Subsequence/window length. If None, tries to use df.attrs['m'].

    Returns:
        np.ndarray: Matrix profile array computed by stumpy.

    Raises:
        RuntimeError: If window_size is not provided and not found in df.attrs.
        ValueError: If columns are not all numeric.
    """
    df = df_o.copy()  
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
        
    # Use window_size from DataFrame attributes if not explicitly provided
    if window_size is None and 'm' in df.attrs:
        window_size = df.attrs['m'] // 2
    # Raise error if window_size is still not set
    if window_size is None:
        raise RuntimeError("Critical error: window_size not provided")

    # Ensure all columns are numeric
    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
        raise ValueError("All columns must be numeric for matrix profile computation.")

    # Do NOT check for NaNs: stumpy functions handle NaN natively

    # Compute matrix profile: use aamp for 1D, maamp for multi-D (both unnormalized)
    if df.shape[1] == 1:
        # aamp expects 1D input for univariate
        matrix_profile = sp.aamp(df.values.ravel(), window_size)
    else:
        # maamp handles multivariate case (no normalization)
        matrix_profile = sp.maamp(df.values, window_size)
    df_profile = pd.DataFrame(matrix_profile, columns=['value', 'index_1', 'index_2', 'index_3'])
    profile_len = len(df_o) - window_size + 1

    # Center timestamp index on matrix profile
    center_indices = np.arange(profile_len) + window_size
    # Vérification pour ne pas dépasser les bornes
    center_indices = center_indices[center_indices < len(df_o)]
    df_profile = df_profile.iloc[:len(center_indices)]
    df_profile.index = df_o.index[center_indices]

    return df_profile

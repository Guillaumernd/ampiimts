"""Preprocessed module for panda DataFrame with timestamp column."""
from typing import Union, List
from collections import Counter
from joblib import Parallel, delayed
import warnings
import numpy as np
import pandas as pd
from numba import njit
import faiss
import time
import random

def synchronize_on_common_grid(
    dfs: List[pd.DataFrame],
    gap_multiplier: float = 15,
    sort_by_variables: bool = True,
) -> List[pd.DataFrame]:
    """
    Synchronize multiple time series DataFrames on a common regular time grid.

    Each input DataFrame is first interpolated independently
    (using the provided `interpolate_func`, which should handle small gaps
    and leave large gaps as NaN).
    Then, all DataFrames are reindexed and time-interpolated on a shared
    regular time axis computed from the median sampling interval across
    all DataFrames.
    This ensures perfect timestamp alignment for all DataFrames while
    preserving large gaps as NaN.

    Warnings are issued if the sampling intervals of the DataFrames differ
    significantly.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        List of DataFrames to synchronize. Each DataFrame must have a
        DatetimeIndex.
    interpolate_func : callable
        Function to interpolate a single DataFrame. Must accept
        (df, gap_multiplier) and return a DataFrame with gaps properly
        handled (small gaps interpolated, large gaps as NaN).
    gap_multiplier : float, default=15
        Maximum gap (as a multiple of the base frequency) to interpolate
        when calling `interpolate_func`. Larger gaps will remain as NaN.

    Returns
    -------
    dfs_synced : list of pd.DataFrame
        List of synchronized DataFrames, each reindexed and interpolated
        on the same regular time grid (with identical DatetimeIndex),
        ready for further analysis.

    Notes
    -----
    - Large gaps (as defined by `gap_multiplier`) are preserved as NaN after
      synchronization.
    - Small discrepancies in start/end times or slight clock drifts are
      handled by interpolation on the common grid.
    - A warning is raised if the input DataFrames have significantly different
      time intervals.

    Example
    -------
    >>> synced = synchronize_on_common_grid
        (dfs, interpolate_func=my_interpolate)
    >>> # Now, synced[0].index == synced[1].index == ... for all dataframes
    """
    dfs = [
            d.drop([c for c in ['latitude', 'longitude'] if c in d.columns], axis=1)
            for d in dfs.copy()
        ]
    # Independent interpolation
    dfs = [interpolate(df, gap_multiplier=gap_multiplier) for df in dfs]
    # Sampling frequency for each DataFrame
    freqs = [df.index.to_series().diff().median() for df in dfs]
    freqs_seconds = [f.total_seconds() for f in freqs]
    median_freq = np.median(freqs_seconds)
    max_diff = max(abs(f - median_freq) for f in freqs_seconds)
    if max_diff > 1.2 * median_freq:
        warnings.warn(
            f"[SYNC WARNING] Not all DataFrames have the same time step: "
            f"Found steps (in seconds): {freqs_seconds} "
            f"(median: {median_freq}s)"
        )
    common_freq = pd.to_timedelta(median_freq, unit="s")
    min_time = min(df.index.min() for df in dfs)
    max_time = max(df.index.max() for df in dfs)
    reference_index = pd.date_range(
        start=min_time, end=max_time, freq=common_freq)
    # Reindex and interpolate on the same timestamp for all DataFrames
    dfs_synced = [
        df.reindex(
            reference_index).interpolate(
                method="time", limit_direction="both")
        for df in dfs
    ]
    
    # Ensure same numeric columns
    numeric_cols = [sorted(d.select_dtypes(include=[np.number]).columns.tolist())
                    for d in dfs_synced]
    ref_cols = numeric_cols[0]
    for idx, cols in enumerate(numeric_cols[1:], 1):
        if cols != ref_cols:
            raise ValueError(
                f"DataFrame {idx} does not have the same numerical columns as DataFrame 0:\n"
                f"Reference: {ref_cols}\nCurrent: {cols}"
            )
    # split per variable if requested
    if sort_by_variables:
        # build per-variable DataFrames
        dataframes_by_var = []
        for col_name in ref_cols:
            var_frames = []
            for i, d in enumerate(dfs_synced):
                renamed = d[[col_name]].rename(columns={col_name: f"{col_name}_{i}"})
                var_frames.append(renamed)
            df_var = pd.concat(var_frames, axis=1)
            df_var.index = dfs_synced[0].index
            dataframes_by_var.append(df_var)
        dfs_synced = dataframes_by_var
    return dfs_synced

def interpolate(
    df: pd.DataFrame,
    gap_multiplier: float = 15,
    column_thresholds: dict = None  # facultatif : {'T': 5.0, 'RH': 3.5, ...}
) -> pd.DataFrame:
    df = df.copy()

    # --- Nettoyage de l'index temporel ---
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df.index = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.drop(columns=["timestamp"])
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")
    df.index = df.index.tz_localize(None)
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # --- Estimation automatique des seuils par colonne si non fourni ---
    def estimate_column_thresholds(data: pd.DataFrame) -> dict:
        thresholds = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            top_mean = data[col].nlargest(int(len(data) * 0.15)).mean()
            bot_mean = data[col].nsmallest(int(len(data) * 0.15)).mean()
            thresholds[col] = (top_mean - bot_mean) * 3.0
        return thresholds

    if column_thresholds is None:
        column_thresholds = estimate_column_thresholds(df)

    # --- Détection des outliers extrêmes (pics + plateaux aberrants) ---
    outlier_timestamps = set()
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col]
        threshold = column_thresholds.get(col, np.inf)

        # Marquer tous les points trop élevés ou trop bas comme outliers
        mask = (series > threshold) | (series < -threshold)
        outlier_timestamps.update(df.index[mask])

    df_wo_outliers = df.drop(index=outlier_timestamps)

    # --- Fréquence ---
    idx = df_wo_outliers.index.unique().sort_values()
    if len(idx) < 2:
        raise ValueError("Not enough points to estimate frequency.")

    inferred = pd.infer_freq(idx)
    if inferred:
        if len(inferred) == 1 or (len(inferred) == 2 and inferred[0] == "W"):
            inferred = "1" + inferred
        try:
            if inferred.isalpha():
                inferred = "1" + inferred
            freq = pd.to_timedelta(inferred)
        except ValueError:
            freq = idx.to_series().diff().median()
    else:
        freq = idx.to_series().diff().median()

    max_gap = freq * gap_multiplier

    # Rééchantillonnage
    full_idx = pd.date_range(start=idx.min(), end=idx.max(), freq=freq)
    union_idx = idx.union(full_idx)
    df_union = df_wo_outliers.reindex(union_idx)
    df_union = df_union.infer_objects(copy=False)

    diffs = idx.to_series().diff()
    seg_ids = (diffs > max_gap).cumsum()
    seg_index = pd.Series(seg_ids.values, index=idx)
    seg_union = seg_index.reindex(union_idx, method="ffill").fillna(0).astype(int)

    df_interp = df_union.groupby(seg_union, group_keys=False).apply(
        lambda seg: seg.interpolate(method="time", limit_direction="both")
    )

    df_out = df_interp.reindex(full_idx)
    df_out.index.name = "timestamp"

    # Grandes coupures = NaN
    gap_mask = pd.Series(False, index=df_out.index)
    for prev, nxt in zip(idx[:-1], idx[1:]):
        if nxt - prev > max_gap:
            in_gap = (df_out.index > prev) & (df_out.index <= nxt)
            gap_mask |= in_gap
    df_out.loc[gap_mask, :] = np.nan

    # Réinjecter les timestamps outliers en tant que NaN
    outlier_idx = sorted(ts for ts in outlier_timestamps if ts in df_out.index)
    df_out.loc[outlier_idx] = np.nan

    return df_out




@njit  # comment njit for coverage
def _compute_aswn(
    values: np.ndarray,
    window_size: int,
    min_std: float,
    min_valid_ratio: float
) -> np.ndarray:
    """
    Fast local window normalization (ASWN) with Numba.

    Args:
        values (np.ndarray): Input 1D array.
        window_size (int): Sliding window size.
        min_std (float): Minimum allowed standard deviation.
        min_valid_ratio (float): Min ratio of valid (non-NaN) values.

    Returns:
        np.ndarray: Normalized array, same shape as input.
    """
    n = len(values)
    half = window_size // 2
    result = np.empty(n)
    result[:] = np.nan
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        count_valid = 0
        sum_valid = 0.0
        sumsq_valid = 0.0
        for j in range(start, end):
            v = values[j]
            if not np.isnan(v):
                count_valid += 1
                sum_valid += v
                sumsq_valid += v * v
        total = end - start
        if count_valid / total >= min_valid_ratio and count_valid > 1:
            mean = sum_valid / count_valid
            var = (sumsq_valid / count_valid) - (mean * mean)
            std = np.sqrt(var)
            if std < min_std:
                std = min_std
            v = values[i]
            result[i] = (v - mean) / std if not np.isnan(v) else np.nan
        else:
            result[i] = np.nan
    return result


def aswn_with_trend(
    series: pd.Series,
    window_size: int = 50,
    min_std: float = 1e-2,
    min_valid_ratio: float = 0.8,
    alpha: float = 0.65,
) -> pd.Series:
    """
    ASWN normalization with optional trend blending.

    Args:
        series (pd.Series): Input time series.
        window_size (int): Sliding window size.
        min_std (float): Minimum allowed standard deviation.
        min_valid_ratio (float): Min ratio of valid (non-NaN) values.
        alpha (float): Trend blending coefficient (0-1).

    Returns:
        pd.Series: Normalized series (same index).
    """
    values = series.values.astype(np.float64)
    normed = _compute_aswn(values, window_size, min_std, min_valid_ratio)
    result = pd.Series(normed, index=series.index)
    if alpha > 0:
        # print(window_size)
        trend = series.rolling(
            window=window_size * 10, center=True, min_periods=1
        ).mean()
        return (1 - alpha) * result + alpha * (series - trend)
    return result


def normalization(
    df: pd.DataFrame,
    min_std: float = 1e-2,
    min_valid_ratio: float = 0.8,
    alpha: float = 0.65,
    window_size: str = None,
) -> pd.DataFrame:
    """
    Applies ASWN normalization (with trend) to all
    numeric columns in a DataFrame.
    The window size is automatically computed from the median time delta.

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.
        min_std (float): Minimum allowed standard deviation.
        min_valid_ratio (float): Min ratio of valid (non-NaN) values.
        alpha (float): Trend blending coefficient (0-1).

    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    if window_size is None:
        window_size = 50
    elif isinstance(window_size, str):
        window_size = int(
            pd.Timedelta(window_size) / df.index.to_series().diff().median()
        )
    elif isinstance(window_size, (int, np.integer)):
        pass
    else:
        raise RuntimeError(
            "Window size must be None, str or int")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = aswn_with_trend(
                df[col], window_size, min_std, min_valid_ratio, alpha
            )
    df.attrs["m"] = window_size
    return df


def define_m_using_clustering(
    df: pd.DataFrame,
    k: int = 3,
    window_sizes: list[str] = None,
    max_points: int = 5000,
    max_window_sizes: int = 50,
    max_segments: int = 2000,
    n_jobs: int = 8,
) -> list[tuple[int, str, float, float]]:
    """
    Determines the best window sizes for motif extraction in a time series
    using clustering. Handles empty or oversized cases gracefully.

    Returns a list of (ws_pts, ws_str, stability_score, density_score).
    """
    # Estimate base sampling frequency
    freq = df.index.to_series().diff().median()
    if pd.isna(freq) or freq <= pd.Timedelta(0):
        freq = pd.Timedelta(seconds=1)

    # Generate candidate window sizes
    if window_sizes is None:
        max_pts = max(len(df) // 4, 2)
        n_candidates = min(max_window_sizes * 2, max_pts)
        pts_candidates = np.unique(
            np.round(
                np.logspace(np.log10(2), np.log10(max_pts), num=n_candidates)
            ).astype(int)
        )
        window_sizes = [str(p * freq) for p in pts_candidates]

    # Convert to point counts and filter
    window_sizes_pts = []
    for ws in window_sizes:
        delta = pd.Timedelta(ws)
        pts = int(delta / freq) if delta >= freq else 0
        max_len = max(len(df) // 4, 2)
        if 10 <= pts < max_len:
            window_sizes_pts.append((pts, ws))

    if not window_sizes_pts:
        default_pts = max(2, len(df) // 2)
        ws_delta = default_pts * freq
        ws_str = pd.tseries.frequencies.to_offset(ws_delta).freqstr
        window_sizes_pts.append((default_pts, ws_str))

    # Subsample df if too long
    if len(df) > max_points:
        idx = np.linspace(0, len(df)-1, max_points, dtype=int)
        df = df.iloc[idx]

    # Define evaluation of one window size
    def eval_window(values: np.ndarray, ws_pts: int, ws_str: str):
        try:
            segs = np.lib.stride_tricks.sliding_window_view(values, ws_pts)
        except ValueError:
            return None
        n = segs.shape[0]
        if n < 2:
            return None
        if n > max_segments:
            idx = np.random.choice(n, max_segments, replace=False)
            segs = segs[idx]
            n = max_segments
        # Remove windows containing NaNs
        mask = ~np.isnan(segs).any(axis=1)
        segs = segs[mask]
        n = segs.shape[0]
        if n < 2:
            return None
        # Normalize
        segs = (segs - segs.mean(axis=1, keepdims=True)) / (segs.std(axis=1, keepdims=True) + 1e-8)
        # FAISS nearest neighbor distances
        index = faiss.IndexFlatL2(ws_pts)
        segs_f = segs.astype(np.float32)
        index.add(segs_f)
        dists, _ = index.search(segs_f, 2)
        nn = dists[:, 1]
        stability = np.median(nn) / ws_pts
        density = (nn < stability * 1.5).sum() / n
        return (ws_pts, ws_str, stability, density)

    # Evaluate all numeric columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:3]
    if not num_cols:
        raise ValueError("No numeric columns available for clustering.")

    # Compute metrics for each window size
    results_by_ws = {ws_str: {"stability": [], "density": [], "pts": pts}
                     for pts, ws_str in window_sizes_pts}
    for col in num_cols:
        vals = df[col].values
        res = Parallel(n_jobs=n_jobs)(
            delayed(eval_window)(vals, pts, ws_str)
            for pts, ws_str in window_sizes_pts
        )
        for r in res:
            if r is None:
                continue
            pts, ws_str, stab, dens = r
            results_by_ws[ws_str]["stability"].append(stab)
            results_by_ws[ws_str]["density"].append(dens)

    # Aggregate metrics
    aggregated = []
    for ws_str, metrics in results_by_ws.items():
        if not metrics["stability"]:
            continue
        med_stab = np.median(metrics["stability"])
        mean_dens = np.mean(metrics["density"])
        aggregated.append((metrics["pts"], ws_str, med_stab, mean_dens))

    if not aggregated:
        raise ValueError("No window size evaluated successfully.")

    # Determine best window for inclusion
    # Score = density / stability
    scores = {ws_str: np.mean(m["density"]) / np.median(m["stability"])
              for ws_str, m in results_by_ws.items() if m["stability"]}
    best_ws = max(
        window_sizes_pts,
        key=lambda tpl: scores.get(tpl[1], -np.inf)
    )

    # Random sample others but always include best_ws
    if len(window_sizes_pts) > max_window_sizes:
        candidates = [ws for ws in window_sizes_pts if ws != best_ws]
        sampled = random.sample(candidates, max_window_sizes - 1)
        window_sizes_pts = sampled + [best_ws]
        random.shuffle(window_sizes_pts)

    # Final sort or limit to top k by stability
    aggregated.sort(key=lambda x: x[2])
    final = aggregated[:k]
    return final

def pre_processed(
    df: Union[pd.DataFrame, List[pd.DataFrame]],
    gap_multiplier: float = 15,
    min_std: float = 1e-2,
    min_valid_ratio: float = 0.8,
    alpha: float = 0.65,
    window_size: str = None,
    sort_by_variables: bool = True,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """Full preprocessing pipeline for one or several DataFrames.

    Parameters
    ----------
    df : DataFrame or list of DataFrame
        Input data to preprocess.
    gap_multiplier : float, optional
        Maximum gap in multiples of the base frequency considered for
        interpolation.
    min_std : float, optional
        Minimum standard deviation used during normalization.
    min_valid_ratio : float, optional
        Minimum ratio of valid samples in the sliding window.
    alpha : float, optional
        Trend blending coefficient for ASWN normalization.
    window_size : str, optional
        Window size as pandas offset string (e.g. ``"1h"``). If ``None`` it is
        estimated.
    sort_by_variables : bool, optional
        Whether to preprocess each variable independently when ``df`` is a list.

    Returns
    -------
    DataFrame or list of DataFrame
        Preprocessed data ready for motif discovery.
    """

    window_size_forward = None
    # Type check
    if not (
        isinstance(df, pd.DataFrame)
        or (isinstance(df, list) and all(isinstance(x, pd.DataFrame) for x in df))
    ):
        raise TypeError("df must be a pd.DataFrame or a list of pd.DataFrame")

    # If list of DataFrames
    if isinstance(df, list):
        # Drop lat/lon and sync on common grid
        dataframes = synchronize_on_common_grid(df, sort_by_variables=sort_by_variables)

        # estimate or override window_size
        if window_size is None:
            wins = define_m_using_clustering(dataframes[0])
            window_size_forward = wins[0][1]
            print(f"[WINDOW LIST] Fenêtre retenue (variables séparées) → {window_size_forward}")
        else:
            window_size_forward = window_size
            print(f"[WINDOW LIST] Fenêtre utilisateur → {window_size_forward}")

        return [
            normalization(
                df_v,
                min_std=min_std,
                min_valid_ratio=min_valid_ratio,
                alpha=alpha,
                window_size=window_size_forward
            )
            for df_v in dataframes
        ]

    # Single DataFrame path
    df_preprocessed = df.copy()
    # Step 1: interpolation
    df_preprocessed = interpolate(df_preprocessed, gap_multiplier)

    # Step 2: estimate or override window size
    if window_size is None:
        wins = define_m_using_clustering(df_preprocessed)
        window_size_forward = wins[0][1]
        print(f"[WINDOW] Fenêtre retenue (meilleure) → {window_size_forward}")
    else:
        window_size_forward = window_size
        print(f"[WINDOW] Fenêtre utilisateur → {window_size_forward}")

    # Step 3: normalization
    df_preprocessed = normalization(
        df_preprocessed,
        min_std=min_std,
        min_valid_ratio=min_valid_ratio,
        alpha=alpha,
        window_size=window_size_forward
    )

    return df_preprocessed
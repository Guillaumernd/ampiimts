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
    gap_multiplier: float = 15
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
    return dfs_synced


def interpolate(df: pd.DataFrame, gap_multiplier: float = 15) -> pd.DataFrame:
    """
    Interpolates a multivariate, irregular time series DataFrame onto a
    regular time grid.
    - Infers base frequency.
    - Only fills small gaps (less than gap_multiplier * base frequency).
    - Leaves large gaps as NaN.

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.
        gap_multiplier (float): Maximum gap (in multiples of base freq) to
        interpolate.

    Returns:
        pd.DataFrame: DataFrame resampled on a regular grid, with only small
        gaps interpolated.
    """
    df = df.copy()
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

    idx = df.index.unique().sort_values()
    if len(idx) < 2:
        raise ValueError("Not enough points to estimate frequency.")
    inferred = pd.infer_freq(idx)
    if inferred:
        # If inferred is a single character
        # like "T", "S", "H", add "1" in front
        if len(inferred) == 1 or (len(inferred) == 2 and inferred[0] == "W"):
            inferred = "1" + inferred
        try:
            freq = pd.to_timedelta(inferred)
        except ValueError:
            # fallback to median diff in case of weird infer_freq
            freq = idx.to_series().diff().median()
    else:
        freq = idx.to_series().diff().median()

    max_gap = freq * gap_multiplier

    full_idx = pd.date_range(start=idx.min(), end=idx.max(), freq=freq)
    union_idx = idx.union(full_idx)
    df_union = df.reindex(union_idx)
    if "timestamp" in df_union.columns:
        df_union = df_union.drop(columns=["timestamp"])
    df_union = df_union.infer_objects(copy=False)
    diffs = idx.to_series().diff()
    seg_ids = (diffs > max_gap).cumsum()
    seg_index = pd.Series(seg_ids.values, index=idx)
    seg_union = seg_index.reindex(
        union_idx, method="ffill").fillna(0).astype(int)
    seg_union = seg_union.infer_objects(copy=False)
    df_interp = df_union.groupby(seg_union, group_keys=False).apply(
        lambda seg: seg.interpolate(method="time", limit_direction="both")
    )

    df_out = df_interp.reindex(full_idx)
    df_out.index.name = "timestamp"

    gap_mask = pd.Series(False, index=df_out.index)
    for prev, nxt in zip(idx[:-1], idx[1:]):
        if nxt - prev > max_gap:
            # Mark all points between ``prev`` (exclusive) and ``nxt`` (inclusive)
            # as belonging to a gap
            in_gap = (df_out.index > prev) & (df_out.index <= nxt)
            gap_mask |= in_gap

    # Insert NaN wherever a gap has been detected
    df_out.loc[gap_mask, :] = np.nan

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
    if isinstance(window_size, str):
        window_size = int(
            pd.Timedelta(window_size) / df.index.to_series().diff().median()
        )
    else:
        raise RuntimeError(
            "Window size isn't a str (e.g. : 1D, 1h, 1S etc...)")
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
    max_points: int = 4000,
    max_window_sizes: int = 12,
    max_segments: int = 2000,
    n_jobs: int = -1
) -> list[tuple[int, str, float, float]]:
    """
    Determines the best window sizes for motif extraction in a time series
    using clustering. Handles empty or oversized cases gracefully.

    Args:
        df: DataFrame with DatetimeIndex and numeric columns.
        k: Number of top window sizes per variable.
        window_sizes: List of pandas offsets (e.g. ['1m','1h']).
        max_points: Max total points to subsample for speed.
        max_window_sizes: Max distinct window sizes to test.
        max_segments: Max sliding-window segments to index per WS.
        n_jobs: Cores for parallel.

    Returns:
        List of tuples: (ws_pts, ws_str, stability_score, density_score).
    """
    # Default window sizes
    if window_sizes is None:
        window_sizes = [
            "1s","15s","30s","1m","5m","10m","15m","30m",
            "1h","2h", "3h","4h","5h", "6h", "7h","8h","10h","12h", "14h", "16h", "18h", "20h", "22h", "24h", "48h"]

    # Median sample interval
    freq = df.index.to_series().diff().median()

    # Convert and filter window sizes
    window_sizes_pts = []
    for ws in window_sizes:
        delta = pd.Timedelta(ws)
        pts = int(delta / freq) if delta >= freq else 0
        if 10 <= pts < len(df) // 4:
            window_sizes_pts.append((pts, ws))

    if not window_sizes_pts:
        raise ValueError("No valid window sizes: adjust your `window_sizes` or data length.")

    # Subsample rows if too long
    if len(df) > max_points:
        idx = np.linspace(0, len(df)-1, max_points, dtype=int)
        df = df.iloc[idx]

    # Limit number of window sizes
    if len(window_sizes_pts) > max_window_sizes:
        window_sizes_pts = random.sample(window_sizes_pts, max_window_sizes)

    def eval_window(values: np.ndarray, ws_pts: int, ws_str: str):
        # Build segments
        try:
            segs = np.lib.stride_tricks.sliding_window_view(values, ws_pts)
        except ValueError:
            return None
        n = segs.shape[0]
        if n < 2:
            return None
        # If too many segments, random sample
        if n > max_segments:
            idx = np.random.choice(n, max_segments, replace=False)
            segs = segs[idx]
            n = max_segments
        # Normalize
        segs = (segs - segs.mean(axis=1, keepdims=True)) / (segs.std(axis=1, keepdims=True) + 1e-8)
        # FAISS index
        index = faiss.IndexFlatL2(ws_pts)
        segs_f = segs.astype(np.float32)
        index.add(segs_f)
        dists, _ = index.search(segs_f, 2)
        nn = dists[:, 1]
        stability = np.median(nn) / ws_pts
        dens = (nn < stability * 1.5).sum() / n
        return (ws_pts, ws_str, stability, dens)

    # Evaluate per numeric column
    results_per_col = []
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:3]
    for col in num_cols:
        vals = df[col].values
        res = Parallel(n_jobs=n_jobs)(
            delayed(eval_window)(vals, pts, s) for pts, s in window_sizes_pts
        )
        res = [r for r in res if r is not None]
        if not res:
            continue
        results_per_col.append(sorted(res, key=lambda x: x[2])[:k])

    # Flatten
    all_tups = [t for sub in results_per_col for t in sub]
    if not all_tups:
        raise ValueError("No window size evaluated successfully.")

    # Consensus if >1 column
    if len(results_per_col) > 1:
        counts = Counter([t[1] for t in all_tups])
        most = counts.most_common(1)[0][1]
        cons_ws = [ws for ws,c in counts.items() if c == most]
        cand = [t for t in all_tups if t[1] in cons_ws]
        best = min(cand, key=lambda x: x[2])
        final = [best]
    else:
        final = all_tups

    print("Best consensus window size(s):", final)
    return final


def pre_processed(
    df: Union[pd.DataFrame, List[pd.DataFrame]],
    gap_multiplier: float = 15,
    min_std: float = 1e-2,
    min_valid_ratio: float = 0.8,
    alpha: float = 0.65,
    window_size: str = None,
    sort_by_variables: bool = True,
) -> pd.DataFrame:
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
    if not (
        isinstance(df, pd.DataFrame)
        or (isinstance(df, list)
            and all(isinstance(x, pd.DataFrame) for x in df))
            ):
        raise TypeError("df must be a pd.DataFrame or a list of pd.DataFrame")

    if isinstance(df, list):
        dataframes = df.copy()
        dataframes = [
            df.drop(
                [c for c in ['latitude', 'longitude']
                 if c in df.columns], axis=1
            )
            for df in dataframes]
        dataframes = synchronize_on_common_grid(dataframes)
        numeric_cols_lists_by_df = [sorted(
            df.select_dtypes(
                include=[np.number]).columns.tolist()) for df in dataframes]
        ref_name_dataframe = numeric_cols_lists_by_df[0]
        for idx, cols in enumerate(numeric_cols_lists_by_df[1:], 1):
            if cols != ref_name_dataframe:
                raise ValueError(
                    f"DataFrame {idx} does not have the same numerical "
                    "columns as DataFrame 0:\n"
                    f"Reference: {ref_name_dataframe}\nCurrent: {cols}"
                )
                
        if sort_by_variables:
            # Process each variable independently
            dataframes_by_variable = []
            for col_name in ref_name_dataframe:
                cols_for_this_var = []
                for i, df_ in enumerate(dataframes):
                    col_name_numerate = f"{col_name}_{i}"
                    df_col = df_[[col_name]].copy().rename(
                        columns={col_name: col_name_numerate})
                    cols_for_this_var.append(df_col)
                df_by_variable = pd.concat(cols_for_this_var, axis=1)
                df_by_variable.index = dataframes[0].index
                dataframes_by_variable.append(df_by_variable)
            dataframes = dataframes_by_variable

            dataframes_preprocessed = []
            for df_preprocessed in dataframes:
                if window_size is None:
                    # Compute the window size individually for each DataFrame
                    window_size_forward = define_m_using_clustering(df_preprocessed)
                    window_size_forward = window_size_forward[0][1]
                else:
                    window_size_forward = window_size

                df_preprocessed = normalization(
                    df_preprocessed,
                    min_std=min_std,
                    min_valid_ratio=min_valid_ratio,
                    alpha=alpha,
                    window_size=window_size_forward,
                )
                dataframes_preprocessed.append(df_preprocessed)
            return dataframes_preprocessed

        else:
            # sort_by_variables == False -> same window size for all DataFrames
            if window_size is None:
                ms = []
                df_ref = dataframes[0]
                for _ in range(3):
                    wins = define_m_using_clustering(df_ref)
                    if isinstance(wins, list):
                        ms.append(wins[0][1])
                    else:
                        ms.append(wins)
                window_size_forward = Counter(ms).most_common(1)[0][0]
                print(f"Most frequent window size after 3 runs (all dfs): {window_size_forward}")
            else:
                window_size_forward = window_size

            dataframes_preprocessed = []
            for df_preprocessed in dataframes:
                df_preprocessed = normalization(
                    df_preprocessed,
                    min_std=min_std,
                    min_valid_ratio=min_valid_ratio,
                    alpha=alpha,
                    window_size=window_size_forward,
                )
                dataframes_preprocessed.append(df_preprocessed)
            return dataframes_preprocessed
    df_preprocessed = df.copy()
    # Step 1: Interpolate small gaps on a regular grid, leave large gaps as NaN
    df_preprocessed = interpolate(df_preprocessed, gap_multiplier)
    if window_size is None:
        window_size_forward = define_m_using_clustering(df_preprocessed)
        window_size_str = [win[1] for win in window_size_forward]
        print("Best window sizes (hours):", ", ".join(window_size_str))
        window_size_forward = window_size_forward[0][1]
    # Step 2: Apply local normalization to all numeric columns
    # (ASWN, with trend blending if alpha>0)
    df_preprocessed = normalization(
        df_preprocessed,
        min_std=min_std,
        min_valid_ratio=min_valid_ratio,
        alpha=alpha,
        window_size=window_size_forward or window_size,
    )

    # Return the processed DataFrame
    return df_preprocessed


def missing_values(
        df: pd.DataFrame, percent_missing: float = 0.2, random_state: int = 1
) -> pd.DataFrame:
    """
    Simulate random missing data in all columns, including timestamp, in a
    vectorized way.

    Detects or creates a 'timestamp' column from the index or any datetime
    column.
    Applies a single boolean mask to assign NaN/NaT simultaneously across all
    columns.
    Sets 'timestamp' as the DataFrame index (as a DatetimeIndex), removing
    duplicate columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame; index may be DatetimeIndex or default.
    percent_missing : float, optional
        Fraction of rows to mark as missing (0 < percent_missing < 1). Default
        is 0.4.
    random_state : int, optional
        Seed for reproducibility. Default is 1.

    Returns
    -------
    pd.DataFrame
        Copy of input with missing values inserted, indexed by timestamp.
    """
    # 1. Input checks
    if not 0 < percent_missing < 1:
        raise ValueError("percent_missing must be between 0 and 1.")

    df_copy = df.copy()

    # 2. Ensure a 'timestamp' column of datetime type exists
    if isinstance(df_copy.index, pd.DatetimeIndex):
        name = df_copy.index.name or "timestamp"
        df_copy = df_copy.reset_index().rename(columns={name: "timestamp"})
    elif ("timestamp" in df_copy.columns
          and pd.api.types.is_datetime64_any_dtype(df_copy["timestamp"])
          ):
        pass
    else:
        dt_cols = [
            c
            for c in df_copy.columns
            if pd.api.types.is_datetime64_any_dtype(df_copy[c])
        ]
        if dt_cols:
            df_copy = df_copy.rename(columns={dt_cols[0]: "timestamp"})
        else:
            df_copy = df_copy.reset_index().rename(
                columns={"index": "timestamp"})

    # 3. Convert index to datetime (for safety)
    df_copy.index = pd.to_datetime(df_copy.index, errors="coerce")

    # 4. Prepare random mask for missingness
    rng = np.random.default_rng(random_state)
    mask = rng.random(df_copy.shape) < percent_missing

    # 5. Apply mask: pandas will choose NaN or NaT depending on dtype
    df_missing = df_copy.mask(mask)

    # 6. Remove duplicate columns, keeping the last 'timestamp'
    df_missing = df_missing.loc[:, ~df_missing.columns.duplicated(keep="last")]

    # 7. Set 'timestamp' as index and convert to DatetimeIndex
    df_missing = df_missing.set_index("timestamp")
    df_missing.index = pd.to_datetime(df_missing.index, errors="coerce")

    return df_missing


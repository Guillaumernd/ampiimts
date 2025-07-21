"""Preprocessed module for panda DataFrame with timestamp column."""
from typing import Union, List
from collections import defaultdict
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import pandas as pd
from numba import njit
import faiss
import time
import re
import random
import stumpy


def synchronize_on_common_grid(
    dfs_original: List[pd.DataFrame],
    gap_multiplier: float = 15,
    propagate_nan: bool = True,
    display_info: bool = False,
) -> pd.DataFrame:
    """
    Align multiple time series on a common timestamp grid for uniform processing.

    This function:
    - Interpolates each DataFrame individually.
    - Filters series based on compatible sampling frequencies.
    - Aligns time ranges if overlapping or generates a synthetic index otherwise.
    - Resamples all series to a common time grid using mean aggregation.

    Parameters
    ----------
    dfs_original : list of pd.DataFrame
        List of input DataFrames, each with a datetime index and numeric columns.
    gap_multiplier : float, optional
        Threshold multiplier for interpolation gap detection, passed to `interpolate()`. Default is 15.
    propagate_nan : bool, optional
        If True, keeps NaN values introduced during cleaning in each series. Default is True.
    display_info : bool, optional
        If True, prints information about the synchronization process. Default is False.

    Returns
    -------
    pd.DataFrame
        A unified multivariate time series where each original DataFrame is suffixed (`_0`, `_1`, etc.)
        and reindexed on the common timestamp grid.

    Raises
    ------
    ValueError
        If none of the input DataFrames have compatible sampling frequencies.

    Notes
    -----
    - The common frequency is chosen as the median of all valid inferred frequencies.
    - The synchronization strategy depends on time alignment:
        - If ranges are sufficiently close, a shared timeline is used.
        - Otherwise, a synthetic index starting from epoch is generated.
    - All series are trimmed to match the shortest series in the set.

    Examples
    --------
    >>> synced_df = synchronize_on_common_grid([df1, df2])
    >>> synced_df.shape
    (1000, 10)
    """

    # 1. Interpolate each DataFrame independently
    dfs = [
        interpolate(df, gap_multiplier=gap_multiplier, propagate_nan=propagate_nan)
        for df in dfs_original
    ]
    dfs = [df for df in dfs if not df.empty]

    # 2. Compute sampling frequencies
    freqs = [df.index.to_series().diff().median() for df in dfs]
    freqs_seconds = [f.total_seconds() for f in freqs if pd.notna(f)]
    median_freq = np.median(freqs_seconds)
    common_freq = pd.to_timedelta(median_freq, unit="s")

    # 3. Filter DataFrames too far from the median frequency
    allowed_diff = median_freq * 0.2
    dfs_filtered = [
        df for i, df in enumerate(dfs)
        if abs(freqs_seconds[i] - median_freq) <= allowed_diff
    ]
    if len(dfs_filtered) < len(dfs) and verb:
        print(f"[INFO] {len(dfs) - len(dfs_filtered)} DataFrame(s) ignored due to frequency mismatch.")
    dfs = dfs_filtered

    if not dfs:
        raise ValueError("No DataFrame matches the dominant frequency.")

    # 4. Check if time ranges are close (<10% of total duration)
    reference_ranges = [(df.index.min(), df.index.max()) for df in dfs]
    ref_start, ref_end = reference_ranges[0]
    total_duration = (ref_end - ref_start).total_seconds()
    allowed_shift = total_duration * 0.10  # 10% of the duration

    time_overlaps = all(
        abs((start - ref_start).total_seconds()) <= allowed_shift and
        abs((end - ref_end).total_seconds()) <= allowed_shift
        for start, end in reference_ranges
    )

    if time_overlaps:
        # Use the time grid of the shortest series
        durations = [(end - start).total_seconds() for start, end in reference_ranges]
        min_idx = int(np.argmin(durations))
        min_time = dfs[min_idx].index.min()
        max_time = dfs[min_idx].index.max()
        reference_index = pd.date_range(start=min_time, end=max_time, freq=common_freq)
        if display_info:
            print(f"[INFO] Using aligned time grid from {min_time} to {max_time} "
              f"with freq {common_freq} based on DataFrame #{min_idx}")
    else:
        # Neutral grid starting from 1970
        min_len = min(len(df) for df in dfs)
        reference_index = pd.date_range(
            start=pd.Timestamp("1970-01-01 00:00:00"),
            periods=min_len,
            freq=common_freq
        )
        print(f"[INFO] No aligned ranges → Using fresh index from 1970 "
              f"with length {min_len} and frequency {common_freq}")

    # 5. Reindex and resample on the common grid
    dfs_synced = []
    for i, df in enumerate(dfs):
        df_interp = df.interpolate(method="time", limit=gap_multiplier, limit_area="inside", limit_direction="both")
        df_synced = df_interp.resample(common_freq).mean()
        df_synced = df_synced.iloc[:len(reference_index)]
        df_synced.index = reference_index
        dfs_synced.append(df_synced)

    # 6. Concatenate into a multivariate DataFrame
    renamed_dfs = [df.add_suffix(f"_{i}") for i, df in enumerate(dfs_synced)]
    df_combined = pd.concat(renamed_dfs, axis=1)
    df_combined.index.name = "timestamp"

    return df_combined



def remove_linear_columns(df: pd.DataFrame, r2_threshold: float = 0.985) -> pd.DataFrame:
    """Remove numeric columns that are approximately linear.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    r2_threshold : float, optional
        Minimum R^2 value for a column to be considered linear.

    Returns
    -------
    pandas.DataFrame
        Dataframe without quasi-linear columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    to_drop = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 3:
            continue
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series.values, 1)
        fit = slope * x + intercept
        ss_res = np.sum((series.values - fit) ** 2)
        ss_tot = np.sum((series.values - series.values.mean()) ** 2)
        r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
        if r2 >= r2_threshold:
            to_drop.append(col)
    if to_drop and len(numeric_cols) > len(to_drop):
        df = df.drop(columns=to_drop)
    return df


def interpolate(
    df_origine: pd.DataFrame,
    gap_multiplier: float = 15,
    column_thresholds: dict = None,
    propagate_nan: bool = True,
) -> pd.DataFrame:
    """
    Clean and interpolate a time-indexed multivariate time series DataFrame.

    This function performs robust preprocessing of a time series by:
    - removing plateau-like constant value regions,
    - detecting and discarding extreme outliers using adaptive thresholds,
    - identifying and interpolating short gaps using time-aware linear interpolation,
    - reinjecting NaNs if desired (for later analysis or imputation),
    - removing nearly constant (linear) columns after interpolation.

    Parameters
    ----------
    df_origine : pd.DataFrame
        Input time series with a mandatory `"timestamp"` column (either as index or column).
    gap_multiplier : float, optional
        Determines what is considered a "large gap" by multiplying the inferred base frequency.
        Gaps larger than `gap_multiplier * freq` are not interpolated. Default is 15.
    column_thresholds : dict, optional
        Dictionary specifying per-column outlier detection thresholds.
        If not provided, thresholds are estimated automatically.
    propagate_nan : bool, optional
        If True, reintroduces NaN values resulting from plateau detection and outlier removal. Default is True.

    Returns
    -------
    pd.DataFrame
        Cleaned and interpolated DataFrame with uniform time index (`timestamp`),
        outliers removed, plateaus cleaned, and optionally missing data reinjected.

    Raises
    ------
    ValueError
        If the `"timestamp"` column is missing or cannot be parsed.
        If the time series is too short to infer a frequency.

    Notes
    -----
    - Outliers are detected based on deviation from the column median beyond a dynamic threshold.
    - Plateaus are constant regions that persist for ≥ 5 time steps and are globally removed.
    - Linear interpolation is applied within segments of consistent frequency,
      with discontinuities across large temporal gaps.
    - After interpolation, columns with <50% valid data or linear/constant trends are dropped.
    - The output is guaranteed to be indexed by timestamp and cleaned of duplicates or misaligned indices.

    Examples
    --------
    >>> df_clean = interpolate(df, gap_multiplier=10)
    >>> df_clean.shape
    (2000, 12)
    """

    df = df_origine.copy()
    
    def remove_plateau_values_globally(df: pd.DataFrame, min_duration=5) -> pd.DataFrame:
        df = df.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col]
            to_remove_values = set()
            run_value = None
            run_length = 0
            for i in range(len(s)):
                current = s.iat[i]
                if pd.isna(current):
                    run_value = None
                    run_length = 0
                    continue
                if current == run_value:
                    run_length += 1
                else:
                    if run_length >= min_duration:
                        to_remove_values.add(run_value)
                    run_value = current
                    run_length = 1
            if run_length >= min_duration:
                to_remove_values.add(run_value)
            if to_remove_values:
                mask = s.isin(to_remove_values)
                df.loc[mask, col] = np.nan
        return df

    # Remove non-numeric columns except "timestamp"
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns
    non_numeric_cols = [col for col in non_numeric_cols if col != "timestamp"]
    df = df.drop(columns=non_numeric_cols)

    # Save original NaN positions
    original_nan_mask = df.isna()

    # Remove plateau-like segments
    df = remove_plateau_values_globally(df)

    # Identify NaNs introduced by plateau removal
    new_nan_mask = df.isna() & ~original_nan_mask

    # Drop columns with too many NaNs after plateau removal
    min_valid_points_plateaux = int(len(df) * 0.5)
    df = df.dropna(axis=1, thresh=min_valid_points_plateaux)

    # Keep only remaining columns in the mask
    new_nan_mask = new_nan_mask[df.columns]
    # Handle timestamp column or index
    if isinstance(df.index, pd.DatetimeIndex):
        pass  # Already set correctly
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")
    else:
        raise ValueError("The 'timestamp' column is required for interpolation.")
    df.index = df.index.tz_localize(None)
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # Seuils automatiques
    def estimate_column_thresholds(data: pd.DataFrame) -> dict:
        thresholds = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            top_mean = data[col].nlargest(int(len(data) * 0.15)).mean()
            bot_mean = data[col].nsmallest(int(len(data) * 0.15)).mean()
            thresholds[col] = (top_mean - bot_mean) * 2.0
        return thresholds

    if column_thresholds is None:
        column_thresholds = estimate_column_thresholds(df)

    # Extreme outliers
    outlier_timestamps = set()
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col]
        threshold = column_thresholds.get(col, np.inf)
        center = series.median()
        mask = np.abs(series - center) > threshold
        outlier_timestamps.update(df.index[mask])

    # Remove outlier rows
    df_wo_outliers = df.drop(index=outlier_timestamps)
    min_valid_points = int(len(df_wo_outliers) * 0.5)
    df_wo_outliers = df_wo_outliers.dropna(axis=1, thresh=min_valid_points)

    # Estimate sampling frequency
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

    if pd.isna(freq) or freq <= pd.Timedelta(seconds=0):
        freq = pd.Timedelta(seconds=1)

    max_gap = freq * gap_multiplier

    # Resampling
    full_idx = pd.date_range(start=idx.min(), end=idx.max(), freq=freq)
    union_idx = idx.union(full_idx)
    df_union = df_wo_outliers.reindex(union_idx)
    df_union = df_union.infer_objects(copy=False)

    diffs = idx.to_series().diff()
    seg_ids = (diffs > max_gap).cumsum()
    seg_index = pd.Series(seg_ids.values, index=idx)
    seg_union = seg_index.reindex(union_idx, method="ffill").fillna(0).astype(int)

    df_interp = df_union.groupby(seg_union, group_keys=False).apply(
        lambda seg: seg.interpolate(method="time",
                                        limit=gap_multiplier,
                                        limit_direction="both", 
                                        limit_area="inside",
                                    )
    )

    df_out = df_interp.reindex(full_idx)
    df_out.index.name = "timestamp"

    # Large gaps become NaN
    gap_mask = pd.Series(False, index=df_out.index)
    for prev, nxt in zip(idx[:-1], idx[1:]):
        if nxt - prev > max_gap:
            in_gap = (df_out.index > prev) & (df_out.index <= nxt)
            gap_mask |= in_gap
    df_out.loc[gap_mask, :] = np.nan

    # Reinject NaNs from outliers
    if propagate_nan:
        outlier_idx = sorted(ts for ts in outlier_timestamps if ts in df_out.index)
        df_out.loc[outlier_idx, :] = np.nan
    else:
        for col in df_out.columns:
            original_col = col if col in df.columns else None
            if original_col is None:
                continue
            series = df[original_col]
            threshold = column_thresholds.get(original_col, np.inf)
            center = series.median()
            mask = np.abs(series - center) > threshold
            ts_nan = df.index[mask]
            ts_nan = [ts for ts in ts_nan if ts in df_out.index]
            df_out.loc[ts_nan, col] = np.nan
    # Reinject NaNs from plateau removal
    new_nan_mask = new_nan_mask.reindex(df.index)

    if propagate_nan:
        plateau_nan_locs = []
        for col in new_nan_mask.columns:
            if col == "timestamp":
                continue
            mask = new_nan_mask[col]
            if mask.dtype != bool:
                mask = mask.astype("boolean").fillna(False)
                mask = mask.fillna(False)
                mask = mask.fillna(False)
                mask = mask.astype(bool)
            plateau_nan_locs.extend(mask[mask].index)
        plateau_nan_locs = sorted(set(ts for ts in plateau_nan_locs if ts in df_out.index))
        df_out.loc[plateau_nan_locs, :] = np.nan
    else:
        for col in new_nan_mask.columns:
            if col == "timestamp":
                continue
            mask = new_nan_mask[col]
            if mask.dtype != bool:
                mask = mask.astype("boolean").fillna(False)
                mask = mask.fillna(False)
                mask = mask.astype(bool)
            ts_nan = mask[mask].index
            ts_nan = [ts for ts in ts_nan if ts in df_out.index]
            df_out.loc[ts_nan, col] = np.nan

    min_valid_ratio = 0.50
    min_valid_points = int(len(df_out) * min_valid_ratio)
    df_out = df_out.dropna(axis=1, thresh=min_valid_points)
    df_out = remove_linear_columns(df_out)
    return df_out


@njit  # comment njit for coverage
def _compute_aswn(
    values: np.ndarray,
    window_size: int,
    min_std: float,
    min_valid_ratio: float
) -> np.ndarray:
    """Fast local window normalization (ASWN) with Numba.

    Parameters
    ----------
    values : numpy.ndarray
        Input one-dimensional array.
    window_size : int
        Length of the sliding window.
    min_std : float
        Minimum allowed standard deviation.
    min_valid_ratio : float
        Minimum ratio of valid values within a window.

    Returns
    -------
    numpy.ndarray
        Normalized array with the same shape as ``values``.
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
    """ASWN normalization with optional trend blending.

    Parameters
    ----------
    series : pandas.Series
        Input time series.
    window_size : int, optional
        Sliding window size used for the normalization.
    min_std : float, optional
        Minimum allowed standard deviation.
    min_valid_ratio : float, optional
        Minimum ratio of valid values in a window.
    alpha : float, optional
        Trend blending coefficient (0 to 1).

    Returns
    -------
    pandas.Series
        Normalized series with the same index as ``series``.
    """
    values = series.values.astype(np.float64)
    normed = _compute_aswn(values, window_size, min_std, min_valid_ratio)
    result = pd.Series(normed, index=series.index)
    if alpha > 0:
        trend = series.rolling(
            window=window_size * 10, center=True, min_periods=1
        ).mean()
        return (1 - alpha) * result + alpha * (series - trend)
    return result


def normalization(
    df: pd.DataFrame,
    min_std: float = 1e-2,
    min_valid_ratio: float = 0.5,
    alpha: float = 0.65,
    window_size: str = None,
    gap_multiplier: int = 15,
    propagate_nan:bool = True
) -> pd.DataFrame:
    """Apply ASWN normalization with trend correction.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe indexed by time.
    min_std : float, optional
        Minimum allowed standard deviation.
    min_valid_ratio : float, optional
        Minimum ratio of valid values within a window.
    alpha : float, optional
        Trend blending coefficient (0 to 1).
    window_size : str or None, optional
        Window size string or number of samples. ``None`` uses a default.
    gap_multiplier : float, optional
        Gap multiplier for interpolation.
    propagate_nan : bool, optional
        Reinject NaN values introduced during cleaning steps.

    Returns
    -------
    pandas.DataFrame or None
        Normalized dataframe or ``None`` if no valid columns remain.
    """
    df = df.copy()
    if df.empty or len(df.columns) == 0:
        print("[INFO] All columns removed (linear or invalid). Returning None.")
        return None

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    if not isinstance(window_size, int):
        window_size_int = int(pd.Timedelta(window_size) / df.index.to_series().diff().median())
    else:
        window_size_int = window_size
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = aswn_with_trend(
                df[col], window_size_int, min_std, min_valid_ratio, alpha
            )
            df[col] = df[col].interpolate(method="linear", limit=gap_multiplier, limit_direction="both", limit_area="inside")

    df.attrs["m"] = [window_size, window_size_int]
    df = df.loc[:, df.notna().sum() >= window_size_int]  # at least the size of one window        
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if propagate_nan:
        any_nan_mask = df.isna().any(axis=1)
        df.loc[any_nan_mask] = np.nan
    def has_enough_valid_windows(series, window_size, min_ratio):
        if len(series) < window_size:
            return False

        # Count how many valid values are in each window
        valid_counts = series.rolling(window_size).count()

        # A window is valid if it contains exactly ``window_size`` non-NaN values
        valid_windows = (valid_counts == window_size).sum()

        total_windows = len(series) - window_size + 1
        ratio = valid_windows / max(1, total_windows)

        return True if ratio >= min_ratio else False

    if not has_enough_valid_windows(df[df.columns[0]], window_size_int, 0.5):
        print("[INFO] No dimension has enough valid windows. Returning None")
        return None
    else:
        return df

@njit
def normalize_segments(segs: np.ndarray) -> np.ndarray:
    n, m = segs.shape
    for i in range(n):
        mean = 0.0
        std = 0.0
        for j in range(m):
            mean += segs[i, j]
        mean /= m
        for j in range(m):
            std += (segs[i, j] - mean) ** 2
        std = (std / m) ** 0.5
        if std < 1e-8:
            std = 1e-8
        for j in range(m):
            segs[i, j] = (segs[i, j] - mean) / std
    return segs

@njit
def compute_variances(segs: np.ndarray) -> np.ndarray:
    n, m = segs.shape
    variances = np.empty(n, dtype=np.float32)
    for i in range(n):
        mean = 0.0
        for j in range(m):
            mean += segs[i, j]
        mean /= m
        var = 0.0
        for j in range(m):
            var += (segs[i, j] - mean) ** 2
        variances[i] = var / m
    return variances


def define_m(
    df: pd.DataFrame,
    k: int = 3,
    max_window_sizes: int = 25,
    max_points: int = 5000,
    verbose: bool = False
) -> list[tuple[int, str, float]]:
    """
    Suggest optimal window sizes for time series motif discovery using STUMP or MSTUMP.

    This function analyzes a time-indexed DataFrame to extract the `k` most promising window sizes
    for motif extraction. It adapts to the data's temporal resolution and supports both univariate
    and multivariate input. Window sizes are scored based on a composite metric that includes
    motif strength, discord separation, and local stability.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame with numerical columns. Index must be a `pd.DatetimeIndex`.
    k : int, optional
        Number of top window sizes to return (sorted by descending score), by default 3.
    max_window_sizes : int, optional
        Maximum number of distinct window durations (in seconds) to consider for evaluation, by default 25.
    max_points : int, optional
        Maximum number of rows to keep from `df` for performance reasons. A subsample is used if exceeded, by default 5000.
    verbose : bool, optional
        If True, prints errors encountered during STUMP/MSTUMP computation, by default False.

    Returns
    -------
    list of tuple (int, str, float)
        A list of the top `k` window sizes, each as a tuple:
        - number of points (int),
        - human-readable timedelta string (str),
        - associated composite score (float), where higher is better.

    Raises
    ------
    ValueError
        If the index is not a `pd.DatetimeIndex`, or if no numeric columns are found.

    Notes
    -----
    - The function determines an adaptive range of time-based window sizes depending on the data's sampling frequency.
    - Motif score is penalized for high global noise and insufficient discord separation.
    - Final scores combine:
        * inverse of mean/percentile distances (motif strength),
        * discord separation (p98 - p90),
        * low variation of local gradient (stability),
        * penalties for noise and overly long windows.

    Examples
    --------
    >>> define_m(my_dataframe, k=5)
    [(60, '0 days 00:01:00', 0.0721), (120, '0 days 00:02:00', 0.0653), ...]
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be datetime")

    # Fréquence et durée totales
    freq = df.index.to_series().diff().median()
    freq = freq if pd.notna(freq) and freq > pd.Timedelta(0) else pd.Timedelta(seconds=1)
    freq_secs = freq.total_seconds()
    total_secs = (df.index[-1] - df.index[0]).total_seconds()
    N = len(df)


    if freq_secs <= 1e-6:                 # ≤ 1 µs (nanoseconde)
        base_secs, max_secs = 0.01, 1             # 10 ms à 1 s
    elif freq_secs <= 1e-3:              # ≤ 1 ms (microseconde)
        base_secs, max_secs = 0.1, 5              # 100 ms à 5 s
    elif freq_secs <= 1e-2:              # ≤ 10 ms
        base_secs, max_secs = 1, 60               # 1 s à 1 min
    elif freq_secs <= 0.1:               # ≤ 100 ms
        base_secs, max_secs = 5, 60               # 5 s à 1 min
    elif freq_secs <= 1:                 # ≤ 1 s
        base_secs, max_secs = 10, 336             # 10 s à ~5.6 min
    elif freq_secs <= 10:                # ≤ 10 s
        base_secs, max_secs = 30, 3360            # 30 s à 56 min
    elif freq_secs <= 60:                # ≤ 1 min
        base_secs, max_secs = 300, 21_600          # 5 min à 6 h
    elif freq_secs <= 300:               # ≤ 5 min
        base_secs, max_secs = 600, 86_400          # 10 min à 24 h
    elif freq_secs <= 1800:              # ≤ 30 min
        base_secs, max_secs = 1800, 604_800        # 30 min à 7 j
    elif freq_secs <= 3600:              # ≤ 1 h
        base_secs, max_secs = 3600, 604_800        # 1 h à 7 j
    elif freq_secs <= 43200:             # ≤ 12 h
        base_secs, max_secs = 21600, 604_800       # 6 h à 7 j
    elif freq_secs <= 86400:             # ≤ 1 jour
        base_secs, max_secs = 43200, 2_678_400       # 12 h à 1 M
    else:                              # > 1 jour
        base_secs, max_secs = 86400, min(2592000, total_secs / 2)  # 1 j à 30 j

    max_secs = min(max_secs, total_secs / 2)
    base_secs = min(base_secs, max_secs)

    secs_cand = np.unique(
        np.round(np.logspace(np.log10(base_secs), np.log10(max_secs), num=max_window_sizes)).astype(int)
    )
    window_sizes = [str(pd.to_timedelta(int(s), unit='s')) for s in secs_cand]

    window_sizes_pts = []
    for ws in window_sizes:
        pts = int(pd.Timedelta(ws) / freq)
        if pts >= 10 and pts < N // 2:
            window_sizes_pts.append((pts, ws))

    if N > max_points:
        idx = np.linspace(0, N - 1, max_points, dtype=int)
        df = df.iloc[idx]
        N = len(df)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric columns")
    data = df[numeric_cols].values.astype(float)

    aggregated = []
    global_noise = np.nanmean(np.std(data, axis=0))

    for pts, ws_str in window_sizes_pts:
        try:
            mp = stumpy.stump(data[:, 0], m=pts) if data.shape[1] == 1 else stumpy.mstump(data.T, m=pts)
            profile = (mp[:, 0] if data.shape[1] == 1 else mp[-1])
            profile = np.array(profile, dtype=np.float64)
            profile = profile[~np.isnan(profile)]
            if len(profile) < 2:
                continue

            mean_p = profile.mean()
            p90 = np.percentile(profile, 90)
            p98 = np.percentile(profile, 98)
            disco_ratio = (profile > 2 * np.median(profile)).mean()
            grad = np.abs(np.diff(profile)).mean()

            blocks = np.array_split(profile, 10)
            local_vars = [np.var(b) for b in blocks if len(b) > 1]
            local_var_med = np.median(local_vars) if local_vars else np.var(profile)

            motif_score = 1 / ((mean_p + p90) * (local_var_med + 1e-6))
            disco_sep = max(0.001, p98 - p90)
            score = 0.6 * motif_score + 0.2 * disco_sep + 0.2 / (grad + 1e-6)

            if disco_ratio > 0.1:
                score *= 0.5
            if global_noise > 0.25:
                score *= 0.8

            score /= (1 + pts / 3000)
            aggregated.append((pts, ws_str, score))

        except Exception as e:
            if verbose:
                print(f"Erreur fenêtre {pts}: {e}")
            continue

    aggregated = [t for t in aggregated if not np.isnan(t[2])]
    aggregated.sort(key=lambda x: -x[2])
    return aggregated[:k]



def cluster_dimensions(
    df: Union[pd.DataFrame, List[pd.DataFrame]],
    group_size: int = 6,
    top_k: int = 4,
    min_std: float = 1e-2,
    min_valid_ratio: float = 0.8,
    min_cluster_size: int = 2,
    mode: str = 'hybrid',
    display_info: bool = False,
) -> List[List[str]]:
    """
    Cluster time series dimensions (columns) based on correlation coherence.

    This function groups sensor or signal dimensions from one or several DataFrames
    into clusters of highly correlated columns, based on their temporal patterns.
    The clustering is hierarchical and uses pairwise correlation-derived distances.
    Only informative (non-flat, sufficiently populated) columns are retained.

    Parameters
    ----------
    df : Union[pd.DataFrame, List[pd.DataFrame]]
        Input data, either a single DataFrame or a list of DataFrames.
        Each must have a time-based index and numeric columns.
    group_size : int, optional
        Maximum number of sensor modalities per cluster, by default 6.
    top_k : int, optional
        Maximum number of clusters to return, by default 4.
    min_std : float, optional
        Minimum standard deviation required for a column to be considered valid, by default 1e-2.
    min_valid_ratio : float, optional
        Minimum ratio of non-NaN values required for a column, by default 0.8.
    min_cluster_size : int, optional
        Minimum number of columns for a cluster to be kept, by default 2.
    mode : {'motif', 'discord', 'hybrid'}, optional
        Strategy to compute distances between columns:
        - 'motif': favors similarly varying curves (square of correlation),
        - 'discord': penalizes opposite sign trends (absolute correlation),
        - 'hybrid': uses raw correlation (positive and negative preserved).
    display_info : bool, optional
        If True, displays inter- and intra-cluster correlations for diagnostics.

    Returns
    -------
    List[List[str]]
        A list of clusters, each a list of column names including 'timestamp'.
        Only clusters containing sufficiently complete sensor units are returned.

    Raises
    ------
    TypeError
        If `df` is neither a DataFrame nor a list of DataFrames.
    ValueError
        If `mode` is not one of {'motif', 'discord', 'hybrid'}.

    Notes
    -----
    - Clusters are filtered to ensure that they contain diverse sensor families (modalities).
    - Correlation matrices are used as similarity measures, transformed into distances.
    - Final clusters are cleaned to retain only complete sensor units (i.e., sets of modalities with matching indices).
    - This function is well-suited for preprocessing multivariate time series before dimensionality reduction or motif analysis.

    Examples
    --------
    >>> clusters = cluster_dimensions(df, group_size=3, top_k=2)
    >>> print(clusters)
    [['sensorA_1', 'sensorB_1', 'sensorC_1', 'timestamp'], 
     ['sensorA_2', 'sensorB_2', 'sensorC_2', 'timestamp']]
    """
    if isinstance(df, pd.DataFrame):
        df_list = [df]
    elif isinstance(df, list) and all(isinstance(x, pd.DataFrame) for x in df):
        df_list = df
    else:
        raise TypeError("df must be a DataFrame or a list of DataFrames")

    if mode not in {'motif', 'discord', 'hybrid'}:
        raise ValueError("mode must be 'motif', 'discord', or 'hybrid'")

    clusters = []

    for base_df in df_list:
        base_df = base_df.select_dtypes(include=[np.number])

        valid_cols = [
            col for col in base_df.columns
            if base_df[col].std() >= min_std and base_df[col].notna().mean() >= min_valid_ratio
        ]
        if len(valid_cols) < 2:
            continue

        # 1. Normalisation z-score
        base_df[valid_cols] = (
            base_df[valid_cols] - base_df[valid_cols].mean()
        ) / base_df[valid_cols].std()

        # 2. Matrice de distance
        corr = base_df[valid_cols].corr()

        if mode == 'motif':
            dist = 1 - corr.pow(2).abs()
        elif mode == 'discord':
            dist = 1 - corr.abs()
        elif mode == 'hybrid':
            dist = 1 - corr

        dist = dist.fillna(1.0)

        tril_values = dist.where(np.tril(np.ones(dist.shape), -1).astype(bool)).stack()
        mean_dist = tril_values.mean()
        std_dist = tril_values.std()
        distance_threshold = mean_dist - 0.3 * std_dist
        model = AgglomerativeClustering(
            metric='precomputed',
            linkage='average',
            distance_threshold=distance_threshold,
            n_clusters=None
        )
        labels = model.fit_predict(dist.values)

        label_map = {}
        for col, label in zip(valid_cols, labels):
            label_map.setdefault(label, []).append(col)

        def avg_corr(cols):
            if len(cols) < 2:
                return 0
            matrix = base_df[cols].corr().abs()
            tril = matrix.where(np.tril(np.ones(matrix.shape), -1).astype(bool))
            return tril.stack().mean()

        def diverse_enough(cols):
            families = set(c.rsplit("_", 1)[0] for c in cols)
            return len(families) > 1

        sorted_clusters = sorted(
            [c for c in label_map.values()
             if len(c) >= min_cluster_size and diverse_enough(c)
            ],
            key=avg_corr,
            reverse=True
        )

        for cluster_cols in sorted_clusters:
            if len(clusters) >= top_k:
                break
            clusters.append(cluster_cols[:group_size+1] + ["timestamp"])
        if display_info:
            print("\n[CLUSTERING]")
            print("\n Cross correlation between columns in each clusters :")

        for i, cluster in enumerate(clusters):
            cluster_vars = [col for col in cluster if col != "timestamp"]
            others = [col for col in base_df.columns if col not in cluster_vars]

            if not cluster_vars or not others:
                continue

            sub_corr = base_df[cluster_vars + others].corr().loc[cluster_vars, others]
            mean_cross_corr = sub_corr.abs().mean().mean()
            if display_info:
                print(f"    Cluster {i+1:02d} ({len(cluster_vars)} variables) correlation = {mean_cross_corr:.3f}")
        
        if display_info:
            print("\n Correlation between parameters in sensors :")

            family_map = defaultdict(list)
            for col in base_df.columns:
                if col == "timestamp":
                    continue
                match = re.match(r"(.*?)(?:_\d+)?$", col)
                if match:
                    family = match.group(1)
                    family_map[family].append(col)

            family_names = sorted(family_map)
            family_corr = pd.DataFrame(index=family_names, columns=family_names, dtype=float)

            for f1 in family_names:
                for f2 in family_names:
                    cols1, cols2 = family_map[f1], family_map[f2]
                    sub_corr = base_df[cols1 + cols2].corr().loc[cols1, cols2]
                    mean_corr = sub_corr.abs().mean().mean()
                    family_corr.loc[f1, f2] = mean_corr

            print(f"{family_corr.round(2)}")

            print("\n")
    if not clusters:
        print("Skipping DataFrame: not enough rich dimensions — too much noise, constant values, or missing data.")
        
    def filter_cluster_to_complete_units(cluster_cols):
        cols = [c for c in cluster_cols if c != 'timestamp']
        units = defaultdict(set)
        for col in cols:
            try:
                modality, idx = col.rsplit("_", 1)
                units[idx].add(modality)
            except ValueError:
                continue

        if not units:
            return []

        max_modalities = max(len(mods) for mods in units.values())

        min_modalities = max(1, max_modalities // 1.5)

        complete_indices = [(idx) for idx, mods in sorted(units.items(), key=lambda x: int(x[0])) if len(mods) >= min_modalities]
        
        complete_cols = [
            c for c in cols if c.rsplit("_", 1)[1] in complete_indices
        ]

        if 'timestamp' in cluster_cols:
            complete_cols.append('timestamp')

        return complete_cols

    clusters = [filter_cluster_to_complete_units(c) for c in clusters]
    clusters = [c for c in clusters if c] 
    return clusters



def pre_processed(
    data: Union[pd.DataFrame, List[pd.DataFrame]],
    gap_multiplier: float = 15,
    min_std: float = 1e-2,
    min_valid_ratio: float = 0.8,
    alpha: float = 0.65,
    window_size: str = None,
    sort_by_variables: bool = True,
    cluster: bool = False,
    mode: str = 'hybrid',
    top_k_cluster: int = 4,
    group_size: int = 16,
    display_info: bool = False,
    smart_interpolation: bool = True,
) -> Union[pd.DataFrame, List[pd.DataFrame], List[List[pd.DataFrame]]]:
    """
    Interpolate, normalize, and optionally cluster time series data.

    This function processes one or multiple multivariate time series by:
    - interpolating missing values,
    - normalizing signals using adaptive windowing and trend blending,
    - optionally clustering correlated dimensions for grouped processing.

    It returns interpolated and normalized versions of the input data,
    as well as optionally a version preserving missing values post-normalization.

    Parameters
    ----------
    data : Union[pd.DataFrame, List[pd.DataFrame]]
        Input dataset(s), either a single DataFrame or a list of DataFrames.
        Each must have a datetime index and numeric columns.
    gap_multiplier : float, optional
        Controls sensitivity for detecting gaps during interpolation, by default 15.
    min_std : float, optional
        Minimum standard deviation for a column to be retained during normalization, by default 1e-2.
    min_valid_ratio : float, optional
        Minimum fraction of valid (non-NaN) values in a column, by default 0.8.
    alpha : float, optional
        Blending factor between trend and residual in normalization (0 to 1), by default 0.65.
    window_size : str or None, optional
        Time-based window size (e.g., '30s', '1min') for normalization. If None, will be estimated dynamically.
    sort_by_variables : bool, optional
        Whether to sort columns by variance before clustering (not currently used), by default True.
    cluster : bool, optional
        If True, performs clustering of time series dimensions before normalization, by default False.
    mode : {'motif', 'discord', 'hybrid'}, optional
        Distance mode used for clustering (via `cluster_dimensions`), by default 'hybrid'.
    top_k_cluster : int, optional
        Maximum number of clusters to return, by default 4.
    group_size : int, optional
        Target number of modalities per cluster, by default 6.
    display_info : bool, optional
        If True, prints diagnostic information about correlation structure and clustering, by default False.
    smart_interpolation : bool, optional
        If True, also returns a normalization with missing values preserved, allowing for matrix-profile-based imputation, by default True.

    Returns
    -------
    Union[
        Tuple[pd.DataFrame, pd.DataFrame, None],
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]
    ]
        Depending on the input and options:
        - Without clustering: (interpolated_df, normalized_df, None) or with smart_interpolation=True, the third element is the normalized version with NaNs preserved.
        - With clustering: three lists of DataFrames corresponding to clustered/interpolated, normalized, and normalized-with-NaN-preserved outputs.

    Raises
    ------
    TypeError
        If input data is not a DataFrame or list of DataFrames.
    ValueError
        If the estimated or provided window size is too large for a given signal length.

    Notes
    -----
    - For clustered inputs, each cluster is processed independently with its own optimal window size.
    - The third return value is designed for cases where post-processing (e.g., imputation or motif detection) may require keeping original NaN structure.
    - `define_m()` is used internally to suggest a suitable window size based on stability criteria.

    Examples
    --------
    >>> interpolated, normalized, _ = pre_processed(df)
    >>> interpolated_list, normalized_list, with_nan_list = pre_processed(list_of_dfs, cluster=True)
    """

    def get_window_size(df, fallback_len):
        if window_size is not None:
            if int(pd.Timedelta(window_size) / df.index.to_series().diff().median()) >= len(df) // 2:
                raise ValueError("Window size is too large for the signal length.")
            return window_size
        try:
            win_list = define_m(df)
            return win_list[0][1] if win_list else max(2, fallback_len // 2)
        except ValueError:
            return max(2, fallback_len // 2)

    def process_group(df_group, smart_interpolation):
        interpolated = interpolate(df_group, gap_multiplier=gap_multiplier)
        if interpolated.empty:
            return None, None, None
        final_ws = get_window_size(interpolated, len(interpolated))
        normalized = normalization(
            interpolated,
            min_std=min_std,
            min_valid_ratio=min_valid_ratio,
            alpha=alpha,
            window_size=final_ws,
            gap_multiplier=gap_multiplier,
        ) 
        if smart_interpolation:
            normalized_whithout_inter = normalization(
                interpolated,
                min_std=min_std,
                min_valid_ratio=min_valid_ratio,
                alpha=alpha,
                window_size=final_ws,
                gap_multiplier=gap_multiplier,
                propagate_nan = False,
            ) 
        else:
            normalized_whithout_inter = None
        return interpolated, normalized, normalized_whithout_inter

    # === Type checking ===
    if isinstance(data, list) and not all(isinstance(x, pd.DataFrame) for x in data):
        raise TypeError("df must be a DataFrame or a list of DataFrames")

    # === Case 1: list of DataFrames ===
    if isinstance(data, list):
        if cluster:
            synced_dfs = synchronize_on_common_grid(data, propagate_nan=False, display_info=display_info, gap_multiplier=gap_multiplier)
            clustered_groups = cluster_dimensions(synced_dfs, top_k=top_k_cluster, mode=mode, group_size=group_size, display_info=display_info)

            group_result, group_result_normalize, group_result_normalize_whithout_Nan = [], [], []
            for col_names in sorted(clustered_groups):
                df_cluster = synced_dfs.reset_index()[col_names]
                interpolated, normalized, normalized_whithout_inter = process_group(df_cluster, smart_interpolation)
                if interpolated is not None:
                    group_result.append(interpolated)
                if normalized is not None:
                    group_result_normalize.append(normalized)
                if normalized_whithout_inter is not None:
                    group_result_normalize_whithout_Nan.append(normalized_whithout_inter)
            return group_result, group_result_normalize, group_result_normalize_whithout_Nan
        else:
            df_interpolate = synchronize_on_common_grid(data, propagate_nan=True, display_info=display_info)
            final_ws = get_window_size(df_interpolate, len(df_interpolate))
            df_normalize = normalization(
                df_interpolate,
                min_std=min_std,
                min_valid_ratio=min_valid_ratio,
                alpha=alpha,
                window_size=final_ws
            ) 
            return df_interpolate, df_normalize, None

    # === Case 2 : one DataFrame ===
    df = data
    df_interpolate = interpolate(df.copy(), gap_multiplier=gap_multiplier)
    final_ws = get_window_size(df_interpolate, len(df_interpolate))
    df_normalize = normalization(
        df_interpolate,
        min_std=min_std,
        min_valid_ratio=min_valid_ratio,
        alpha=alpha,
        window_size=final_ws
    ) 
    return df_interpolate, df_normalize, None


def interpolate_from_matrix_profile(
    df: pd.DataFrame,
    matrix_profile: pd.DataFrame,
    top_k: int = 3,
    min_valid_ratio: float = 0.5,
) -> pd.DataFrame:
    """
    Interpolate missing values in each column using similar columns (with the same prefix),
    based on their precomputed matrix profiles.

    Parameters
    ----------
    df : pd.DataFrame
        Original input data.
    matrix_profile : pd.DataFrame
        Centered matrix profile (indexed by timestamps, with columns named "mp_dim_<col>").
    top_k : int
        Number of most similar columns to use for interpolation.
    min_valid_ratio : float
        Minimum ratio of overlapping valid (non-NaN) values required to compare matrix profiles.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with interpolated columns where applicable.
    """
    df_result = df.copy()
    matrix_profile = matrix_profile["matrix_profile"]

    for target_col in df.columns:
        if df[target_col].isna().any():

            if f"mp_dim_{target_col}" not in matrix_profile.columns:
                continue

            prefix = target_col.split("_")[0]
            candidates = [
                col for col in df.columns
                if col.startswith(prefix)
                and col != target_col
                and f"mp_dim_{col}" in matrix_profile.columns
            ]

            if not candidates:
                continue

            target_mp = matrix_profile[f"mp_dim_{target_col}"]

            scores = {}
            for col in candidates:
                mp_col = matrix_profile[f"mp_dim_{col}"]

                valid = target_mp.notna() & mp_col.notna()
                if valid.mean() < min_valid_ratio:
                    continue

                score = np.mean(np.abs(target_mp[valid] - mp_col[valid]))
                scores[col] = score

            if not scores:
                continue

            best_cols = sorted(scores, key=scores.get)[:top_k]
            fill_values = df[best_cols].mean(axis=1, skipna=True)

            result = df[target_col].copy()
            result[result.isna()] = fill_values[result.isna()]
            df_result[target_col] = result

    return df_result

    
def interpolate_all_columns_by_similarity(
    pds,
    matrix_profiles,
    top_k: int = 3,
    min_valid_ratio: float = 0.5,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Interpolate missing values in all columns based on similarity in matrix profile.

    Parameters
    ----------
    pds : pd.DataFrame or list of pd.DataFrame
        Input time series or clustered time series.
    matrix_profiles : pd.DataFrame or list of pd.DataFrame
        Corresponding matrix profiles (1:1 with df if list).
    top_k : int
        Number of most similar signals to use for interpolation.
    min_valid_ratio : float
        Threshold to ignore columns with too few valid points.

    Returns
    -------
    pd.DataFrame or list of pd.DataFrame
        Interpolated time series.
    """
    # Cas multiple (liste de clusters)
    interpolated_clusters = []
    for df, matrix_profile in zip(pds, matrix_profiles):
        interpolated = interpolate_from_matrix_profile(
            df, matrix_profile, top_k, min_valid_ratio
        )
        interpolated_clusters.append(interpolated)
    return interpolated_clusters

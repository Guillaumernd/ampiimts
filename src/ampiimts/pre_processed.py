"""Utility functions for preprocessing time series DataFrames."""
from typing import Union, List
from collections import Counter, defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations
from joblib import Parallel, delayed
import warnings
import numpy as np
import pandas as pd
from numba import njit
import faiss
import time
import re
import random


def synchronize_on_common_grid(
    dfs_original: List[pd.DataFrame],
    gap_multiplier: float = 15,
    propagate_nan: bool = True,
) -> pd.DataFrame:
    """
    Synchronizes multiple time series DataFrames on a common timestamp grid
    and returns a single multivariate DataFrame (columns renamed to avoid duplicates).
    """

    # 1. Independent interpolation of each DataFrame
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

    # 3. Drop DataFrames far from the median frequency
    allowed_diff = median_freq * 0.2
    dfs_filtered = [
        df for i, df in enumerate(dfs)
        if abs(freqs_seconds[i] - median_freq) <= allowed_diff
    ]
    if len(dfs_filtered) < len(dfs):
        print(f"[INFO] {len(dfs) - len(dfs_filtered)} DataFrame(s) skipped due to distant frequency")
    dfs = dfs_filtered

    if not dfs:
        raise ValueError("No DataFrame matches the dominant frequency")

    # 4. Check if time ranges are aligned (<10% total duration)
    reference_ranges = [(df.index.min(), df.index.max()) for df in dfs]
    ref_start, ref_end = reference_ranges[0]
    total_duration = (ref_end - ref_start).total_seconds()
    allowed_shift = total_duration * 0.10  # 10% of duration

    time_overlaps = all(
        abs((start - ref_start).total_seconds()) <= allowed_shift and
        abs((end - ref_end).total_seconds()) <= allowed_shift
        for start, end in reference_ranges
    )

    if time_overlaps:
        # Time grid based on the shortest series
        durations = [(end - start).total_seconds() for start, end in reference_ranges]
        min_idx = int(np.argmin(durations))
        min_time = dfs[min_idx].index.min()
        max_time = dfs[min_idx].index.max()
        reference_index = pd.date_range(start=min_time, end=max_time, freq=common_freq)
        print(f"[INFO] Using aligned time grid from {min_time} to {max_time} "
              f"with freq {common_freq} based on DataFrame #{min_idx}")
    else:
        # Neutral grid starting in 1970
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
        df_interp = df.interpolate(method="time", limit=2, limit_area="inside", limit_direction="both")
        df_synced = df_interp.resample(common_freq).mean()
        df_synced = df_synced.iloc[:len(reference_index)]
        df_synced.index = reference_index
        dfs_synced.append(df_synced)

    # 6. Concatenate all series
    renamed_dfs = [df.add_suffix(f"_{i}") for i, df in enumerate(dfs_synced)]
    df_combined = pd.concat(renamed_dfs, axis=1)
    df_combined.index.name = "timestamp"

    return df_combined



def remove_linear_columns(df: pd.DataFrame, r2_threshold: float = 0.985) -> pd.DataFrame:
    """Remove numeric columns that are approximately linear.

    A column is considered linear if a first-degree polynomial fitted on its
    non-NaN values explains more than ``r2_threshold`` of the variance.
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
    df_original: pd.DataFrame,
    gap_multiplier: float = 15,
    column_thresholds: dict = None,
    propagate_nan: bool = True,
) -> pd.DataFrame:
    df = df_original.copy()
    
    def remove_plateau_values_globally(df: pd.DataFrame, min_duration=5) -> pd.DataFrame:
        """Replace long constant plateaus by NaN for all numeric columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        min_duration : int, optional
            Minimum length of a constant run to be considered a plateau.

        Returns
        -------
        pd.DataFrame
            DataFrame with plateaus replaced by ``NaN``.
        """
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

    # Drop no numeric columns except for an explicit timestamp column
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns
    non_numeric_cols = [col for col in non_numeric_cols if col != "timestamp"]
    df = df.drop(columns=non_numeric_cols)

    # Save original NaN positions
    original_nan_mask = df.isna()

    # Apply plateau cleaning
    df = remove_plateau_values_globally(df)

    # Identify NaN introduced by plateau removal
    new_nan_mask = df.isna() & ~original_nan_mask

    # Drop columns with too many NaN after plateau removal
    min_valid_points_plateaux = int(len(df) * 0.5)
    df = df.dropna(axis=1, thresh=min_valid_points_plateaux)

    # Keep only remaining columns in the mask
    new_nan_mask = new_nan_mask[df.columns]

    # Index cleaning
   # Use "timestamp" column as index if present otherwise rely on existing index
    if "timestamp" in df.columns:
        df.index = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.drop(columns=["timestamp"])
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")

    df.index = df.index.tz_localize(None)
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # Seuils automatiques
    def estimate_column_thresholds(data: pd.DataFrame) -> dict:
        """Estimate outlier detection thresholds for each numeric column."""
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

    # Remove outliers
    df_wo_outliers = df.drop(index=outlier_timestamps)
    min_valid_points = int(len(df_wo_outliers) * 0.9)
    df_wo_outliers = df_wo_outliers.dropna(axis=1, thresh=min_valid_points)

    # Frequency
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
        lambda seg: seg.interpolate(method="time", limit_direction="both")
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

    # Reinsert NaNs from outliers
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
    # Reinsert NaNs from inconsistent plateaus
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
    Applies ASWN normalization (with trend) to all numeric columns in a DataFrame.
    The window size is automatically computed from the median time delta.

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.
        min_std (float): Minimum allowed standard deviation.
        min_valid_ratio (float): Min ratio of valid (non-NaN) values within window.
        alpha (float): Trend blending coefficient (0-1).

    Returns:
        pd.DataFrame or None: Normalized DataFrame or None if no valid columns remain.
    """
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()

    if window_size is None:
        window_size = 50
    elif isinstance(window_size, str):
        window_size = int(pd.Timedelta(window_size) / df.index.to_series().diff().median())
    elif isinstance(window_size, (int, np.integer)):
        pass
    else:
        raise RuntimeError("Window size must be None, str or int")

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = aswn_with_trend(
                df[col], window_size, min_std, min_valid_ratio, alpha
            )
            # Preserve gaps; do not interpolate after normalization

    df.attrs["m"] = window_size
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    any_nan_mask = df.isna().any(axis=1)
    df.loc[any_nan_mask] = np.nan
    def has_enough_valid_windows(series, window_size, min_ratio):
        """Return True if a series has a sufficient ratio of valid windows."""
        if len(series) < window_size:
            return True

        # Count how many valid values in each window
        valid_counts = series.rolling(window_size).count()

        # A window is valid if it has exactly ``window_size`` non-NaN values
        valid_windows = (valid_counts == window_size).sum()

        total_windows = len(series) - window_size + 1
        ratio = valid_windows / max(1, total_windows)

        # print(f"[CHECK] {series.name} : {valid_windows} valid windows ({ratio:.1%})")

        return True if ratio >= min_ratio else False

    if not has_enough_valid_windows(df[df.columns[0]], window_size, 0.1):
        print("[INFO] No dimension has enough valid windows → returning input as is")
    return df



def define_m_using_clustering(
    df: pd.DataFrame,
    k: int = 3,
    window_sizes: list[str] = None,
    max_points: int = 5000,
    max_window_sizes: int = 5,
    max_segments: int = 2000,
    n_jobs: int = 8,
) -> list[tuple[int, str, float, float]]:
    """
    Determine suitable window sizes for motif extraction based on cluster
    stability and density using FAISS. This version performs variance based
    sampling of candidate windows.
    """
    freq = df.index.to_series().diff().median()
    if pd.isna(freq) or freq <= pd.Timedelta(0):
        freq = pd.Timedelta(seconds=1)

    if window_sizes is None:
        max_pts = max(len(df) // 4, 2)
        n_candidates = min(max_window_sizes * 5, max_pts)
        pts_candidates = np.unique(
            np.round(np.logspace(np.log10(2), np.log10(max_pts), num=n_candidates)).astype(int)
        )

        window_sizes = [str(p * freq) for p in pts_candidates]

    window_sizes_pts = []
    for ws in window_sizes:
        delta = pd.Timedelta(ws)
        pts = int(delta / freq) if delta >= freq else 0
        if 10 <= pts < max(len(df) // 4, 2):
            window_sizes_pts.append((pts, ws))

    if not window_sizes_pts:
        default_pts = max(2, len(df) // 2)
        ws_delta = default_pts * freq
        ws_str = pd.tseries.frequencies.to_offset(ws_delta).freqstr
        window_sizes_pts.append((default_pts, ws_str))

    if len(df) > max_points:
        idx = np.linspace(0, len(df) - 1, max_points, dtype=int)
        df = df.iloc[idx]

    def eval_window(values: np.ndarray, ws_pts: int, ws_str: str):
        """Evaluate a candidate window size for clustering stability."""
        if len(values) < ws_pts or ws_pts >= len(values):
            return None
        try:
            segs = np.lib.stride_tricks.sliding_window_view(values, ws_pts)
        except ValueError:
            return None
        if segs.shape[0] < 2:
            return None

        # Smart sampling by variance
        valid_counts = np.sum(~np.isnan(segs), axis=1)
        segs = segs[valid_counts >= 2]
        seg_var = np.nanstd(segs, axis=1, ddof=0)
        idx_sorted = np.argsort(-seg_var)  # descending
        if len(idx_sorted) > max_segments:
            segs = segs[idx_sorted[:max_segments]]
        else:
            segs = segs[idx_sorted]

        segs = segs[~np.isnan(segs).any(axis=1)]
        if len(segs) < 2:
            return None

        segs = (segs - segs.mean(axis=1, keepdims=True)) / (segs.std(axis=1, keepdims=True) + 1e-8)
        segs_f = segs.astype(np.float32)

        index = faiss.IndexFlatL2(ws_pts)
        index.add(segs_f)
        dists, _ = index.search(segs_f, 2)
        nn = dists[:, 1]
        stability = np.median(nn) / ws_pts
        density = (nn < stability * 1.5).sum() / len(nn)
        return (ws_pts, ws_str, stability, density)

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("No numeric columns available for clustering.")

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

    aggregated = []
    for ws_str, metrics in results_by_ws.items():
        if not metrics["stability"]:
            continue
        med_stab = np.median(metrics["stability"])
        mean_dens = np.mean(metrics["density"])
        aggregated.append((metrics["pts"], ws_str, med_stab, mean_dens))

    if not aggregated:
        raise ValueError("No window size evaluated successfully.")

    scores = {
        ws_str: np.mean(m["density"]) / np.median(m["stability"])
        for ws_str, m in results_by_ws.items() if m["stability"]
    }
    best_ws = max(window_sizes_pts, key=lambda tpl: scores.get(tpl[1], -np.inf))

    if len(window_sizes_pts) > max_window_sizes:
        candidates = [ws for ws in window_sizes_pts if ws != best_ws]
        sampled = random.sample(candidates, max_window_sizes - 1)
        window_sizes_pts = sampled + [best_ws]
        random.shuffle(window_sizes_pts)

    aggregated.sort(key=lambda x: x[2])
    final = aggregated[:k]
    return final

def cluster_dimensions(
    df: Union[pd.DataFrame, List[pd.DataFrame]],
    group_size: int = 5,
    top_k: int = 4,
    min_std: float = 1e-2,
    min_valid_ratio: float = 0.8,
    min_cluster_size: int = 2,
    mode: str = 'hybrid'  # 'motif', 'discord', 'hybrid'
) -> List[List[str]]:
    """
    Cluster DataFrame dimensions based on temporal coherence.
    Modes:
        - ``motif``   : homogeneous groups
        - ``discord`` : heterogeneous groups
        - ``hybrid``  : complementary groups
    The function also displays:
        - Average correlation between clusters and remaining columns
        - Average correlation between sensor families (by name)
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

        # Correlation matrix
        corr = base_df[valid_cols].corr()

        # Distance depending on the chosen mode
        if mode == 'motif':
            dist = 1 - corr.pow(2).abs()
        elif mode == 'discord':
            dist = 1 - corr.abs()
        elif mode == 'hybrid':
            dist = 1 - corr

        dist = dist.fillna(1.0)

        # Mean distance used for dynamic threshold
        tril_values = dist.where(np.tril(np.ones(dist.shape), -1).astype(bool)).stack()
        mean_dist = tril_values.mean()
        std_dist = tril_values.std()
        distance_threshold = mean_dist - 0.5 * std_dist

        # Hierarchical clustering
        model = AgglomerativeClustering(
            metric='precomputed',
            linkage='average',
            distance_threshold=distance_threshold,
            n_clusters=None
        )
        labels = model.fit_predict(dist.values)

        # Build clusters
        label_map = {}
        for col, label in zip(valid_cols, labels):
            label_map.setdefault(label, []).append(col)

        # Average coherence function used for sorting
        def avg_corr(cols):
            if len(cols) < 2:
                return 0
            matrix = base_df[cols].corr().abs()
            tril = matrix.where(np.tril(np.ones(matrix.shape), -1).astype(bool))
            return tril.stack().mean()

        # Sort clusters
        sorted_clusters = sorted(
            [c for c in label_map.values() if len(c) >= min_cluster_size],
            key=avg_corr,
            reverse=True
        )

        for cluster_cols in sorted_clusters:
            if len(clusters) >= top_k:
                break
            clusters.append(cluster_cols + ["timestamp"])

        # === Correlation between each cluster and the remaining columns ===
        print("\n[CLUSTERING]")
        print("\n Cross correlation between columns in each clusters :")

        for i, cluster in enumerate(clusters):
            cluster_vars = [col for col in cluster if col != "timestamp"]
            others = [col for col in base_df.columns if col not in cluster_vars]

            if not cluster_vars or not others:
                continue

            sub_corr = base_df[cluster_vars + others].corr().loc[cluster_vars, others]
            mean_cross_corr = sub_corr.abs().mean().mean()
            print(f"    Cluster {i+1:02d} ({len(cluster_vars)} variables) correlation = {mean_cross_corr:.3f}")

        # === Correlation between sensor families ===
        print("\n Correlation between parameters in sensors :")

        # Group by sensor family (before the trailing underscore)
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

    return clusters


def pre_processed(
    df: Union[pd.DataFrame, List[pd.DataFrame]],
    gap_multiplier: float = 15,
    min_std: float = 1e-2,
    min_valid_ratio: float = 0.8,
    alpha: float = 0.65,
    window_size: str = None,
    sort_by_variables: bool = True,
    cluster: bool = False,
    normalize: bool = True,
    mode: str = 'hybrid',
    top_k_cluster: int = 4,
) -> Union[pd.DataFrame, List[pd.DataFrame], List[List[pd.DataFrame]]]:
    """
    Full preprocessing pipeline for one or several DataFrames.
    - Interpolation & synchronization
    - Optional clustering
    - Optional normalization
    - Optional window detection
    """

    final_window_size = None

    if isinstance(df, list) and not all(isinstance(x, pd.DataFrame) for x in df):
        raise TypeError("df must be a DataFrame or a list of DataFrames")

    # === Cas d'une liste de DataFrames ===
    if isinstance(df, list):

        if cluster:
            synced_dfs = synchronize_on_common_grid(df, propagate_nan=False)
            clustered_groups = cluster_dimensions(synced_dfs, top_k=top_k_cluster, mode=mode)
            cluster_outputs = []
            group_result_normalize = []
            group_result = []

            for i, col_names in enumerate(sorted(clustered_groups)):
                cluster_df = synced_dfs.reset_index()[col_names].copy()
                clustered_df = interpolate(cluster_df, gap_multiplier=gap_multiplier)
                if clustered_df.empty:
                    continue
                if window_size is None:
                    try:
                        win_list = define_m_using_clustering(clustered_df)
                        final_window_size = win_list[0][1]
                    except ValueError:
                        final_window_size = max(2, len(clustered_df) // 2)
                else:
                    final_window_size = window_size
                     
                print(f"Cluster {i+1} [WINDOW SIZE] → {final_window_size}")

                if clustered_df is not None and not clustered_df.empty:
                    group_result.append(clustered_df)
                clustered_df_normalize = normalization(
                    clustered_df,
                    min_std=min_std,
                    min_valid_ratio=min_valid_ratio,
                    alpha=alpha,
                    window_size=final_window_size
                )
                           
                if clustered_df_normalize is not None and not clustered_df_normalize.empty:
                    group_result_normalize.append(clustered_df_normalize)

            return group_result, group_result_normalize

        df_interpolate = synchronize_on_common_grid(df, propagate_nan=True)

        if window_size is None:
            try:
                win_list = define_m_using_clustering(df_interpolate)
                final_window_size = win_list[0][1]
            except ValueError:
                final_window_size = max(2, len(df_interpolate) // 2)
        else:
            final_window_size = window_size

        print(f"[WINDOW SIZE] → {final_window_size}")

        df_normalize = normalization(
            df_single,
            min_std=min_std,
            min_valid_ratio=min_valid_ratio,
            alpha=alpha,
            window_size=final_window_size
        )

        return df_interpolate, df_normalize

    if cluster:

        # Step 1: interpolate without global NaN propagation
        interpolated_df = interpolate(df.copy(), gap_multiplier=gap_multiplier, propagate_nan=False)

        # Step 2: clustering to retrieve column names for each cluster
        clusters = cluster_dimensions(interpolated_df, top_k=top_k_cluster, mode=mode)

        cluster_outputs = []
        group_result = []
        group_result_normalize = []

        for i, col_names in enumerate(clusters):
            print(len(col_names))
            cluster_df = df[col_names].copy()

            clustered_df = interpolate(cluster_df, gap_multiplier=gap_multiplier)
            if clustered_df.empty:
                    continue
            if window_size is None:
                win_list = define_m_using_clustering(clustered_df)
                final_window_size = win_list[0][1]
            else:
                final_window_size = window_size

            print(f"Cluster {i+1} [WINDOW SIZE] → {final_window_size}")

            if clustered_df is not None and not clustered_df.empty:
                group_result.append(clustered_df)

            clustered_df_normalize = normalization(
                clustered_df,
                min_std=min_std,
                min_valid_ratio=min_valid_ratio,
                alpha=alpha,
                window_size=final_window_size
            )
                    
            if clustered_df_normalize is not None and not clustered_df_normalize.empty:
                group_result_normalize.append(clustered_df_normalize)

        return group_result, group_result_normalize

    # === Case without clustering ===
    interpolated_df = interpolate(df.copy(), gap_multiplier)

    if window_size is None:
        try:
            win_list = define_m_using_clustering(interpolated_df)
            final_window_size = win_list[0][1]
        except ValueError:
            final_window_size = max(2, len(interpolated_df) // 2)
    else:
        final_window_size = window_size

    print(f"[WINDOW SIZE] → {final_window_size}")

    interpolated_df_normalize = normalization(
        interpolated_df,
        min_std=min_std,
        min_valid_ratio=min_valid_ratio,
        alpha=alpha,
        window_size=final_window_size
    )

    return interpolated_df_normalize

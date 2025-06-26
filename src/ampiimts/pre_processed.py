"""Preprocessed module for panda DataFrame with timestamp column."""
from typing import Union, List
from collections import Counter
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
    df_origine: pd.DataFrame,
    gap_multiplier: float = 15,
    column_thresholds: dict = None,
    propagate_nan: bool = True,
) -> pd.DataFrame:
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

    # Sauvegarder les NaN d'origine
    original_nan_mask = df.isna()

    # Appliquer le nettoyage par plateaux
    df = remove_plateau_values_globally(df)

    # Identifier les NaN ajoutés par les plateaux incohérents
    new_nan_mask = df.isna() & ~original_nan_mask

    # Supprimer les colonnes avec trop de NaN après suppression des plateaux
    min_valid_points_plateaux = int(len(df) * 0.9)
    df = df.dropna(axis=1, thresh=min_valid_points_plateaux)

    # Ne garder que les colonnes restantes dans le masque
    new_nan_mask = new_nan_mask[df.columns]

    # Nettoyage index
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

    # Outliers extrêmes
    outlier_timestamps = set()
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col]
        threshold = column_thresholds.get(col, np.inf)
        center = series.median()
        mask = np.abs(series - center) > threshold
        outlier_timestamps.update(df.index[mask])

    # Nettoyage outliers
    df_wo_outliers = df.drop(index=outlier_timestamps)
    min_valid_points = int(len(df_wo_outliers) * 0.9)
    df_wo_outliers = df_wo_outliers.dropna(axis=1, thresh=min_valid_points)

    # Fréquence
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

    # Réinjecter les NaN des outliers
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

    # Réinjecter les NaN des plateaux incohérents
    if propagate_nan:
        plateau_nan_locs = []
        new_nan_mask = new_nan_mask.reindex(df.index)
        for col in new_nan_mask.columns:
            if col == "timestamp":
                continue 
            mask = df[col].copy()
            mask = mask.fillna(False)
            mask = mask.infer_objects(copy=False)
            mask = mask.astype(bool)
            plateau_nan_locs.extend(df.index[mask])
        plateau_nan_locs = sorted(set(ts for ts in plateau_nan_locs if ts in df_out.index))
        df_out.loc[plateau_nan_locs, :] = np.nan
    else:
        new_nan_mask = new_nan_mask.reindex(df.index)
        for col in new_nan_mask.columns:
            if col == "timestamp":
                continue 
            mask = df[col].copy()
            mask = mask.fillna(False)
            mask = mask.infer_objects(copy=False)
            mask = mask.astype(bool)
            ts_nan = df.index[mask]
            ts_nan = [ts for ts in ts_nan if ts in df_out.index]
            df_out.loc[ts_nan, col] = np.nan
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
            df[col] = df[col].interpolate(method="linear", limit=3, limit_direction="both")

    df.attrs["m"] = window_size
    df = df.loc[:, df.notna().sum() >= window_size]  # Au moins la taille d'une fenêtre
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    any_nan_mask = df.isna().any(axis=1)
    df.loc[any_nan_mask] = np.nan
    def has_enough_valid_windows(series, window_size, min_ratio):
        if len(series) < window_size:
            return False

        # Calcule combien de valeurs valides dans chaque fenêtre
        valid_counts = series.rolling(window_size).count()

        # Une fenêtre est valide si elle a exactement window_size valeurs non-NaN
        valid_windows = (valid_counts == window_size).sum()

        total_windows = len(series) - window_size + 1
        ratio = valid_windows / max(1, total_windows)

        print(f"[CHECK] {series.name} : {valid_windows} fenêtres valides ({ratio:.1%})")

        return True if ratio >= min_ratio else False

    if not has_enough_valid_windows(df[df.columns[0]], window_size, 0.1):
        print("[INFO] Aucune dimension avec suffisamment de fenêtres valides. → retour None")
        return None
    else:
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
    Détermine les meilleures tailles de fenêtre pour l'extraction de motifs
    en se basant sur la stabilité et la densité de clusters via FAISS.

    Cette version intègre un échantillonnage intelligent des fenêtres par variance.
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
        try:
            segs = np.lib.stride_tricks.sliding_window_view(values, ws_pts)
        except ValueError:
            return None
        if segs.shape[0] < 2:
            return None

        # Échantillonnage intelligent par variance
        valid_counts = np.sum(~np.isnan(segs), axis=1)
        segs = segs[valid_counts >= 2]
        seg_var = np.nanstd(segs, axis=1, ddof=0)
        idx_sorted = np.argsort(-seg_var)  # décroissant
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

from typing import Union, List
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import re

def cluster_dimensions(
    df: Union[pd.DataFrame, List[pd.DataFrame]],
    group_size: int = 5,
    top_k: int = 5,
    min_std: float = 1e-2,
    min_valid_ratio: float = 0.8,
    min_cluster_size: int = 2,
    mode: str = 'hybrid'  # 'motif', 'discord', 'hybrid'
) -> List[List[str]]:
    """
    Cluster les dimensions d’un DataFrame selon leur cohérence temporelle.
    Trois modes : 'motif' (groupes homogènes), 'discord' (groupes variés),
    ou 'hybrid' (groupes complémentaires).
    Affiche aussi :
      - La corrélation moyenne entre clusters et le reste des colonnes
      - La corrélation moyenne entre familles de capteurs (par nom)
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

        # Matrice de corrélation
        corr = base_df[valid_cols].corr()

        # Distance en fonction du mode choisi
        if mode == 'motif':
            dist = 1 - corr.pow(2).abs()
        elif mode == 'discord':
            dist = 1 - corr.abs()
        elif mode == 'hybrid':
            dist = 1 - corr

        dist = dist.fillna(1.0)

        # Distance moyenne pour seuil dynamique
        tril_values = dist.where(np.tril(np.ones(dist.shape), -1).astype(bool)).stack()
        mean_dist = tril_values.mean()
        std_dist = tril_values.std()
        distance_threshold = mean_dist - 0.5 * std_dist

        # Clustering hiérarchique
        model = AgglomerativeClustering(
            metric='precomputed',
            linkage='average',
            distance_threshold=distance_threshold,
            n_clusters=None
        )
        labels = model.fit_predict(dist.values)

        # Construction des clusters
        label_map = {}
        for col, label in zip(valid_cols, labels):
            label_map.setdefault(label, []).append(col)

        # Fonction de cohérence moyenne (pour tri)
        def avg_corr(cols):
            if len(cols) < 2:
                return 0
            matrix = base_df[cols].corr().abs()
            tril = matrix.where(np.tril(np.ones(matrix.shape), -1).astype(bool))
            return tril.stack().mean()

        # Tri des clusters
        sorted_clusters = sorted(
            [c for c in label_map.values() if len(c) >= min_cluster_size],
            key=avg_corr,
            reverse=True
        )

        for cluster_cols in sorted_clusters:
            if len(clusters) >= top_k:
                break
            clusters.append(cluster_cols + ["timestamp"])

        # === Corrélation entre chaque cluster et le reste ===
        print("\n[Corrélation croisée entre clusters et autres colonnes :]")

        for i, cluster in enumerate(clusters):
            cluster_vars = [col for col in cluster if col != "timestamp"]
            others = [col for col in base_df.columns if col not in cluster_vars]

            if not cluster_vars or not others:
                continue

            sub_corr = base_df[cluster_vars + others].corr().loc[cluster_vars, others]
            mean_cross_corr = sub_corr.abs().mean().mean()
            print(f"  ↪ Cluster {i+1:02d} ({len(cluster_vars)} variables) ↔ autres : corr moyenne = {mean_cross_corr:.3f}")

        # === Corrélation entre familles de capteurs ===
        print("\n[Corrélation entre familles de capteurs :]")

        # Regroupement par famille (avant l’underscore final)
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

        print(family_corr.round(2))

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
) -> Union[pd.DataFrame, List[pd.DataFrame], List[List[pd.DataFrame]]]:
    """
    Full preprocessing pipeline for one or several DataFrames.
    - Interpolation & synchronisation
    - Clustering (optionnel)
    - Normalisation (optionnelle)
    - Détection de fenêtre (optionnelle)
    """

    final_window_size = None

    # === Cas d'une liste de DataFrames ===
    if isinstance(df, list):
        synced_dfs = synchronize_on_common_grid(df, sort_by_variables=sort_by_variables)

        if cluster:
            clustered_groups = cluster_dimensions(synced_dfs)
            cluster_outputs = []

            for group in sorted(clustered_groups, reverse=True):
                group_result = []

                for clustered_df in group:
                    if window_size is None:
                        win_list = define_m_using_clustering(clustered_df)
                        final_window_size = win_list[0][1]
                        print(f"[WINDOW LIST - Cluster] Fenêtre → {final_window_size}")
                    else:
                        final_window_size = window_size
                        print(f"[WINDOW LIST - Cluster] Fenêtre utilisateur → {final_window_size}")

                    if normalize:
                        clustered_df = normalization(
                            clustered_df,
                            min_std=min_std,
                            min_valid_ratio=min_valid_ratio,
                            alpha=alpha,
                            window_size=final_window_size
                        )
                    group_result.append(clustered_df)

                # Garder seulement les groupes complets
                group_result = [df_ for df_ in group_result if df_ is not None and not df_.empty]
                if len(group_result) == len(group):
                    cluster_outputs.extend(group_result)  # ✅ Ajoute chaque cluster individuellement

            return cluster_outputs

        else:
            if window_size is None:
                win_list = define_m_using_clustering(synced_dfs[0])
                final_window_size = win_list[0][1]
                print(f"[WINDOW LIST] Fenêtre retenue → {final_window_size}")
            else:
                final_window_size = window_size
                print(f"[WINDOW LIST] Fenêtre utilisateur → {final_window_size}")

            if normalize:
                return [
                    normalization(
                        df_single,
                        min_std=min_std,
                        min_valid_ratio=min_valid_ratio,
                        alpha=alpha,
                        window_size=final_window_size
                    ) for df_single in synced_dfs
                ]
            else:
                return synced_dfs

    if cluster:

        # Étape 1 : interpolation sans propagation globale de NaN
        interpolated_df = interpolate(df.copy(), gap_multiplier=gap_multiplier, propagate_nan=False)

        # 2. Étape : clustering → tu récupères les noms de colonnes de chaque cluster
        clusters = cluster_dimensions(interpolated_df)

        cluster_outputs = []
        group_result = []
        group_result_normalize = []

        for i, col_names in enumerate(clusters):
            print(len(col_names))
            cluster_df = df[col_names].copy()

            clustered_df = interpolate(cluster_df, gap_multiplier=gap_multiplier)
            if window_size is None:
                win_list = define_m_using_clustering(clustered_df)
                final_window_size = win_list[0][1]
                print(f"[WINDOW] Fenêtre retenue → {final_window_size}")
            else:
                final_window_size = window_size
           
            
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

    # === Cas sans clustering ===
    interpolated_df = interpolate(df.copy(), gap_multiplier)

    if window_size is None:
        win_list = define_m_using_clustering(interpolated_df)
        final_window_size = win_list[0][1]
        print(f"[WINDOW] Fenêtre retenue → {final_window_size}")
    else:
        final_window_size = window_size
        print(f"[WINDOW] Fenêtre utilisateur → {final_window_size}")


    interpolated_df_normalize = normalization(
        interpolated_df,
        min_std=min_std,
        min_valid_ratio=min_valid_ratio,
        alpha=alpha,
        window_size=final_window_size
    )

    return interpolated_df, interpolated_df_normalize

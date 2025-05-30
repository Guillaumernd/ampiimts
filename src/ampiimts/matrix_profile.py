from typing import Optional, Union, List
from tslearn.metrics import dtw_path, dtw
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import stumpy as sp
import faiss
import matplotlib.pyplot as plt


def batch_process(df_list, window_size, n_jobs=4, batch_size=4):
    results = []
    # On traite batch_size dataframes à la fois
    for i in range(0, len(df_list), batch_size):
        batch = df_list[i:i+batch_size]
        batch_result = Parallel(n_jobs=n_jobs)(
            delayed(matrix_profile_process)(df, window_size=window_size)
            for df in batch
        )
        results.extend(batch_result)
    return results


def multi_aamp_with_nan_parallel(
    df: pd.DataFrame,
    window_size: int,
    n_jobs: int = 4
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

    def batch_col_parallel(X, window_size, n_jobs=4, batch_size=4):
        k = X.shape[1]
        results = []

        
        def compute_aamp(col):
            try:
                arr = X[:, col]
                if np.all(np.isnan(arr)) or arr.shape[0] < window_size:
                    print(f"[SKIP] Col {col}: full NaN or too short.")
                    return (np.full(profile_len, np.nan), np.full(profile_len, -1))
                aamp_result = sp.aamp(arr, window_size)
                return aamp_result[:, 0], aamp_result[:, 1]
            except Exception as e:
                print(f"[ERROR] compute_aamp failed on col={col}, shape={arr.shape}, err={e}")
                return (np.full(profile_len, np.nan), np.full(profile_len, -1))
        for i in range(0, k, batch_size):
            batch_cols = range(i, min(i+batch_size, k))
            batch_result = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(compute_aamp)(col) for col in batch_cols
            )
            results.extend(batch_result)
        return results

    
    results = batch_col_parallel(
        X, window_size, n_jobs=n_jobs, batch_size=2)

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

    # Replace df_normalize and df_profile with your own preprocessed DataFrames
    result = discover_patterns_faiss_kmeans_with_discords(
        df_o, df_profile, window_size)
    return result


def matrix_profile(
    df_o: Union[pd.DataFrame, List[pd.DataFrame]],
    window_size: Optional[int] = None,
    n_jobs: int = 4
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
        matrix_profiles = batch_process(
            df_o, window_size, n_jobs=n_jobs, batch_size=n_jobs)

        return matrix_profiles
    return matrix_profile_process(df_o, window_size=window_size)

def select_core_motifs(
    cluster_indices: np.ndarray,
    df: pd.DataFrame,
    window_size: int,
    top_n: int = 5
) -> np.ndarray:
    """
    Selects the top_n motifs closest to the mean pattern within a cluster, 
    avoiding overlaps.

    Args:
        cluster_indices (np.ndarray): Indices of motif windows in the cluster.
        df (pd.DataFrame): DataFrame containing the time series with 'value' column.
        window_size (int): Size of the motif window.
        top_n (int): Number of core motifs to select.

    Returns:
        np.ndarray: Indices of the selected non-overlapping core motifs.
    """
    # Extract all motif segments from the cluster indices
    motif_segments = np.array([
        df['value'].values[i:i + window_size] for i in cluster_indices
    ])
    # Compute the mean motif for the cluster
    center = motif_segments.mean(axis=0)
    # Compute the DTW distance to the cluster mean for each segment
    distances = [dtw(center, seg) for seg in motif_segments]
    idxs_sorted = np.argsort(distances)
    selected = []
    for idx in idxs_sorted:
        idx_val = cluster_indices[idx]
        # Check for overlaps with already selected motifs
        overlap = any(abs(idx_val - s) < window_size for s in selected)
        if not overlap:
            selected.append(idx_val)
        if len(selected) == top_n:
            break
    return np.array(selected)


def extract_all_windows(
    df: pd.DataFrame,
    window_size: int
) -> np.ndarray:
    """
    Extracts all sliding windows of fixed size from the series.

    Args:
        df (pd.DataFrame): DataFrame with a 'value' column.
        window_size (int): Length of the sliding window.

    Returns:
        np.ndarray: 2D array with shape (n_windows, window_size).
    """
    values = df['value'].values
    n = len(values) - window_size + 1
    return np.array([values[i:i + window_size] for i in range(n)])


def find_motif_matches(
    df: pd.DataFrame,
    ref_pattern: np.ndarray,
    window_size: int,
    dist_func: str = 'dtw',
    dist_thresh: float = 6
) -> list:
    """
    Finds all motif matches in the series, within a DTW or L2 distance threshold.

    Args:
        df (pd.DataFrame): DataFrame with 'value' column.
        ref_pattern (np.ndarray): Reference motif pattern.
        window_size (int): Window size for motif.
        dist_func (str): Distance function, either 'dtw' or 'l2'.
        dist_thresh (float): Distance threshold for matching.

    Returns:
        list: List of start indices for detected motif matches.
    """
    values = df['value'].values
    n = len(values) - window_size + 1
    motif_idxs = []
    for i in range(n):
        window = values[i:i + window_size]
        if len(window) < window_size or not np.all(np.isfinite(window)):
            continue  
        if ref_pattern is None or len(ref_pattern) != window_size or not np.all(np.isfinite(ref_pattern)):
            continue 
        if dist_func == 'dtw':
            dist = dtw(ref_pattern, window)
        elif dist_func == 'l2':
            dist = np.linalg.norm(ref_pattern - window)
        else:
            raise ValueError("dist_func must be 'dtw' or 'l2'")
        if dist <= dist_thresh:
            motif_idxs.append(i)
    motif_idxs = filter_non_overlapping_idxs(motif_idxs, window_size)
    return motif_idxs



def exclude_discord_windows(
    windows: np.ndarray,
    df_profile: pd.DataFrame,
    window_size: int,
    top_discord_k: int = 0.02,
    discord_margin: int = None
):
    """
    Excludes windows around discord locations.

    Args:
        windows (np.ndarray): All sliding windows of the time series.
        df_profile (pd.DataFrame): Matrix profile with anomaly scores.
        window_size (int): Length of motif window.
        top_discord_k (int): Number of top discords to exclude.
        discord_margin (int, optional): Margin around discord to mask (default: 2 * window_size).

    Returns:
        tuple: (filtered windows, boolean mask, discord indices)
    """
    if discord_margin is None:
        discord_margin = 2 * window_size
    top_discord_k = max(1, int(len(df_profile) * top_discord_k))
    discord_idxs = np.argsort(df_profile['value'].values)[-top_discord_k:][::-1]
    mask = np.ones(len(windows), dtype=bool)
    for d_idx in discord_idxs:
        start = max(0, d_idx - discord_margin)
        end = min(len(windows), d_idx + discord_margin + 1)
        mask[start:end] = False
    return windows[mask], mask, discord_idxs


def filter_non_overlapping_idxs(
    motif_idxs: list,
    window_size: int
) -> list:
    """
    Removes overlapping motif indices.

    Args:
        motif_idxs (list): Motif start indices.
        window_size (int): Motif window size.

    Returns:
        list: Filtered, non-overlapping motif indices.
    """
    motif_idxs_sorted = sorted(set(motif_idxs))
    filtered = []
    last_end = -1
    for idx in motif_idxs_sorted:
        if idx > last_end:
            filtered.append(idx)
            last_end = idx + window_size - 1
    return filtered


def align_motifs_dtw(
    motif_segments: np.ndarray
) -> np.ndarray:
    """
    Aligns motifs to the first reference motif using DTW path.

    Args:
        motif_segments (np.ndarray): Motif segments to align.

    Returns:
        np.ndarray: Array of DTW-aligned motifs.
    """
    ref = motif_segments[0]
    aligned_segments = [ref]
    for motif in motif_segments[1:]:
        path, _ = dtw_path(ref, motif)
        aligned = []
        for i in range(len(ref)):
            js = [j for ii, j in path if ii == i]
            aligned.append(np.mean([motif[j] for j in js]))
        aligned_segments.append(aligned)
    return np.array(aligned_segments)


def plot_superposed_aligned_motifs_dtw(
    df: pd.DataFrame,
    motif_idxs: list,
    window_size: int,
    title: str = "Aligned Superposed Motifs (DTW)"
):
    """
    Plots aligned and superposed motifs using DTW alignment.

    Args:
        df (pd.DataFrame): DataFrame with 'value' column.
        motif_idxs (list): List of motif start indices.
        window_size (int): Motif window size.
        title (str): Plot title.
    """
    values = df['value'].values
    motif_segments = []
    for idx in motif_idxs:
        start = idx
        end = idx + window_size
        if end > len(values):
            continue
        motif_segments.append(values[start:end])
    if len(motif_segments) == 0:
        print("No motifs for this cluster.")
        return
    motif_segments = np.array(motif_segments)
    aligned = align_motifs_dtw(motif_segments)
    t = np.arange(window_size)
    plt.figure(figsize=(10, 5))
    for i, seg in enumerate(aligned):
        plt.plot(t, seg, alpha=0.5, lw=2, label=f"Motif {i+1}" if i < 10 else None)
    if len(aligned) > 1:
        plt.plot(t, aligned.mean(axis=0), color='black', linewidth=3, label="Mean motif")
        plt.fill_between(
            t,
            aligned.mean(axis=0) - aligned.std(axis=0),
            aligned.mean(axis=0) + aligned.std(axis=0),
            color='gray', alpha=0.2, label='Std. dev.'
        )
    plt.title(title)
    plt.xlabel("Step in motif window (DTW-aligned)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_motif_locations_on_series(
    df: pd.DataFrame,
    motif_idxs: list,
    window_size: int,
    title: str = "Motif Locations on Series",
    color: str = "green",
    marker: str = "o"
):
    """
    Plots motif segments on the original time series.

    Args:
        df (pd.DataFrame): DataFrame with 'value' column.
        motif_idxs (list): Motif start indices.
        window_size (int): Motif window size.
        title (str): Plot title.
        color (str): Color for motif segments.
        marker (str): Marker for segment starts.
    """
    values = df['value'].values
    t = df.index
    plt.figure(figsize=(15, 6))
    plt.plot(t, values, color='black', label='Time series')
    for i, idx in enumerate(motif_idxs):
        start = idx
        end = idx + window_size
        if end > len(values):
            continue
        plt.plot(
            t[start:end], values[start:end], lw=3, alpha=0.7,
            color=color, label=f"Motif {i+1}" if i < 8 else None
        )
        plt.scatter(t[start], values[start], s=80, marker=marker, color=color)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_cluster_indices(
    labels: np.ndarray,
    keep_mask: np.ndarray,
    window_size: int,
    min_size: int = 2
) -> dict:
    """
    Extracts non-overlapping motif indices per cluster.

    Args:
        labels (np.ndarray): Cluster labels for each window.
        keep_mask (np.ndarray): Boolean mask of kept windows.
        window_size (int): Motif window size.
        min_size (int): Minimum cluster size.

    Returns:
        dict: Dictionary mapping cluster label to motif indices.
    """
    clusters = {}
    real_idxs = np.where(keep_mask)[0]
    for label in np.unique(labels):
        idxs = real_idxs[labels == label]
        idxs_filtered = filter_non_overlapping_idxs(idxs, window_size)
        if len(idxs_filtered) >= min_size:
            clusters[label] = idxs_filtered
    return clusters


def faiss_kmeans(
    windows: np.ndarray,
    n_clusters: int,
    n_iter: int = 20,
    verbose: bool = True
) -> np.ndarray:
    """
    Clusters windows using FAISS KMeans.

    Args:
        windows (np.ndarray): Windows to cluster (2D array).
        n_clusters (int): Number of clusters.
        n_iter (int): Number of KMeans iterations.
        verbose (bool): Verbosity flag.

    Returns:
        np.ndarray: Cluster labels for each window.
    """
    d = windows.shape[1]
    windows32 = windows.astype(np.float32)
    kmeans = faiss.Kmeans(d, n_clusters, niter=n_iter, verbose=verbose, gpu=False)
    kmeans.train(windows32)
    _, labels = kmeans.index.search(windows32, 1)
    return labels.ravel()


def fuse_similar_clusters(
    clusters: dict,
    df: pd.DataFrame,
    window_size: int,
    dtw_thresh: float = 10.
) -> dict:
    """
    Fuses clusters whose mean motifs are close according to DTW.

    Args:
        clusters (dict): Mapping of cluster labels to motif indices.
        df (pd.DataFrame): DataFrame with 'value' column.
        window_size (int): Motif window size.
        dtw_thresh (float): DTW threshold for cluster fusion.

    Returns:
        dict: New mapping of (possibly merged) clusters to indices.
    """
    cluster_means = []
    labels = []
    for label, idxs in clusters.items():
        motif_segments = np.array([df['value'].values[i:i + window_size] for i in idxs])
        mean_motif = motif_segments.mean(axis=0)
        cluster_means.append(mean_motif)
        labels.append(label)
    n = len(cluster_means)
    groups = []
    used = set()
    for i in range(n):
        if i in used:
            continue
        group = [labels[i]]
        for j in range(i + 1, n):
            if j in used:
                continue
            if dtw(cluster_means[i], cluster_means[j]) < dtw_thresh:
                group.append(labels[j])
                used.add(j)
        used.add(i)
        groups.append(group)
    fused_clusters = {}
    for group in groups:
        fused = []
        for label in group:
            fused.extend(clusters[label])
        fused_clusters[tuple(group)] = fused
    return fused_clusters

def extract_all_windows(df: pd.DataFrame, window_size: int):
    """
    Extract all sliding windows of fixed size from the 'value' column of a DataFrame.
    Returns an array of windows (shape: n_windows, window_size), all strictly of length window_size.
    """
    values = df['value'].values
    n = len(values)
    windows = []
    indices = []
    for i in range(n - window_size + 1):
        window = values[i:i+window_size]
        # Check window length and only keep if correct size
        if len(window) == window_size:
            windows.append(window)
            indices.append(i)
    windows = np.array(windows)  # This will be 2D (n_windows, window_size)
    indices = np.array(indices)
    return windows, indices


def get_valid_segments(df, idxs, window_size):
    """
    Returns only segments that are of correct length and all finite (no NaN/Inf).
    """
    values = df['value'].values
    valid_segments = []
    valid_idxs = []
    for i in idxs:
        seg = values[i:i+window_size]
        # Check for correct length and no NaN
        if len(seg) == window_size and np.all(np.isfinite(seg)):
            valid_segments.append(seg)
            valid_idxs.append(i)
    return np.array(valid_segments), np.array(valid_idxs)

def get_valid_indices(idxs, window_size, series_length):
    """
    Returns only indices i such that i+window_size <= series_length.
    Args:
        idxs (array-like): List/array of start indices.
        window_size (int): Length of window.
        series_length (int): Length of the series (len(df)).
    Returns:
        list: List of valid indices.
    """
    return [int(i) for i in idxs if i + window_size <= series_length]


def plot_detected_patterns_on_series(
    df: pd.DataFrame,
    patterns: list,
    all_idxs: list,
    window_size: int,
    pattern_labels: list = None,
    colors: list = None
):
    """
    Plots all detected patterns and their occurrences on the original time series.

    Args:
        df (pd.DataFrame): DataFrame with 'value' column.
        patterns (list): List of reference patterns (unused, for future use).
        all_idxs (list): List of lists of motif indices (per pattern).
        window_size (int): Motif window size.
        pattern_labels (list, optional): Labels for each pattern.
        colors (list, optional): List of colors for plotting.
    """
    t = df.index
    values = df['value'].values
    plt.figure(figsize=(15, 6))
    plt.plot(t, values, color='black', label='Time series')
    if colors is None:
        colors = plt.cm.tab10.colors
    for k, motif_idxs in enumerate(all_idxs):
        color = colors[k % len(colors)]
        label = (
            f"Pattern {pattern_labels[k]}"
            if pattern_labels is not None else f"Pattern {k+1}"
        )
        for i, idx in enumerate(motif_idxs):
            start, end = idx, idx + window_size
            if end > len(values):
                continue
            plt.plot(
                t[start:end], values[start:end], lw=3, color=color, alpha=0.7,
                label=label if i == 0 else None
            )
            plt.scatter(t[start], values[start], s=80, marker='o', color=color)
    plt.title("Detected Patterns in Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

def map_indices_safe(indices, kept_indices):
    """
    Map indices to kept_indices, but only keep those strictly < len(kept_indices).
    """
    indices = np.array(indices)
    mask = indices < len(kept_indices)
    return kept_indices[indices[mask]]

def plot_superposed_motifs_per_pattern(df, pattern_idxs_list, window_size, pattern_labels=None):
    """
    For each pattern, superimpose all motif segments on a separate plot.
    """
    values = df['value'].values
    for k, motif_idxs in enumerate(pattern_idxs_list):
        plt.figure(figsize=(8, 4))
        for idx in motif_idxs:
            seg = values[idx:idx+window_size]
            if len(seg) == window_size and np.all(np.isfinite(seg)):
                plt.plot(np.arange(window_size), seg, color='tab:blue', alpha=0.2)
        # Overlay the mean motif
        if len(motif_idxs) > 0:
            motifs = np.array([
                values[i:i+window_size]
                for i in motif_idxs
                if len(values[i:i+window_size]) == window_size and np.all(np.isfinite(values[i:i+window_size]))
            ])
            if len(motifs) > 0:
                plt.plot(np.arange(window_size), motifs.mean(axis=0), color='black', lw=2, label='Mean motif')
        plt.title(f"Superposed motifs for pattern {pattern_labels[k] if pattern_labels else k+1}")
        plt.xlabel("Motif step")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_all_matched_motifs_aligned_dtw(df, motif_idxs, window_size, pattern_ref=None, title="Matched Motifs (DTW Aligned)"):
    """
    Affiche tous les segments du signal matchés avec le motif de référence,
    tous réalignés par DTW sur le pattern_ref (ou le premier motif si None).
    """
    values = df['value'].values
    motif_segments = [values[idx:idx+window_size] for idx in motif_idxs if idx+window_size <= len(values)]
    if len(motif_segments) == 0:
        print("No motifs for this cluster.")
        return
    if pattern_ref is None:
        pattern_ref = motif_segments[0]
    # On réaligne tous les motifs sur le pattern_ref
    aligned = align_motifs_dtw_with_ref(motif_segments, pattern_ref)
    t = np.arange(window_size)
    plt.figure(figsize=(10, 5))
    for i, seg in enumerate(aligned):
        plt.plot(t, seg, alpha=0.3, lw=2, color='tab:blue')
    plt.plot(t, pattern_ref, color='black', lw=3, label='Pattern ref (medoid)')
    if len(aligned) > 1:
        plt.plot(t, np.mean(aligned, axis=0), color='red', lw=2, label="Mean motif (aligned)")
    plt.title(title)
    plt.xlabel("Motif step (DTW aligned)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

def align_motifs_dtw_with_ref(motif_segments, ref):
    """
    Aligne chaque segment avec le motif de référence 'ref' via DTW.
    """
    aligned_segments = []
    for motif in motif_segments:
        path, _ = dtw_path(ref, motif)
        aligned = []
        for i in range(len(ref)):
            js = [j for ii, j in path if ii == i]
            aligned.append(np.mean([motif[j] for j in js]))
        aligned_segments.append(aligned)
    return np.array(aligned_segments)



def discover_patterns_faiss_kmeans_with_discords(
    df: pd.DataFrame,
    df_profile: pd.DataFrame,
    window_size: int,
    n_clusters: int = 10,
    min_cluster_size: int = 2,
    top_discord_k: int = 0.02,
    scale_windows: bool = False,
    dtw_fusion_thresh: float = 10,
    core_top_n: int = 5,
    dist_thresh: float = 100,
    min_final_matches: int = 5
) -> dict:
    """
    Robust pipeline for motif and discord (anomaly) detection using FAISS KMeans.
    All motif/segment operations are NaN-safe.

    Returns:
        dict: Patterns, labels, occurrences and discords.
    """
    print("Extracting all sliding windows...")
    windows, window_indices = extract_all_windows(df, window_size)
    valid_mask = np.all(np.isfinite(windows), axis=1)
    windows = windows[valid_mask]
    window_indices = window_indices[valid_mask]
    df_profile_valid = df_profile.iloc[window_indices].reset_index(drop=True)

    print("Excluding discord windows...")
    windows_filtered, keep_mask, discord_idxs = exclude_discord_windows(
        windows, df_profile_valid, window_size, top_discord_k=top_discord_k
    )
    kept_indices = window_indices[keep_mask]

    if scale_windows:
        scaler = StandardScaler()
        windows_filtered = scaler.fit_transform(windows_filtered)

    print("Clustering windows with FAISS KMeans (Euclidean distance)...")
    labels = faiss_kmeans(
        windows_filtered, n_clusters=n_clusters, n_iter=25, verbose=True
    )

    clusters = get_cluster_indices(
        labels, keep_mask, window_size, min_size=min_cluster_size
    )

    clusters_true = {
        label: map_indices_safe(idxs, kept_indices)
        for label, idxs in clusters.items()
    }

    fused_clusters = fuse_similar_clusters(
        clusters_true, df, window_size, dtw_thresh=dtw_fusion_thresh
    )
    print(f"{len(fused_clusters)} patterns detected (excluding discords).")

    selected_refs = []
    selected_labels = []
    pattern_idxs_list = []

    for i, (label, idxs) in enumerate(fused_clusters.items()):
        idxs = get_valid_indices(idxs, window_size, len(df))
        motif_segments, valid_core_idxs = get_valid_segments(df, idxs, window_size)
        if len(motif_segments) < core_top_n:
            continue  # skip clusters with too few valid motifs

        # Récupérer les core motifs réels
        core_idxs = valid_core_idxs[:core_top_n]
        core_segments, _ = get_valid_segments(df, core_idxs, window_size)
        if len(core_segments) == 0:
            continue

        # --- Matching sur tous les core motifs (union filtrée) ---
        all_found_idxs = []
        for core in core_segments:
            idxs_found = find_motif_matches(df, core, window_size, dist_func='dtw', dist_thresh=dist_thresh)
            all_found_idxs.extend(idxs_found)
        motif_idxs = filter_non_overlapping_idxs(all_found_idxs, window_size)

        # Pour l'affichage : utiliser le médaïde comme "pattern_ref"
        mean_core = core_segments.mean(axis=0)
        dists = [dtw(mean_core, seg) for seg in core_segments]
        medoid_idx = np.argmin(dists)
        pattern_ref = core_segments[medoid_idx]

        # Tu gardes les segments trouvés (motif_idxs), le motif de ref, le label
        if len(motif_idxs) < min_final_matches:
            continue
        selected_refs.append(pattern_ref)
        selected_labels.append(label)
        pattern_idxs_list.append(motif_idxs)
        print(f"Pattern {i} {label}: {len(motif_idxs)} matches (after selection)")

            
        motif_idxs = find_motif_matches(
            df, pattern_ref, window_size, dist_func='dtw', dist_thresh=dist_thresh
        )
        _, motif_idxs_valid = get_valid_segments(df, motif_idxs, window_size)
        motif_idxs = motif_idxs_valid

        # *** Affichage supplémentaire : tous les motifs identifiés pour ce pattern ***
        plot_all_matched_motifs_aligned_dtw(
            df, motif_idxs, window_size, pattern_ref=pattern_ref,
            title=f"All motifs matched, DTW aligned (Pattern {i} {label})"
            )

        # --- Affichage : superposer les core motifs, puis tous les segments trouvés ---
        plot_superposed_aligned_motifs_dtw(
            df, core_idxs, window_size, title=f"Pattern {i} {label} (core motifs, DTW aligned)"
        )
        plot_motif_locations_on_series(
            df, motif_idxs, window_size, title=f"Pattern {i} {label} locations (all found motifs)"
        )
        

        

    discord_idxs = discord_idxs[discord_idxs < len(window_indices)]
    safe_discord_idxs = get_valid_indices(window_indices[discord_idxs], window_size, len(df))

    print("\nDetected discord(s) (anomalies):")
    plot_motif_locations_on_series(
        df, safe_discord_idxs, window_size,
        title="Discord Locations (Anomalies)", color="red", marker="X"
    )
    plot_detected_patterns_on_series(
        df, selected_refs, pattern_idxs_list, window_size, pattern_labels=selected_labels
    )

    return {
        'df_profile': df_profile,
        'core_patterns': selected_refs,
        'pattern_labels': selected_labels,
        'pattern_occurrences': pattern_idxs_list,
        'discord_idxs': safe_discord_idxs,
    }
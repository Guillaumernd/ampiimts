from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from tslearn.metrics import dtw_path, dtw
from sklearn.preprocessing import StandardScaler
import faiss

# ====== 1. ALIGNEMENT DTW "PURE" ======

def align_segments_to_reference(segments: np.ndarray, reference: np.ndarray = None) -> np.ndarray:
    """Aligne chaque segment sur une référence (par défaut le 1er), en utilisant DTW."""
    if reference is None:
        reference = segments[0]
    aligned_segments = []
    for motif in segments:
        path, _ = dtw_path(reference, motif)
        aligned = []
        for i in range(len(reference)):
            js = [j for ii, j in path if ii == i]
            aligned.append(np.mean([motif[j] for j in js]))
        aligned_segments.append(aligned)
    return np.array(aligned_segments)

def filter_outlier_motifs(aligned_segments, threshold=1.25):
    """
    Supprime les motifs trop éloignés de la moyenne (en multiples d'écart-type de la distance).
    """
    mean_curve = aligned_segments.mean(axis=0)
    distances = np.linalg.norm(aligned_segments - mean_curve, axis=1)
    median = np.median(distances)
    std = np.std(distances)
    # Seuil : par défaut, motifs à plus de 2.5 écart-types de la médiane sont supprimés
    mask = distances < (median + threshold * std)
    return aligned_segments[mask], mask


# ====== 2. EXTRACTION DES FENÊTRES (WINDOWS) ======

def extract_all_windows(df: pd.DataFrame, window_size: int, column: str = None):
    """
    Extrait toutes les fenêtres de taille window_size :
    - Si column est fourni, extrait uniquement cette colonne (univarié)
    - Sinon, extrait toutes les colonnes (multivarié)
    """
    if column is not None and column in df.columns:
        values = df[column].values
        n = len(values)
        windows = []
        indices = []
        for i in range(n - window_size + 1):
            window = values[i:i+window_size]
            if len(window) == window_size:
                windows.append(window)
                indices.append(i)
        return np.array(windows), np.array(indices)
    else:
        # Multivarié : toutes colonnes
        values = df.values
        n = len(values)
        windows = []
        indices = []
        for i in range(n - window_size + 1):
            window = values[i:i+window_size, :]
            if window.shape == (window_size, df.shape[1]):
                windows.append(window)
                indices.append(i)
        return np.array(windows), np.array(indices)


def get_valid_segments(df, idxs, window_size):
    values = df['value'].values
    valid_segments = []
    valid_idxs = []
    for i in idxs:
        seg = values[i:i+window_size]
        if len(seg) == window_size and np.all(np.isfinite(seg)):
            valid_segments.append(seg)
            valid_idxs.append(i)
    return np.array(valid_segments), np.array(valid_idxs)

def get_valid_indices(idxs, window_size, series_length):
    return [int(i) for i in idxs if i + window_size <= series_length]

# ====== 3. CLUSTERING ET UTILS ======

def is_valid_cluster(aligned_segments, min_std=10, min_amplitude=30, max_intra_std=1200):
    # 1. Tous les motifs doivent être dynamiques
    seg_stds = aligned_segments.std(axis=1)
    seg_amps = aligned_segments.max(axis=1) - aligned_segments.min(axis=1)
    if np.any(seg_stds < min_std) or np.any(seg_amps < min_amplitude):
        return False
    # 2. Le cluster doit être homogène (pas trop dispersé)
    mean_curve = aligned_segments.mean(axis=0)
    intra_std = np.mean(np.linalg.norm(aligned_segments - mean_curve, axis=1))
    if intra_std > max_intra_std:
        return False
    return True


def filter_non_overlapping_idxs(motif_idxs: list, window_size: int) -> list:
    motif_idxs_sorted = sorted(set(motif_idxs))
    filtered = []
    last_end = -1
    for idx in motif_idxs_sorted:
        if idx > last_end:
            filtered.append(idx)
            last_end = idx + window_size - 1
    return filtered

def faiss_kmeans(windows: np.ndarray, n_clusters: int, n_iter: int = 20, verbose: bool = True) -> np.ndarray:
    d = windows.shape[1]
    windows32 = windows.astype(np.float32)
    kmeans = faiss.Kmeans(d, n_clusters, niter=n_iter, verbose=verbose, gpu=False)
    kmeans.train(windows32)
    _, labels = kmeans.index.search(windows32, 1)
    return labels.ravel()

def get_cluster_indices(labels: np.ndarray, keep_mask: np.ndarray, window_size: int, min_size: int = 2) -> dict:
    clusters = {}
    real_idxs = np.where(keep_mask)[0]
    for label in np.unique(labels):
        idxs = real_idxs[labels == label]
        idxs_filtered = filter_non_overlapping_idxs(idxs, window_size)
        if len(idxs_filtered) >= min_size:
            clusters[label] = idxs_filtered
    return clusters

def fuse_similar_clusters(
    clusters: dict,
    df: pd.DataFrame,
    window_size: int,
    dtw_thresh: float = 1001
) -> dict:
    cluster_means = []
    labels = []
    for label, idxs in clusters.items():
        motif_segments = np.array([df['value'].values[i:i + window_size] for i in idxs])
        # Nouveau : filtrer les motifs qui sont vides ou avec trop de NaN
        motif_segments = motif_segments[
            [np.isfinite(seg).all() and len(seg) == window_size for seg in motif_segments]
        ]
        if len(motif_segments) == 0:
            continue  # On saute ce cluster
        mean_motif = motif_segments.mean(axis=0)
        if not np.all(np.isfinite(mean_motif)):
            continue  # On saute aussi si la moyenne est encore NaN (cas extrême)
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


# ====== 4. EXCLUSION DES DISCORDS (ANOMALIES) ======

def exclude_discord_windows(
    windows: np.ndarray,
    df_profile: pd.DataFrame,
    window_size: int,
    top_discord_k: int = 0.02,
    discord_margin: int = None
):
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


# ====== 5. PIPELINE PRINCIPAL ======
def discover_univariate_patterns(
    df: pd.DataFrame,
    df_profile: pd.DataFrame,
    window_size: int,
    column :str,
    n_clusters: int = 1000,
    min_cluster_size: int = 2,
    top_discord_k: float = 0.02,
    scale_windows: bool = False,
    dtw_fusion_thresh: float = 100,
    core_top_n: int = 5,
    min_final_matches: int = 5
) -> dict:
    # 1. Extraction de toutes les fenêtres glissantes et indices d'origine
    windows, window_indices = extract_all_windows(df, window_size, column)
    valid_mask = np.all(np.isfinite(windows), axis=1)
    windows = windows[valid_mask]
    window_indices = window_indices[valid_mask]
    df_profile_valid = df_profile.iloc[window_indices].reset_index(drop=True)

    # 2. Exclusion des discords
    windows_filtered, keep_mask, discord_idxs = exclude_discord_windows(
        windows, df_profile_valid, window_size, top_discord_k=top_discord_k
    )
    kept_indices = window_indices[keep_mask]  # Indices dans la série d'origine

    # 3. (optionnel) Normalisation des fenêtres
    if scale_windows:
        scaler = StandardScaler()
        windows_filtered = scaler.fit_transform(windows_filtered)
    n_clusters = len(windows_filtered) // 40
    print(n_clusters)

    # 4. Clustering (KMeans FAISS)
    labels = faiss_kmeans(
        windows_filtered, n_clusters=n_clusters, n_iter=25, verbose=False
    )

    # 5. Regroupement par cluster
    clusters = get_cluster_indices(
        labels, keep_mask, window_size, min_size=min_cluster_size
    )
    clusters_true = {
        label: [int(kept_indices[i]) for i in idxs if 0 <= i < len(kept_indices)]
        for label, idxs in clusters.items()
    }

    # 6. Fusion des clusters similaires (DTW)
    fused_clusters = fuse_similar_clusters(
        clusters_true, df, window_size, dtw_thresh=dtw_fusion_thresh
    )

    # 7. Extraction et alignement des motifs pour chaque pattern
    motifs_by_pattern = []
    for label, idxs in fused_clusters.items():
        idxs = [i for i in idxs if 0 <= i <= len(df) - window_size]
        if not idxs:
            continue
        motif_segments, valid_core_idxs = get_valid_segments(df, idxs, window_size)
        # Garder juste ce filtre pour éviter les motifs strictement constants
        stds = motif_segments.std(axis=1)
        keep_mask = stds > 1e-6  # seuil très bas
        motif_segments = motif_segments[keep_mask]
        valid_core_idxs = valid_core_idxs[keep_mask]
        if len(motif_segments) < core_top_n:
            continue

        # Sélectionne les top N motifs comme "core"
        core_idxs = valid_core_idxs[:core_top_n]
        core_segments, _ = get_valid_segments(df, core_idxs, window_size)
        if len(core_segments) == 0:
            continue

        # Alignement DTW
        aligned_core_segments = align_segments_to_reference(core_segments)
        filtered_segments, mask = filter_outlier_motifs(aligned_core_segments)
        pattern_idxs = [
            (int(idx), int(idx) + window_size) for idx in valid_core_idxs
            if 0 <= int(idx) <= len(df) - window_size
        ]
        if not is_valid_cluster(filtered_segments, min_std=10, min_amplitude=30, max_intra_std=1200):
            continue
        if len(filtered_segments) >= min_final_matches:
            motifs_by_pattern.append({
                "pattern_label": label,
                "core_motifs_aligned": filtered_segments,
                "all_motif_indices": [pattern_idxs[i] for i, k in enumerate(mask) if k]
            })

    # Gestion sûre des discord indices
    discord_idxs = discord_idxs[discord_idxs < len(window_indices)]
    safe_discord_idxs = [
        int(window_indices[d]) for d in discord_idxs
        if 0 <= window_indices[d] <= len(df) - window_size
    ]

    return {
        "patterns": motifs_by_pattern,
        "matrix_profile": df_profile,
        "discord_indices": safe_discord_idxs,
    }

def discover_multivariate_patterns(
    df_list: List[pd.DataFrame],
    df_profiles: List[pd.DataFrame],
    window_size: int,
    top_k: int = 5,
    n_clusters: int = 100,
    min_signals: int = 2,
    min_cluster_size: int = 2,
    dtw_fusion_thresh: float = 100,
    core_top_n: int = 3,
    min_final_matches: int = 3,
    scale_windows: bool = False,
) -> dict:
    """
    Détection de motifs communs multi-signal à partir des matrix profiles (une par variable).
    """
    # 1. Repère les top_k indices (timestamps) de chaque profile
    motif_indices_by_signal = []
    for profile in df_profiles:

        idxs = np.argsort(profile["value"].values)[:top_k]
        motif_indices_by_signal.append(set(profile.index[idxs]))

    # 2. Trouve les timestamps où au moins `min_signals` signaux détectent un motif
    all_indices = sorted(set.union(*motif_indices_by_signal))
    timeline = np.array(sorted(set.union(*motif_indices_by_signal)))
    index_hits = {idx: 0 for idx in timeline}
    for idx_set in motif_indices_by_signal:
        for idx in idx_set:
            index_hits[idx] += 1
    # Prend les indices "populaires" (>= min_signals)
    common_indices = [idx for idx, count in index_hits.items() if count >= min_signals]
    common_indices = sorted(common_indices)
    # Option: tolérer un petit décalage (+/- 1 fenêtre)
    # (À améliorer si besoin...)

    # 3. Extrait les fenêtres multidimensionnelles synchrones à ces indices
    X = []
    for idx in common_indices:
        win = []
        for df in df_list:
            if idx in df.index:
                pos = df.index.get_loc(idx)
                if pos + window_size <= len(df):
                    segment = df.values[pos:pos+window_size, :]
                    if segment.shape[0] == window_size:
                        win.append(segment)
        if len(win) == len(df_list):
            motif = np.concatenate(win, axis=1)
            X.append(motif)
    if not X:
        return {"patterns": [], "msg": "Aucun motif commun trouvé."}
    X = np.array(X)  # [n_windows, window_size, n_vars]

    # 4. Optionnel : normalisation
    if scale_windows:
        shape = X.shape
        X = X.reshape(shape[0], -1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = X.reshape(shape)

    # 5. Clustering FAISS sur les fenêtres extraites (flatten pour faiss)
    X_flat = X.reshape(X.shape[0], -1)
    n_clusters = min(len(X_flat)//4, n_clusters)
    labels = faiss_kmeans(X_flat, n_clusters=n_clusters, n_iter=25, verbose=False)
    clusters = get_cluster_indices(labels, np.ones(len(labels)), window_size, min_size=min_cluster_size)
    # Récupère les indices d'origine pour chaque cluster
    clusters_true = {label: [common_indices[i] for i in idxs] for label, idxs in clusters.items()}

    # 6. Fusion clusters similaires (DTW multivarié)
    # À faire sur la moyenne de chaque cluster, comme dans ton pipeline d'origine
    # On simplifie ici :
    patterns = []
    for label, idxs in clusters_true.items():
        segments = np.array([X[i] for i, ci in enumerate(common_indices) if ci in idxs])
        if len(segments) < core_top_n:
            continue
        # Aligne les motifs (moyenne DTW multivarié)
        # À affiner : ici, tu peux aligner chaque variable séparément ou utiliser tslearn.metrics.dtw
        patterns.append({
            "pattern_label": label,
            "segments": segments,
            "indices": idxs,
        })

    return {"patterns": patterns}

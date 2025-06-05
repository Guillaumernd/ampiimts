import numpy as np
import pandas as pd
import stumpy
from tslearn.metrics import dtw, dtw_path

def align_segments_to_reference(segments, reference=None):
    """Aligne chaque segment sur la médoïde en utilisant DTW."""
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

def filter_outlier_motifs(aligned_segments, threshold=5):
    """Filtre les motifs trop loin de la moyenne (écart-type)."""
    mean_curve = aligned_segments.mean(axis=0)
    distances = np.linalg.norm(aligned_segments - mean_curve, axis=1)
    median = np.median(distances)
    std = np.std(distances)
    mask = distances < (median + threshold * std)
    return aligned_segments[mask], mask

def medoid_index(segments):
    """Retourne l'indice du segment le plus proche de tous les autres (médoïde)."""
    D = np.array([[dtw(s1, s2) for s2 in segments] for s1 in segments])
    return np.argmin(D.sum(axis=1))

def exclude_discords(mp, top_percent_discords=10, margin=0):
    """Indices à exclure : les discord (= les plus grandes valeurs du matrix profile)."""
    top_n = int(top_percent_discords * len(mp))
    P = mp[:, 0].astype(float)
    valid_idx = np.where(~np.isnan(P))[0]
    discords = valid_idx[np.argsort(P[valid_idx])[-top_n:][::-1]]
    if margin > 0:
        # Exclusion d'une marge autour des discord indices
        mask = np.ones(len(P), dtype=bool)
        for idx in discords:
            mask[max(0, idx-margin):min(len(P), idx+margin+1)] = False
        return discords, mask
    else:
        mask = np.ones(len(P), dtype=bool)
        mask[discords] = False
        return discords, mask

def discover_patterns_stumpy_mixed(
    df, window_size, max_motifs=3, top_percent_discords=0.01, margin_discord=0,
    max_matches=10):
    """
    - Détection des motifs principaux par stumpy.motifs (indices)
    - Exclusion des discord windows (par indices et marge éventuelle)
    - Alignement DTW sur la médoïde, filtrage des motifs éloignés
    - Retourne par motif : segments alignés, indice de la médoïde, indices des motifs, discord indices
    - TOUS les indices sont CENTRÉS (alignés au centre de la fenêtre)
    """
    X = df["value"].values
    mp = stumpy.stump(X, window_size, normalize=False)

    # Centrage de l'index du Matrix Profile
    columns = ['value', 'index_1', 'index_2', 'index_3']
    df_profile = pd.DataFrame(mp, columns=columns)
    profile_len = len(df) - window_size + 1
    center_indices = np.arange(profile_len) + window_size // 2
    center_indices = center_indices[center_indices < len(df)]
    df_profile = df_profile.iloc[:len(center_indices)]
    df_profile.index = df.index[center_indices]

    motif_distances, motif_indices = stumpy.motifs(X, mp[:, 0], max_matches=max_matches, max_motifs=max_motifs, normalize=False)
    discords, keep_mask = exclude_discords(mp, top_percent_discords=top_percent_discords, margin=margin_discord)

    # Gestion des indices de discord avec la sécurité de la fenêtre
    safe_discords_centered = []
    window_indices = np.arange(len(df) - window_size + 1)  # Indices valides pour les fenêtres
    discord_idxs = np.argsort(mp[:, 0])[-int(top_percent_discords * len(mp)):][::-1]  # Top discords

    for idx in discord_idxs:
        # S'assurer que l'indice du discord est dans la plage autorisée
        if 0 <= idx <= len(df) - window_size:
            safe_discords_centered.append(
                (window_indices[idx], window_indices[idx] + window_size)
            )

    results = []
    for i in range(motif_indices.shape[0]):
        group = motif_indices[i]
        group = [int(np.atleast_1d(idx)[0]) for idx in group]
        if len(group) == 0:
            continue
        # Exclusion des motifs chevauchant des discord windows
        group_filtered = [idx for idx in group if keep_mask[idx]]
        if len(group_filtered) < max_matches :
            continue
        # Extraction des segments
        segments = []
        for idx in group_filtered:
            if idx + window_size <= len(X):  # Assurer que le segment ne dépasse pas la taille du signal
                segment = X[idx:idx + window_size]
                if len(segment) == window_size:  # Vérifier que la longueur du segment est correcte
                    segments.append(segment)

        # Convertir les segments en un tableau numpy une fois tous les segments extraits
        segments = np.array(segments)
        group_filtered = [idx for idx in group_filtered if idx + window_size <= len(X)]
        if len(segments) < max_matches :
            continue
        # Sélectionne le cœur
        core_idxs = np.array(group_filtered[:max_matches ])
        core_segments = segments[:max_matches ]
        medoid_idx = medoid_index(core_segments)

        aligned = align_segments_to_reference(core_segments, core_segments[medoid_idx])

        # >>> Calcule la médoïde locale sur le sous-ensemble aligné <<<
        medoid_idx_local = medoid_index(aligned)
        medoid_value_idx = int(core_idxs[medoid_idx_local]) + window_size // 2

        # Centrage des indices de motifs alignés
        all_motif_centered = [
            (int(idx) + window_size // 2, int(idx) + window_size // 2 + window_size) for idx in core_idxs
        ]

        results.append({
            "pattern_label": f"motif_{i+1}",
            "aligned_motifs": aligned,
            "all_motif_indices": all_motif_centered,
            "medoid_idx": medoid_value_idx
        })

    return {
        "patterns": results,
        "matrix_profile": df_profile,   # <-- DataFrame indexé sur le centre
        "discord_indices": safe_discords_centered,  # Renvoie les indices des zones de discord
        "window_size": window_size
    }

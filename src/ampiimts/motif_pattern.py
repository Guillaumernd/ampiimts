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

def medoid_index(segments):
    """Retourne l'indice du segment le plus proche de tous les autres (médoïde)."""
    D = np.array([[dtw(s1, s2) for s2 in segments] for s1 in segments])
    return np.argmin(D.sum(axis=1))

def exclude_discords(mp, window_size, top_percent_discords=0.01, margin=0):
    """
    Retourne les indices des top discord (= plus grandes valeurs du MP).
    top_percent_discords: fraction (ex 0.01 pour 1%)
    """
    P = mp[:, 0].astype(float)
    valid_idx = np.where(~np.isnan(P))[0]
    top_n = max(1, int(top_percent_discords * len(valid_idx)))  # Toujours au moins 1
    # Discords = indices des plus grandes valeurs du MP (dans valid_idx)
    discords_idx = valid_idx[np.argsort(P[valid_idx])[-top_n:][::-1]]
    discords_centered = discords_idx + window_size // 2

    return discords_centered


def discover_patterns_stumpy_mixed(
    df, window_size, max_motifs=3, top_percent_discords=0.01, margin_discord=10,
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

    # Calcul de la longueur et des indices centrés
    profile_len = len(df) - window_size
    center_indices = np.arange(profile_len) + window_size // 2

    # Limiter la longueur de df_profile pour correspondre aux indices centrés
    df_profile = df_profile.iloc[:len(center_indices)]
    df_profile.index = df.index[center_indices]

    # Ajouter des NaN au début et à la fin pour aligner visuellement le Matrix Profile
    nan_values = np.full(window_size // 2, np.nan)  # Crée des NaN pour le début et la fin
    df_profile_with_nan = pd.DataFrame(np.concatenate([nan_values, df_profile['value'].values, nan_values]), columns=['value'])

    # Ajuste l'index du DataFrame pour que sa longueur corresponde à celle de df
    df_profile_with_nan.index = df.index[:len(df_profile_with_nan)]

    motif_distances, motif_indices = stumpy.motifs(X, mp[:, 0], max_matches=max_matches, max_motifs=max_motifs, normalize=False)
    discords = exclude_discords(mp, window_size, top_percent_discords=top_percent_discords, margin=margin_discord)

    # Retourne seulement les indices des discordes
    results = []
    for i in range(motif_indices.shape[0]):
        group = motif_indices[i]
        group = [int(np.atleast_1d(idx)[0]) + window_size // 2 for idx in group]
        if len(group) == 0:
            continue
        # Exclusion des motifs chevauchant des discord windows
        group_filtered = [idx for idx in group if idx not in discords]
        if len(group_filtered) < max_matches :
            continue
        # Extraction des segments
        segments = []
        half = window_size // 2
        for center in group_filtered:
            if window_size % 2 == 0:
                start = center - half
                end = center + half
            else:
                start = center - half
                end = center + half + 1
            if start >= 0 and end <= len(X):
                segment = X[start:end]
                if len(segment) == window_size:
                    segments.append(segment)

        if len(segments) < max_matches :
            continue
        core_idxs = np.array(group_filtered[:max_matches ])

        
        core_segments = segments[:max_matches ]
        medoid_idx = medoid_index(core_segments)

        aligned = align_segments_to_reference(core_segments, core_segments[medoid_idx])

        # >>> Calcule la médoïde locale sur le sous-ensemble aligné <<<
        medoid_idx_local = medoid_index(aligned)
        medoid_value_idx = int(core_idxs[medoid_idx_local])

        # Centrage des indices de motifs alignés
        all_motif_centered = [
            (int(idx), int(idx) + window_size) for idx in core_idxs
        ]

        results.append({
            "pattern_label": f"motif_{i+1}",
            "aligned_motifs": aligned,
            "all_motif_indices": all_motif_centered,
            "medoid_idx": medoid_value_idx,
            "motif_indices_debut": [int(np.atleast_1d(idx)[0]) for idx in group]

        })

    return {
        "patterns": results,
        "matrix_profile": df_profile_with_nan,   # <-- DataFrame avec NaN ajouté
        "discord_indices": discords,  # Renvoie seulement les indices des discord
        "window_size": window_size,
    }

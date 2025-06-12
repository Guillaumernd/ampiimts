import numpy as np
import pandas as pd
import stumpy
import faiss

def exclude_discords(
    mp, window_size, discord_top_pct=0.04, X=None, max_nan_frac=0.0, margin=0
):
    """Return centered indices of discords based on the top discord_top_pct of MP values."""
    P = mp[:, 0].astype(float)
    valid_idx = np.where(~np.isnan(P))[0]
    candidates = []
    if X is not None:
        nan_indices = np.where(np.isnan(X))[0]
        for idx in valid_idx:
            start, end = idx, idx + window_size
            if end > len(X):
                continue
            window = X[start:end]
            if np.isnan(window).mean() > max_nan_frac:
                continue
            if margin > 0 and np.any(
                (nan_indices >= start - margin) & (nan_indices < end + margin)
            ):
                continue
            candidates.append(idx)
    else:
        candidates = valid_idx.tolist()

    if not candidates:
        return np.array([], dtype=int)

    n_top = max(1, int(np.ceil(discord_top_pct * len(candidates))))
    sorted_cands = sorted(candidates, key=lambda i: P[i], reverse=True)
    top_group = sorted_cands[:n_top]
    cutoff = min(P[i] for i in top_group)
    discords = [idx for idx in candidates if P[idx] >= cutoff]
    return np.array(discords) + window_size // 2


def discover_patterns_stumpy_mixed(
    df, window_size, max_motifs=3, discord_top_pct=0.04,
    max_matches=10
):
    """Detect motifs and automatic-percentile discords on a univariate signal,
    puis utilise FAISS pour déterminer le véritable médian, sans se soucier du nom de la colonne."""

    # --- Récupérer la colonne des valeurs, quel que soit son nom ---
    if df.shape[1] != 1:
        raise ValueError("Le DataFrame doit contenir exactement une colonne.")
    # On prend la première et unique colonne
    X = df.iloc[:, 0].values

    # Calcul du matrix profile
    mp = stumpy.stump(X, window_size, normalize=False)

    # Préparation du DataFrame de profile pour retour
    profile_len = len(df) - window_size
    center_indices = np.arange(profile_len) + window_size // 2

    # On crée un DataFrame dédié pour le matrix_profile
    df_profile = pd.DataFrame(
        mp,
        columns=['value', 'index_1', 'index_2', 'index_3']
    ).iloc[:profile_len]
    df_profile.index = df.index[center_indices]

    # On pad avec des NaN pour aligner sur la longueur d'origine
    nan_pad = np.full(window_size // 2, np.nan)
    df_profile_with_nan = pd.DataFrame(
        np.concatenate([nan_pad, df_profile['value'].values, nan_pad]),
        columns=['value']
    )
    df_profile_with_nan.index = df.index[:len(df_profile_with_nan)]

    # 1) Cutoff strict sur les 0.5% meilleurs MP values
    motif_cutoff = np.nanquantile(mp[:, 0], 0.005)

    # Extraction des motifs bruts avec STUMPY
    motif_distances, motif_indices = stumpy.motifs(
        T=X,
        P=mp[:, 0],
        min_neighbors=3,
        cutoff=motif_cutoff,
        max_matches=max_matches,
        max_motifs=max_motifs,
        normalize=False
    )

    # Identification des discords
    discords = exclude_discords(
        mp, window_size, discord_top_pct=discord_top_pct,
        X=X, max_nan_frac=0.1, margin=10
    )

    results = []
    occupied = []

    for i, group in enumerate(motif_indices):
        # 1) Indices de départ fournis par STUMPY
        starts = [int(np.atleast_1d(idx)[0]) for idx in group]

        # 2) Exclusion des chevauchements avec les discords
        starts = [
            s for s in starts
            if not np.any((discords >= s) & (discords < s + window_size))
        ]

        # 3) Filtrer les segments complets (sans NaN)
        segments = []
        valid_starts = []
        for s in starts:
            seg = X[s : s + window_size]
            if len(seg) == window_size and not np.isnan(seg).any():
                segments.append(seg.astype('float32'))
                valid_starts.append(s)
        if len(segments) < 2:
            continue

        # 4) Construction de l'index FAISS
        segs_arr = np.stack(segments)
        index = faiss.IndexFlatL2(window_size)
        index.add(segs_arr)

        # 5) Distances pair-à-pair pour trouver le médian
        D, _ = index.search(segs_arr, len(segments))
        sum_dists = D.sum(axis=1)
        med_loc = int(np.argmin(sum_dists))
        medoid_start = valid_starts[med_loc]
        medoid_seg = X[medoid_start : medoid_start + window_size]

        # 6) Recherche de toutes les occurrences du médian
        matches = stumpy.match(
            Q=medoid_seg,
            T=X,
            max_distance=None,
            max_matches=None,
            normalize=False
        )
        motif_starts = [int(idx) for _, idx in matches]

        # 7) Exclusion des matches chevauchant un discord
        motif_starts = [
            s for s in motif_starts
            if not np.any((discords >= s) & (discords < s + window_size))
        ]

        # 8) Exclusion des chevauchements inter-patterns
        filtered = []
        for s in sorted(motif_starts):
            interval = (s, s + window_size)
            if not any(max(s, os) < min(s + window_size, oe) for os, oe in occupied):
                filtered.append(s)
                occupied.append(interval)
        if not filtered:
            continue

        results.append({
            'pattern_label': f'motif_{i+1}',
            'medoid_idx': medoid_start,
            'motif_indices_debut': filtered
        })

    return {
        'patterns': results,
        'matrix_profile': df_profile_with_nan,
        'discord_indices': discords,
        'window_size': window_size
    }

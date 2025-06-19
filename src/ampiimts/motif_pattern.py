import numpy as np
import pandas as pd
import stumpy
import faiss


def exclude_discords_multi(mp, window_size, discord_top_pct=0.04, X=None, max_nan_frac=0.0, margin=0):
    """Return centered indices of discords for multivariate MSTUMP results."""
    mp = np.asarray(mp)
    P = np.nanmean(mp, axis=0) if mp.ndim == 2 else mp.astype(float)

    valid_idx = np.where(~np.isnan(P))[0]
    candidates = []

    if X is not None:
        X = np.asarray(X, dtype=float)
        n = X.shape[1]

        for idx in valid_idx:
            start, end = idx, idx + window_size
            if end > n:
                continue
            window = X[:, start:end]
            if np.any(np.isnan(window).mean(axis=1) > max_nan_frac):
                continue
            if margin > 0:
                nan_indices = np.where(np.isnan(X))[1]
                if np.any((nan_indices >= start - margin) & (nan_indices < end + margin)):
                    continue
            candidates.append(idx)
    else:
        candidates = valid_idx.tolist()

    if not candidates:
        return np.array([], dtype=int)

    n_top = max(1, int(np.ceil(discord_top_pct * len(candidates))))
    top_group = sorted(candidates, key=lambda i: P[i], reverse=True)[:n_top]
    cutoff = min(P[i] for i in top_group)
    discords = [idx for idx in candidates if P[idx] >= cutoff]

    return np.array(discords) + window_size // 2

def exclude_discords(mp, window_size, discord_top_pct=0.04, X=None, max_nan_frac=0.0, margin=0):
    """Return centered indices of discords for univariate STUMP results."""
    if isinstance(mp, pd.DataFrame):
        P = mp.iloc[:, 0].values.astype(float)
    else:
        mp = np.asarray(mp)
        if mp.ndim == 2:
            P = mp[:, 0].astype(float)
        else:
            P = mp.astype(float)

    # Vérification que P est bien un tableau float utilisable
    P = np.asarray(P, dtype=float)

    valid_idx = np.where(~np.isnan(P))[0]
    candidates = []

    if X is not None:
        X = np.asarray(X, dtype=float)
        nan_indices = np.where(np.isnan(X))[0]
        for idx in valid_idx:
            start, end = idx, idx + window_size
            if end > len(X):
                continue
            window = X[start:end]
            if np.isnan(window).mean() > max_nan_frac:
                continue
            if margin > 0 and np.any((nan_indices >= start - margin) & (nan_indices < end + margin)):
                continue
            candidates.append(idx)
    else:
        candidates = valid_idx.tolist()

    if not candidates:
        return np.array([], dtype=int)

    n_top = max(1, int(np.ceil(discord_top_pct * len(candidates))))
    top_group = sorted(candidates, key=lambda i: P[i], reverse=True)[:n_top]
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


def discover_patterns_mstump_mixed(
    df: pd.DataFrame,
    window_size: int,
    max_motifs: int = 3,
    discord_top_pct: float = 0.02,
    max_matches: int = 10,
):
    """
    1. Calcul de mstump sur X multi-dim : P, I
    2. Extraction de motifs par mmotifs (sous-espaces optimaux MDL)
    3. On ne garde QUE ceux dont motif_subspace == [True]*d
    4. Discords exclus via exclude_discords (pas de NaN)
    5. Retourne aussi le matrix_profile (P) sous forme de DataFrame
    """
    # 0) Préparation
    X = df.to_numpy(dtype=float).T  # d × n
    d, n = X.shape
    if d < 2:
        raise ValueError("Il faut au moins 2 dimensions.")

    # 1) MSTUMP
    P, I = stumpy.mstump(X, m=window_size, normalize=False, discords=False)

    # Matrix Profile DataFrame
    center = np.arange(n - window_size + 1) + window_size // 2
    idx = df.index[center]
    mp_df = pd.DataFrame(
        data=P.T,
        index=idx,
        columns=[f"mp_dim_{col}" for col in df.columns]
    )

    # 2) MMOTIFS
    motif_distances, motif_indices, motif_subspaces, motif_mdls = stumpy.mmotifs(
        X, P, I,
        max_motifs=max_motifs,
        max_matches=max_matches,
        normalize=False,
    )

    # 3) Discords multivariés (filtrés via exclude_discords)
    P_disc, _ = stumpy.mstump(X, m=window_size, normalize=False, discords=True)
    disc_idxs = exclude_discords_multi(
        mp=P_disc,
        window_size=window_size,
        discord_top_pct=discord_top_pct,
        X=X,
        max_nan_frac=0.1,
        margin=10
    )
    discords = sorted(disc_idxs)

    # 4) Traitement des motifs
    aligned_patterns = []
    occupied = []

    for motif_id, subspace in enumerate(motif_subspaces):
        min_dims = max(1, int(0.40 * d))
        if np.count_nonzero(subspace) < min_dims:
            continue

        motif_starts = []
        for group in motif_indices[motif_id]:
            for idx in np.atleast_1d(group):
                idx = int(idx)
                if idx + window_size <= len(df):
                    motif_starts.append(idx)

        if len(motif_starts) < 2:
            continue

        medoid_start = motif_starts[0]

        valid_motif_starts = []
        for s in motif_starts:
            if any(abs(s - d0) < window_size for d0 in discords):
                continue
            span = (s, s + window_size)
            if any(max(s, o[0]) < min(span[1], o[1]) for o in occupied):
                continue
            valid_motif_starts.append(s)
            occupied.append(span)

        if valid_motif_starts:
            aligned_patterns.append({
                "pattern_label": f"mmotif_{motif_id + 1}",
                "medoid_idx": medoid_start,
                "motif_indices_debut": valid_motif_starts
            })

    return {
        "patterns": aligned_patterns,
        "discord_indices": discords,
        "window_size": window_size,
        "matrix_profile": mp_df
    }

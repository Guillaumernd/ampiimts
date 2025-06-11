import numpy as np
import pandas as pd
import stumpy
from tslearn.metrics import dtw, dtw_path
from tslearn.barycenters import dtw_barycenter_averaging

def medoid_index(segments):
    """Return the index of the segment closest to all others (the medoid)."""
    D = np.array([[dtw(s1, s2) for s2 in segments] for s1 in segments])
    return np.argmin(D.sum(axis=1))

def exclude_discords(
    mp, window_size, discord_top_pct=0.04, X=None, max_nan_frac=0.0, margin=0
):
    """Return centered indices of discords based on the top discord_top_pct of MP values.

    Select the highest discord_top_pct fraction of valid MP points, determine the
    minimum value among them, then include all points above this cutoff."""
    P = mp[:, 0].astype(float)
    valid_idx = np.where(~np.isnan(P))[0]

    # Filter windows based on NaN fraction and margin
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
            if margin > 0 and np.any((nan_indices >= start - margin) & (nan_indices < end + margin)):
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
    df, window_size, max_motifs=2, discord_top_pct=0.04,
    max_matches=10
):
    """Detect motifs and automatic-percentile discords on a univariate signal.

    Returns motifs (with medoid and start indices), matrix profile, and
    discord indices based on the top discord_top_pct fraction of MP."""
    X = df['value'].values
    mp = stumpy.stump(X, window_size, normalize=False)

    # Prepare matrix profile for return
    profile_len = len(df) - window_size
    center_indices = np.arange(profile_len) + window_size // 2
    df_profile = pd.DataFrame(mp, columns=['value', 'index_1', 'index_2', 'index_3'])
    df_profile = df_profile.iloc[:len(center_indices)]
    df_profile.index = df.index[center_indices]
    nan_pad = np.full(window_size // 2, np.nan)
    df_profile_with_nan = pd.DataFrame(
        np.concatenate([nan_pad, df_profile['value'].values, nan_pad]),
        columns=['value']
    )
    df_profile_with_nan.index = df.index[:len(df_profile_with_nan)]
        
    # 1) Cutoff très strict : seuls les 5% des plus petits MP
    motif_cutoff_pct = 0.05
    motif_cutoff = np.nanquantile(mp[:, 0], motif_cutoff_pct)

    # # 2) max_distance plus laxiste : on prend le 90% percentile des MP comme proxy
    # max_distance_pct = 0.5
    # max_distance = np.nanquantile(mp[:, 0], max_distance_pct)

    # Appel à stumpy.motifs
    motif_distances, motif_indices = stumpy.motifs(
        T=X,
        P=mp[:, 0],
        min_neighbors=3,        # au moins 3 occurrences
        # max_distance=max_distance,
        cutoff=motif_cutoff,
        max_matches=max_matches,
        max_motifs=max_motifs,
        normalize=False
    )

    # Discords using automated top-percent cutoff
    discords = exclude_discords(
        mp, window_size, discord_top_pct=discord_top_pct,
        X=X, max_nan_frac=0.1, margin=10
    )

    results = []
    for i, group in enumerate(motif_indices):
        # Use true start indices, not centered
        starts = [int(np.atleast_1d(idx)[0]) for idx in group]
        # Filter out any segment where a discord falls inside
        filtered = [start for start in starts
                    if not np.any((discords >= start) & (discords < start + window_size))]
        if not filtered:
            continue

        core_idxs = np.array(filtered)
        segments = []
        for idx in core_idxs:
            seg = X[idx:idx + window_size]
            if len(seg) == window_size and not np.isnan(seg).any():
                segments.append(seg)

        # # Local medoid
        # medoid_local = medoid_index(segments)
        # medoid_idx = int(core_idxs[medoid_local])
        
        bary = dtw_barycenter_averaging(segments)
        dists_to_bary = [dtw(seg, bary) for seg in segments]
        idx_medoid = np.argmin(dists_to_bary)
        medoid_idx = core_idxs[idx_medoid]
        results.append({
            'pattern_label': f'motif_{i+1}',
            'medoid_idx': medoid_idx,
            'motif_indices_debut': filtered
        })

    return {
        'patterns': results,
        'matrix_profile': df_profile_with_nan,
        'discord_indices': discords,
        'window_size': window_size
    }

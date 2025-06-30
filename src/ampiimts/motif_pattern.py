import numpy as np
import pandas as pd
import stumpy
import faiss
from collections import Counter


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

    # Ensure ``P`` is a usable float array
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
    """Detect motifs and percentile-based discords on a univariate signal.
    FAISS is then used to determine the true medoid regardless of column name."""

    # --- Retrieve the value column regardless of its name ---
    if df.shape[1] != 1:
        raise ValueError("The DataFrame must contain exactly one column.")
    # Take the first and only column
    X = df.iloc[:, 0].values

    # Compute the matrix profile
    mp = stumpy.stump(X, window_size, normalize=False)

    # Prepare DataFrame of the profile for return
    profile_len = len(df) - window_size
    center_indices = np.arange(profile_len) + window_size // 2

    # Build a dedicated DataFrame for the matrix profile
    df_profile = pd.DataFrame(
        mp,
        columns=['value', 'index_1', 'index_2', 'index_3']
    ).iloc[:profile_len]
    df_profile.index = df.index[center_indices]

    # Pad with NaN to match original length
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
        # 1) Start indices provided by STUMPY
        starts = [int(np.atleast_1d(idx)[0]) for idx in group]

        # 2) Exclude overlaps with discords
        starts = [
            s for s in starts
            if not np.any((discords >= s) & (discords < s + window_size))
        ]

        # 3) Filter segments that are complete (no NaN)
        segments = []
        valid_starts = []
        for s in starts:
            seg = X[s : s + window_size]
            if len(seg) == window_size and not np.isnan(seg).any():
                segments.append(seg.astype('float32'))
                valid_starts.append(s)
        if len(segments) < 2:
            continue

        # 4) Build the FAISS index
        segs_arr = np.stack(segments)
        index = faiss.IndexFlatL2(window_size)
        index.add(segs_arr)

        # 5) Pairwise distances to find the medoid
        D, _ = index.search(segs_arr, len(segments))
        sum_dists = D.sum(axis=1)
        med_loc = int(np.argmin(sum_dists))
        medoid_start = valid_starts[med_loc]
        medoid_seg = X[medoid_start : medoid_start + window_size]

        # 6) Find all occurrences of the medoid
        matches = stumpy.match(
            Q=medoid_seg,
            T=X,
            max_distance=None,
            max_matches=None,
            normalize=False
        )
        distance_threshold = 0.2 * np.linalg.norm(medoid_seg)
        motif_starts = [int(idx) for dist, idx in matches if dist < distance_threshold]
        # 7) Exclude matches overlapping a discord
        motif_starts = [
            s for s in motif_starts
            if not np.any((discords >= s) & (discords < s + window_size))
        ]

        # 8) Exclude overlaps between patterns
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
    min_mdl_ratio: float = 0.25,
    verbose: bool = True,
    cluster: bool = False,
    motif: bool = False,
):
    """
    1. Compute ``mstump`` on the multivariate signal to obtain ``P`` and ``I``.
    2. Select useful dimensions using MDL.
    3. Extract motifs with ``mmotifs`` on the reduced space.
    4. Filter discords.
    5. Return aligned motifs, discords and the centered matrix profile.
    """
       
    # 0) Preparation
    X = df.to_numpy(dtype=float).T  # d × n
    d, n = X.shape
    if d < 2:
        raise ValueError("At least two dimensions are required.")
    
    if not cluster:
        # 1) Full matrix profile
        P, I = stumpy.mstump(X, m=window_size, normalize=False, discords=False)

        # 2) Select dimensions via MDL
        motif_indices = np.argsort(P[0])[:min(20, P.shape[1])]
        counts = Counter()
        for idx in motif_indices:
            subseq_idx = np.full(X.shape[0], idx)
            nn_idx = np.full(X.shape[0], I[0][idx])
            _, subspaces = stumpy.mdl(
                T=X,
                m=window_size,
                subseq_idx=subseq_idx,
                nn_idx=nn_idx,
                normalize=False
            )
            for subspace in subspaces:
                for dim in subspace:
                    counts[dim] += 1

        if not counts:
            raise ValueError("MDL did not identify any active dimension.")

        max_count = max(counts.values())
        threshold = max(1, int(min_mdl_ratio * max_count))
        selected_dims = [i for i, c in counts.items() if c >= threshold]

        if verbose:
            print(f"[MDL] Dimensions retenues : {selected_dims} sur {list(range(d))}")

        # Reduce dimensions
        X = X[selected_dims, :]
        df = df.iloc[:, selected_dims]
        d = X.shape[0]

    if motif:
            
        # 3) Reduced matrix profile (motifs + discords)
        P, I = stumpy.mstump(X, m=window_size, normalize=False, discords=False)

        # 4) MMOTIFS
        motif_distances, motif_indices, motif_subspaces, motif_mdls = stumpy.mmotifs(
            X, P, I,
            max_motifs=max_motifs,
            min_neighbors=4,
            max_matches=max_matches,
            normalize=False,
        )

    # 5) Filtered discords
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


    # Centered matrix profile
    profile_len = n - window_size + 1
    center_idx = np.arange(profile_len) + window_size // 2
    index_centered = df.index[center_idx]

    mp_df = pd.DataFrame(
        data=P_disc.T,
        index=index_centered,
        columns=[f"mp_dim_{col}" for col in df.columns]
    )

    pre_pad = window_size // 2
    post_pad = window_size - pre_pad - 1  # ensures m-1 in total

    nan_start = np.full((pre_pad, mp_df.shape[1]), np.nan)
    nan_end = np.full((post_pad, mp_df.shape[1]), np.nan)

    mp_full = pd.DataFrame(
        np.vstack([nan_start, mp_df.values, nan_end]),
        index=df.index[:n],  # back to original length
        columns=mp_df.columns
    )


    # 6) Traitement motifs
    aligned_patterns = []
    if motif:
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

            if len(valid_motif_starts) > 2 :
                segments = [
                    X[:, s : s + window_size]
                    for s in valid_motif_starts
                    if s + window_size <= X.shape[1]
                ]
                segments = [seg for seg in segments if seg.shape[1] == window_size]
                segs_arr = np.stack(segments).astype('float32')
                n, d, w = segs_arr.shape
                segs_arr = segs_arr.reshape(n, d * w)

                index = faiss.IndexFlatL2(segs_arr.shape[1])
                index.add(segs_arr)
                D, _ = index.search(segs_arr, n)

                # 3. Identify the medoid segment
                sum_dists = D.sum(axis=1)
                med_loc = int(np.argmin(sum_dists))
                medoid_start = valid_motif_starts[med_loc]
                medoid_seg = segments[med_loc]  # matrice (d, window_size)
                matches = stumpy.match(
                    Q=medoid_seg,
                    T=X,
                    normalize=False
                )
                distance_threshold = 0.135 * np.linalg.norm(medoid_seg)
                motif_starts = [
                    int(idx) for dist, idx in matches if dist < distance_threshold]
                # Filter motifs overlapping discords
                motif_starts = [
                    s for s in motif_starts
                    if not any((d >= s) and (d < s + window_size) for d in discords)
                ]
                filtered = []
                for s in sorted(motif_starts):
                    span = (s, s + window_size)
                    if not any(max(s, o[0]) < min(span[1], o[1]) for o in occupied):
                        filtered.append(s)
                        occupied.append(span)

                if not filtered:
                    continue

                aligned_patterns.append({
                    "pattern_label": f"mmotif_{motif_id + 1}",
                    "medoid_idx": medoid_start,
                    "motif_indices_debut": filtered
                })
    return {
        "patterns": aligned_patterns,
        "discord_indices": discords,
        "window_size": window_size,
        "matrix_profile": mp_full
    }

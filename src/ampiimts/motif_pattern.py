"""Motif discovery helpers built on top of STUMPY."""

import numpy as np
import pandas as pd
import stumpy
from tslearn.metrics import dtw, dtw_path

def align_segments_to_reference(segments, reference=None):
    """Align each segment on the medoid using DTW."""
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
    """Return the index of the segment closest to all others (the medoid)."""
    D = np.array([[dtw(s1, s2) for s2 in segments] for s1 in segments])
    return np.argmin(D.sum(axis=1))

def exclude_discords(
    mp, window_size, top_percent_discords=0.01, X=None, max_nan_frac=0.0, margin=0
):
    """Return centered indices of the top discords.

    Only windows with less than ``max_nan_frac`` NaN values and outside the
    given margin around NaNs are considered. The top N is computed after this
    filtering so that ``top_percent_discords`` truly applies to usable data.
    """
    P = mp[:, 0].astype(float)
    valid_idx = np.where(~np.isnan(P))[0]
    
    # First pass: filter windows based on NaN ratio and margin around NaNs
    discords_candidates = []
    if X is not None:
        nan_mask = np.isnan(X)
        nan_indices = np.where(nan_mask)[0]
        for idx in valid_idx:
            start = idx
            end = idx + window_size
            if end > len(X):
                continue
            window = X[start:end]
            nan_frac = np.isnan(window).mean()
            # Skip if the window contains too many NaNs
            if nan_frac > max_nan_frac:
                continue
            # Skip if the window is close to a NaN (within ``margin`` points)
            if margin > 0 and np.any(
                (nan_indices >= start - margin) & (nan_indices < end + margin)
            ):
                continue
            discords_candidates.append(idx)
    else:
        discords_candidates = list(valid_idx)
    
    # Compute top_n after filtering to apply ``top_percent_discords`` on valid data
    top_n = max(1, int(top_percent_discords * len(discords_candidates)))
    if top_n > len(discords_candidates):
        top_n = len(discords_candidates)
    # Sort candidates by matrix profile value
    sorted_idx = np.argsort(P[discords_candidates])[-top_n:][::-1]
    discords_idx = np.array(discords_candidates)[sorted_idx]
    discords_centered = discords_idx + window_size // 2
    return discords_centered



def discover_patterns_stumpy_mixed(
    df, window_size, max_motifs=3, top_percent_discords=0.01,
    max_matches=10):
    """Detect motifs and discords on a univariate signal.

    The function returns all motif segments aligned on the medoid as well as the
    indices of discord windows. All indices are centered on the sliding window.
    """
    X = df["value"].values
    mp = stumpy.stump(X, window_size, normalize=False)

    # Center the matrix profile indices
    columns = ['value', 'index_1', 'index_2', 'index_3']
    df_profile = pd.DataFrame(mp, columns=columns)

    # Compute length and centered indices
    profile_len = len(df) - window_size
    center_indices = np.arange(profile_len) + window_size // 2

    # Limit df_profile length to match centered indices
    df_profile = df_profile.iloc[:len(center_indices)]
    df_profile.index = df.index[center_indices]

    # Add NaN at beginning and end for visual alignment of the matrix profile
    nan_values = np.full(window_size // 2, np.nan)
    df_profile_with_nan = pd.DataFrame(np.concatenate([nan_values, df_profile['value'].values, nan_values]), columns=['value'])

    # Adjust DataFrame index so its length matches ``df``
    df_profile_with_nan.index = df.index[:len(df_profile_with_nan)]

    motif_distances, motif_indices = stumpy.motifs(X, mp[:, 0], min_neighbors=3, max_matches=max_matches, max_motifs=max_motifs, normalize=False)
    discords = exclude_discords(mp, window_size, top_percent_discords=top_percent_discords, X=X, max_nan_frac=0.1, margin=10)

    # Return only the discord indices along with motif information
    results = []
    for i in range(motif_indices.shape[0]):
        group = motif_indices[i]
        group = [int(np.atleast_1d(idx)[0]) + window_size // 2 for idx in group]
        if len(group) == 0:
            continue
        # Exclusion des motifs chevauchant des discord windows
        group_filtered = [idx for idx in group if idx not in discords]
        # Extraction des segments
        segments = []
        for start in group_filtered:
            if not window_size % 2 == 0:
                end = start + window_size
            else:
                end = start + window_size + 1
            if start >= 0 and end <= len(X):
                segment = X[start:end]
                if len(segment) == window_size:
                    segments.append(segment)
        core_idxs = np.array(group_filtered)

        core_segments = segments
        medoid_idx = medoid_index(core_segments)

        aligned = align_segments_to_reference(core_segments, core_segments[medoid_idx])

        # >>> Calcule la médoïde locale sur le sous-ensemble aligné <<<
        medoid_idx_local = medoid_index(aligned)
        medoid_value_idx = int(core_idxs[medoid_idx_local])
        motif_starts = [int(idx) for idx in core_idxs]
        # Centrage des indices de motifs alignés
        all_motif_centered = [
            (int(idx), int(idx) + window_size) for idx in core_idxs
        ]

        results.append({
            "pattern_label": f"motif_{i+1}",
            "aligned_motifs": aligned,
            "all_motif_indices": all_motif_centered,
            "medoid_idx": medoid_value_idx,
            "motif_indices_debut": motif_starts,

        })

    return {
        "patterns": results,
        "matrix_profile": df_profile_with_nan,   
        "discord_indices": discords,  
        "window_size": window_size,
    }


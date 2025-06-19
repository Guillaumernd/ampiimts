"""Plotting utilities for time series results."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

def plot_patterns_and_discords(df, result, column='value', figsize=(12, 6)):
    """Plot signal, detected motifs and discords along with the matrix profile.

    Motifs are drawn from their start index while discords are shown as vertical
    lines. Below the main plot, the original segments for each pattern are plotted
    with the medoid highlighted in black."""
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        name = df.index.name or "timestamp"
    elif ("timestamp" in df.columns
          and pd.api.types.is_datetime64_any_dtype(df["timestamp"])):
        pass

    motif_colors = ['tab:green', 'tab:purple', 'tab:blue']
    n_patterns = len(result['patterns'])
    window_size = result['window_size']
    height_ratios = [2] + [1] * n_patterns if n_patterns else [2]

    fig, axs = plt.subplots(
        n_patterns + 1,
        1,
        figsize=(figsize[0], figsize[1] * (1 + 0.6 * n_patterns)),
        gridspec_kw={"height_ratios": height_ratios},
    )
    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]

    # -- SUBPLOT 1 : Signal, motifs, discords, MP --
    ax1 = axs[0]
    ax1.plot(df.index, df[column], label='Original Signal', color='black', alpha=0.1)
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel(column)
    matrix_profile_values = result["matrix_profile"]['value']
    mp_len = len(matrix_profile_values)
    center_indices = np.arange(mp_len)
    ax2 = ax1.twinx()
    ax2.plot(df.index[center_indices], matrix_profile_values,
             label='Matrix Profile', color='blue', alpha=0.6)
    ax2.set_ylabel('Matrix Profile')

    # Plot motifs on main signal
    for i, pattern in enumerate(result['patterns']):
        color = motif_colors[i % len(motif_colors)]
        motif_starts = pattern["motif_indices_debut"]
        for j, start in enumerate(motif_starts):
            end = start + window_size
            if 0 <= start < len(df) and end <= len(df):
                ax1.plot(df.index[start:end], df[column].iloc[start:end],
                         color=color,
                         label=f"{pattern['pattern_label']}" if j == 0 else "",
                         linewidth=2)
                # marqueurs début, centre, fin
                label = f"{pattern['pattern_label']}" if j == 0 else None
                ax1.axvspan(
                    df.index[start],
                    df.index[end],
                    color=color,
                    alpha=0.3,
                    label=label
                )


    # Discords
    for discord in result['discord_indices']:
        if 0 <= discord < len(df):
            ax1.axvline(
                df.index[discord], color='red', linestyle='-',
                linewidth=0.8, alpha=0.3, zorder=0)
    ax1.set_title("Motifs (début de fenêtre), Discords, and Matrix Profile")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right')
    ax1.grid(True)

    # -- SUBPLOTS BELOW : Original segments per pattern --
    for i, pattern in enumerate(result["patterns"]):
        ax_motif = axs[i + 1]
        color = motif_colors[i % len(motif_colors)]
        motif_starts = pattern["motif_indices_debut"]
        medoid_start = pattern["medoid_idx"]
        xs = np.arange(window_size)
        # Plot all segments
        for start in motif_starts:
            if 0 <= start <= len(df) - window_size:
                seg = df[column].iloc[start:start + window_size].values
                ax_motif.plot(xs, seg, color=color, alpha=0.2, linewidth=1)
        # Highlight medoid
        if 0 <= medoid_start <= len(df) - window_size:
            medoid_seg = df[column].iloc[medoid_start:medoid_start + window_size].values
            ax_motif.plot(xs, medoid_seg, color='black', linewidth=0.5, label='medoid')
        ax_motif.set_title(pattern["pattern_label"])
        ax_motif.set_xlabel("Relative index in window")
        ax_motif.set_ylabel(column)
        ax_motif.legend()
        ax_motif.grid(True)

    plt.tight_layout(pad=3.0)
    plt.show()


def plot_multidim_patterns_and_discords(df, result, tick_step=500):
    """
    Affiche :
      • chaque dimension de la série multivariée
      • les motifs (zones colorées seulement sur les dimensions actives)
      • les discords (traits rouges)
      • la heat-map du Matrix Profile
    
    Parameters
    ----------
    df : pandas.DataFrame
        Time-series multivariée indexée par dates.
    result : dict
        Sortie de discover_patterns_mstump_mixed :
            - "matrix_profile"  : DataFrame (len=n−m+1 × n_dim), index DatetimeIndex
            - "window_size"     : int
            - "discord_indices" : list
            - "patterns"        : list de dicts
            - "motif_subspaces" : list[np.ndarray[bool]] (facultatif)
    tick_step : int
        Pas pour les ticks (inutile si AutoDateLocator utilisé).
    """
    # --- 0) Récupérations --------------------------------------------
    profile_df   = result["matrix_profile"]             # (n-m+1) × n_dim
    mp           = profile_df.values.T                  # (n_dim, prof_len)
    center_dates = profile_df.index.to_pydatetime()
    window_size  = result["window_size"]
    discords     = result.get("discord_indices", [])
    patterns     = result.get("patterns", [])
    subspaces    = result.get("motif_subspaces", [None] * len(patterns))
    original_cols = [col.replace("mp_dim_", "") for col in profile_df.columns]
    df = df.loc[:, original_cols]  

    n_dim, prof_len = mp.shape
    figsize = (20, 1.5 * (n_dim + 1))

    # --- 1) Préparation des bords X pour la heat-map ------------------
    dnums = mdates.date2num(center_dates)
    diffs = np.diff(dnums)
    xedges = np.empty(prof_len + 1)
    xedges[1:-1] = dnums[:-1] + diffs / 2
    xedges[0]    = dnums[0] - diffs[0] / 2
    xedges[-1]   = dnums[-1] + diffs[-1] / 2
    yedges = np.arange(n_dim + 1)

    # --- 2) Création figure + axes -----------------------------------
    fig, axs = plt.subplots(
        n_dim + 1, 1,
        figsize=(figsize[0], figsize[1] * (1 + 0.2 * n_dim)),
        sharex=True,
        gridspec_kw={"height_ratios": [1] * n_dim + [0.7]},
    )

    motif_colors = ["tab:green", "tab:purple", "tab:blue", "tab:orange",
                    "tab:brown", "tab:pink", "tab:gray", "tab:olive"]

    # Pré-compute : pour chaque motif, la liste des dimensions actives
    pattern_dims = []
    for sp in subspaces:
        if sp is None:
            pattern_dims.append(set(range(n_dim)))
        else:
            pattern_dims.append(set(np.where(np.atleast_1d(sp))[0]))

    # --- 3) Tracé des séries, motifs et discords ---------------------
    for dim, col in enumerate(df.columns):
        ax = axs[dim]
        ax.plot(df.index, df[col], color="black", label=col, linewidth=0.8)

        # motifs (axvspan uniquement si la dimension est active)
        for pat_id, pat in enumerate(patterns):
            c = motif_colors[pat_id % len(motif_colors)]
            if dim not in pattern_dims[pat_id]:
                continue  # dimension non pertinente pour ce motif
            for j, s in enumerate(pat["motif_indices_debut"]):
                if s + window_size >= len(df):
                    continue  # évite un dépassement d'index
                e = s + window_size
                ax.axvspan(
                    df.index[s], df.index[e],
                    color=c,
                    alpha=0.25,
                    label=(pat["pattern_label"] if (j == 0 and dim == 0) else None),
                )


        # discords (traits verticaux rouges)
        for d in discords:
            ax.axvline(df.index[d], color="red", linestyle="--", alpha=0.5, linewidth=0.8)

        ax.set_ylabel(col)
        ax.grid(True, linewidth=0.3, alpha=0.6)

        # mise en forme de l'axe X
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", rotation=45, labelsize="small")
        ax.legend(loc="upper right", fontsize="x-small")

    # --- 4) Heat-map du Matrix Profile -------------------------------
    axh = axs[-1]
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")

    mesh = axh.pcolormesh(
        xedges, yedges, mp,
        cmap=cmap,
        shading="auto",
    )
    fig.colorbar(mesh, ax=axh, label="Matrix Profile")

    # Redessiner discords sur la heat-map
    for d in discords:
        xd = mdates.date2num(df.index[d])
        axh.axvline(xd, color="red", linestyle="--", alpha=0.5, linewidth=0.8)

    # Redessiner motifs (rectangles) mais uniquement sur les dims actives
    for pat_id, pat in enumerate(patterns):
        c = motif_colors[pat_id % len(motif_colors)]
        active_dims = pattern_dims[pat_id]
        if not active_dims:
            continue
        ymin = min(active_dims)
        ymax = max(active_dims) + 1
        for s in pat["motif_indices_debut"]:
            x0 = mdates.date2num(df.index[s])
            x1 = mdates.date2num(df.index[s + window_size])
            rect = plt.Rectangle(
                (x0, ymin),
                x1 - x0,
                ymax - ymin,
                edgecolor=c,
                facecolor="none",
                linewidth=1.2,
                alpha=0.8,
            )
            axh.add_patch(rect)

    # --- 5) Mise en forme finale -------------------------------------
    axh.set_xlim(df.index[0], df.index[-1])
    axh.set_xlabel("Date")
    axh.set_ylabel("Dimension")
    axh.set_yticks(np.arange(n_dim) + 0.5)
    axh.set_yticklabels(df.columns)
    axh.set_title("Multi-dimensional Matrix Profile")

    plt.tight_layout()
    plt.show()

def plot_motif_overlays(df, result, normalize=True):
    """
    Pour chaque motif détecté, affiche ses occurrences superposées par dimension.
    """
    window_size = result["window_size"]
    patterns = result["patterns"]
    profile_df = result["matrix_profile"]
    original_cols = [col.replace("mp_dim_", "") for col in profile_df.columns]
    df = df.loc[:, original_cols]

    n_dim = df.shape[1]
    motif_colors = ["tab:green", "tab:purple", "tab:blue", "tab:orange", "tab:brown", "tab:pink"]

    for i, pat in enumerate(patterns):
        fig, axs = plt.subplots(n_dim, 1, figsize=(12, 2 * n_dim), sharex=True)
        if n_dim == 1:
            axs = [axs]
        motif_label = pat["pattern_label"]
        indices = pat["motif_indices_debut"]
        c = motif_colors[i % len(motif_colors)]

        for dim, col in enumerate(df.columns):
            ax = axs[dim]
            for idx in indices:
                segment = df.iloc[idx:idx + window_size, dim]
                if normalize:
                    segment = (segment - segment.mean()) / (segment.std() + 1e-8)
                ax.plot(np.arange(len(segment)), segment, alpha=0.6, color=c)
            ax.set_ylabel(col)
            ax.grid(True)

        axs[-1].set_xlabel("Window index")
        fig.suptitle(f"Overlay of Motif Occurrences — {motif_label}", fontsize=14)
        plt.tight_layout()
        plt.show()
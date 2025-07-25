"""
Module `plotting`: Visualization utilities for multivariate time series motifs and discords.

This module provides a set of functions for visualizing the output of motif and discord
discovery algorithms on multivariate time series. It supports both raw data plotting
and advanced visualizations such as:

Key Features:
-------------
- Plotting of multivariate time series with motif and discord overlays.
- Heatmaps of multidimensional matrix profiles.
- Overlaid motif segments per dimension for pattern comparison.
- Support for clustered datasets and multiple motif result structures.

All visualizations are built using matplotlib and support both single and clustered datasets.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

def plot_multidim_patterns_and_discords(
    df: pd.DataFrame,
    result: dict | None,
    tick_step: int = 500,
    only_heat_map: bool = True,
) -> None:
    """
    Plot:
      • Each dimension of the multivariate signal
      • Motifs (colored spans only on active dimensions)
      • Discords (red vertical lines)
      • A separate heatmap of the Matrix Profile

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed multivariate signal.
    result : dict
        Output from discover_patterns_mstump_mixed:
            - "matrix_profile"  : DataFrame (len=n−m+1 × n_dim), index DatetimeIndex
            - "window_size"     : int
            - "discord_indices" : list
            - "patterns"        : list of dicts
            - "motif_subspaces" : list[np.ndarray[bool]] (optional)
    tick_step : int, optional
        Tick step for the X axis when using integer ticks.

    Returns
    -------
    None
        Only displays a figure using ``matplotlib``.
    """

    # --- Extract useful information from the result dictionary ---
    profile_df   = result["matrix_profile"]
    mp           = profile_df.values.T
    center_dates = profile_df.index.to_pydatetime()
    window_size  = result["window_size"][1]
    discords     = result.get("discord_indices", [])
    patterns     = result.get("patterns", [])
    subspaces    = result.get("motif_subspaces", [None] * len(patterns))
    original_cols = [col.replace("mp_dim_", "") for col in profile_df.columns]
    df = df.loc[:, original_cols]  # align column order

    n_dim, prof_len = mp.shape
    motif_colors = ["tab:green", "tab:purple", "tab:blue", "tab:orange",
                    "tab:brown", "tab:pink", "tab:gray", "tab:olive"]

    # Pre-compute active dimensions per motif
    pattern_dims = []
    for sp in subspaces:
        if sp is None:
            pattern_dims.append(set(range(n_dim)))
        else:
            pattern_dims.append(set(np.where(np.atleast_1d(sp))[0]))

    if not only_heat_map:
        # === FIGURE 1: Timeseries Plots ============================================
        fig1, axs = plt.subplots(
            n_dim, 1,
            figsize=(20, n_dim * 3),
            sharex=True,
        )
        if n_dim == 1:
            axs = [axs]

        for dim, col in enumerate(df.columns):
            ax = axs[dim]
            ax.plot(df.index, df[col], color="black", label=col, linewidth=0.8)

            # Rescale MP to data range
            mp_series = mp[dim]
            mp_series = np.where(np.isinf(mp_series), np.nan, mp_series)
            valid = ~np.isnan(mp_series)
            mp_rescaled = np.full_like(mp_series, np.nan)

            if valid.any():
                mp_min = np.nanmin(mp_series)
                mp_max = np.nanmax(mp_series)
                denom = mp_max - mp_min
                if denom > 0:
                    mp_rescaled[valid] = (mp_series[valid] - mp_min) / denom
                else:
                    mp_rescaled[valid] = 0

                mp_rescaled = mp_rescaled * (df[col].max() - df[col].min()) + df[col].min()
                ax.plot(center_dates, mp_rescaled, color="blue", alpha=0.8, label="Matrix Profile")

            # Motifs
            for pat_id, pat in enumerate(patterns):
                c = motif_colors[pat_id % len(motif_colors)]
                if dim not in pattern_dims[pat_id]:
                    continue
                for j, s in enumerate(pat["motif_indices_debut"]):
                    if s < 0 or s + window_size >= len(df):
                        continue
                    e = s + window_size
                    ax.axvspan(df.index[s], df.index[e], color=c, alpha=0.25,
                            label=(pat["pattern_label"] if j == 0 and dim == 0 else None))

            # Discords
            for j, d in enumerate(discords):
                ax.axvline(df.index[d], color="red", linestyle="-", alpha=1, linewidth=0.3,
                        label="Discord" if j == 0 else None)

            ax.set_ylabel(col)
            ax.grid(True, linewidth=0.3, alpha=0.6)
            ax.legend(loc="upper right", fontsize="x-small")
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.tick_params(axis="x", rotation=45, labelsize="small")

        axs[-1].set_xlabel("Date")
        axs[dim].set_xlim(df.index.min(), df.index.max())
        fig1.tight_layout()
        plt.show()
        plt.close(fig1)

    # === FIGURE 2: Heatmap Only ===============================================
    dnums = mdates.date2num(center_dates)
    diffs = np.diff(dnums)
    xedges = np.empty(prof_len + 1)
    xedges[1:-1] = dnums[:-1] + diffs / 2
    xedges[0] = dnums[0] - diffs[0] / 2
    xedges[-1] = dnums[-1] + diffs[-1] / 2
    yedges = np.linspace(0, n_dim, n_dim + 1)

    # Dynamic threshold 
    heatmap_height = max(2.5, min(0.35 * n_dim, 10))  # between 2.5 and 10
    fig2, axh = plt.subplots(1, 1, figsize=(20, heatmap_height))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")

    mesh = axh.pcolormesh(xedges, yedges, mp, cmap=cmap, shading="auto")
    fig2.colorbar(mesh, ax=axh, label="Matrix Profile")

    for d in discords:
        xd = mdates.date2num(df.index[d])
        axh.axvline(xd, color="red", linestyle="-", alpha=1, linewidth=0.45)

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
                (x0, ymin), x1 - x0, ymax - ymin,
                edgecolor=c, facecolor="none", linewidth=1.2, alpha=0.8
            )
            axh.add_patch(rect)

    axh.set_xlim(df.index[0], df.index[-1])
    axh.set_xlabel("Date")
    axh.set_ylabel("Dimension")
    axh.set_yticks(np.arange(n_dim) + 0.5)
    axh.set_yticklabels(df.columns)
    axh.set_title("Multi-dimensional Matrix Profile")

    fig2.tight_layout()
    plt.show()
    plt.close(fig2)



def plot_motif_overlays(
    df: pd.DataFrame,
    result: dict | None,
    normalize: bool = False,
) -> None:
    """Plot overlays of all detected motifs for each dimension.

    Parameters
    ----------
    df : pandas.DataFrame
        Original multivariate signal.
    result : dict or None
        Result dictionary from motif discovery. ``None`` plots raw data only.
    normalize : bool, optional
        If ``True`` each segment is z-normalized before plotting.

    Returns
    -------
    None
        Displays the overlay figure using ``matplotlib``.
    """


    window_size = result["window_size"][1]
    patterns = result["patterns"]
    profile_df = result["matrix_profile"]
    
    # 1) Retrieve original column names from the matrix profile
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
                if idx + window_size > len(df):
                    continue  # Skip if segment goes beyond the end
                segment = df.iloc[idx:idx + window_size, dim]
                if len(segment) != window_size:
                    continue
                if normalize:
                    segment = (segment - segment.mean()) / (segment.std() + 1e-8)
                ax.plot(np.arange(len(segment)), segment, alpha=0.6, color=c)
            ax.set_ylabel(col)
            ax.grid(True)

        axs[-1].set_xlabel("Window index")
        fig.suptitle(f"Overlay of Motif Occurrences — {motif_label}", fontsize=14)
        plt.tight_layout()
        plt.show()
        plt.close(fig)


def plot_all_patterns_and_discords(
    df: pd.DataFrame | list,
    result: dict | list | None,
    only_heat_map: bool = True,
    tick_step: int = 500,
) -> None:
    """Plot all multivariate motifs and discords.

    Parameters
    ----------
    df : DataFrame or list
        Dataframes used to compute the motifs.
    result : dict or list or None
        Associated result structure from motif discovery.
    tick_step : int, optional
        Forwarded to :func:`plot_multidim_patterns_and_discords`.
    only_heatmap : optional
        print only heatmap

    Returns
    -------
    None
        Displays figures via ``matplotlib``.
    """

    # --- Normal cases ---
    if isinstance(df, pd.DataFrame) and isinstance(result, dict):
        print(f"Window size : {result['window_size'][0]} ---")
        plot_multidim_patterns_and_discords(df, result, tick_step=tick_step, only_heat_map=only_heat_map)

    else:
        for i, (d, r) in enumerate(zip(df, result)):
            print(f"\n--- Cluster {i+1} (Window size : {r['window_size'][0]}) ---")
            plot_multidim_patterns_and_discords(d, r, tick_step=tick_step, only_heat_map=only_heat_map)


def plot_all_motif_overlays(
    df: pd.DataFrame | list,
    result: dict | list | None,
    normalize: bool = False,
) -> None:
    """Plot motif overlays for one or several datasets.

    Parameters
    ----------
    df : DataFrame or list
        Original data used for motif discovery.
    result : dict or list or None
        Result structure from motif discovery. ``None`` shows raw data only.
    normalize : bool, optional
        If ``True`` z-normalize each segment before plotting.

    Returns
    -------
    None
        Displays overlay figures via ``matplotlib``.
    """
    # --- Cas normaux ---
    if isinstance(df, pd.DataFrame) and isinstance(result, dict):
        plot_motif_overlays(df, result, normalize=normalize)

    else:
        for i, (d, r) in enumerate(zip(df, result)):
            if r["patterns"]:
                print(f"\n--- Cluster {i+1} (Window size : {r["window_size"][0]}) ---")
                plot_motif_overlays(d, r, normalize=normalize)
            else:
                print("No motifs")

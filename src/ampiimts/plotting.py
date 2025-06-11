"""Plotting utilities for time series results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_multiple_dfs(dfs, labels=None, column='value', figsize_per_plot=(12, 4)):
    """
    Plot multiple pandas DataFrames on a single plot and on multiple subplots, 
    each with a distinct main color and label.

    Args:
        dfs (list of pd.DataFrame): List of DataFrames to plot.
        labels (list of str, optional): Labels for each DataFrame. If None, generic labels are used.
        column (str): Column to plot from each DataFrame.
        figsize_per_plot (tuple): Figure size per plot (width, height).
    """
    main_colors = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
    ]
    n = len(dfs)
    if labels is None:
        labels = [f"Series {i+1}" for i in range(n)]
    elif len(labels) != n:
        raise ValueError("`labels` doit avoir la même longueur que `dfs`.")

    # 1) All series on the same plot
    fig, ax = plt.subplots(figsize=(figsize_per_plot[0], figsize_per_plot[1]))
    for df_, label, color in zip(dfs, labels, main_colors):
        df_ = df_.copy()
        if 'timestamp' in df_.columns:
            df_.index = pd.to_datetime(df_['timestamp'], errors='coerce')
            df_ = df_.drop(columns=['timestamp'])
        ax.plot(df_.index, df_[column], label=label, color=color)
    ax.set_title("Summary")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(column)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_all_variables_multiple_dfs(dfs, labels=None, variables=None, figsize_per_plot=(12, 4)):
    """
    For each variable/column, plot all DataFrames on the same graph (superposed),
    then show subplots for each DataFrame if desired.

    Args:
        dfs (list of pd.DataFrame): List of DataFrames to plot (index=timestamp).
        labels (list of str, optional): Labels for each DataFrame.
        variables (list of str, optional): Which columns/variables to plot. If None, uses all common columns.
        figsize_per_plot (tuple): Figure size per plot (width, height).
    """
    main_colors = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
    ]
    n = len(dfs)
    if labels is None:
        labels = [f"Series {i+1}" for i in range(n)]
    elif len(labels) != n:
        raise ValueError("`labels` doit avoir la même longueur que `dfs`.")

    # Détermination des variables à tracer
    if variables is None:
        # Only keep columns that are present in every DataFrame and are numeric
        cols = set(dfs[0].columns)
        for df in dfs[1:]:
            cols &= set(df.columns)
        # Filter numeric columns only
        first_df = dfs[0][list(cols)]
        variables = [c for c in cols if pd.api.types.is_numeric_dtype(first_df[c])]
    if not variables:
        raise ValueError("No numeric variables/columns found in all DataFrames.")

    # Un plot par variable
    for var in variables:
        plt.figure(figsize=figsize_per_plot)
        for df_, label, color in zip(dfs, labels, main_colors):
            if 'timestamp' in df_.columns:
                df_ = df_.copy()
                df_.index = pd.to_datetime(df_['timestamp'], errors='coerce')
                df_ = df_.drop(columns=['timestamp'])
            plt.plot(df_.index, df_[var], label=label, color=color)
        plt.title(f"Comparison for variable '{var}'")
        plt.xlabel("Timestamp")
        plt.ylabel(var)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


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


def plot_multidim_matrix_profile(df, result, figsize=(12, 6)):
    """Plot a heatmap of the multi-dimensional matrix profile.

    Parameters
    ----------
    df : pandas.DataFrame
        Original multi-dimensional time series used for the computation.
    result : dict
        Output dictionary returned by :func:`matrix_profile` on a
        multi-dimensional DataFrame.
    figsize : tuple, default (12, 6)
        Size of the resulting figure.
    """

    profile = result["profile"]
    window_size = result["window_size"]
    discords = result.get("discord_indices", [])
    motif_indices = result.get("motif_indices", [])

    plt.figure(figsize=figsize)
    sns.heatmap(
        profile.T,
        cmap="viridis",
        xticklabels=False,
        yticklabels=df.columns,
        cbar_kws={"label": "Matrix Profile"},
    )

    for d in discords:
        plt.axvline(d, color="red", linestyle="--", linewidth=0.10, alpha=0.3, zorder=0)


    for group in motif_indices:
        for start in np.atleast_1d(group):
            start = int(np.atleast_1d(start)[0])
            plt.axvspan(
                start,
                start + window_size,
                color="orange",
                alpha=0.3,
            )

    plt.title("Multi-dimensional Matrix Profile")
    plt.xlabel("Index")
    plt.ylabel("Dimension")
    plt.tight_layout()
    plt.show()

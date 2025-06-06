import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns


def matrix_profiles_to_array(matrix_profiles):
    """Convertit une liste de DataFrames matrix profile (une par capteur) en une matrice 2D numpy."""
    # On suppose que chaque DataFrame a la colonne 'value'
    arr = np.vstack([df['value'].values for df in matrix_profiles])
    return arr


def plot_matrix_profiles_heatmap(matrix_profiles, time_index=None, capteur_labels=None, figsize=(18, 8), cmap='viridis'):
    """
    Affiche une heatmap de tous les matrix profiles (ex : chaque capteur en ligne, temps en colonne).
    
    Args:
        matrix_profiles (list of pd.DataFrame): Liste de matrix profiles (chaque capteur).
        time_index (pd.DatetimeIndex, optional): Pour les labels de l'axe des X (temps).
        capteur_labels (list of str, optional): Labels pour chaque capteur.
        figsize (tuple): Taille du graphique.
        cmap (str): Palette de couleur (essaye 'viridis', 'magma', 'hot', 'cubehelix', etc.).
    """
    
    arr = matrix_profiles_to_array(matrix_profiles)
    arr_norm = (arr - arr.mean(axis=1, keepdims=True)) / (arr.std(axis=1, keepdims=True) + 1e-8)  # z-score par capteur

    plt.figure(figsize=(20, 10))
    sns.heatmap(arr_norm, cmap="magma", cbar=True)
    plt.title("Matrix Profiles Heatmap (normalized)")
    plt.ylabel("Capteur / Série")
    plt.xlabel("Temps")
    # Ajustement ticks X si besoin
    n = arr.shape[1]
    tick_positions = np.linspace(0, n-1, 8, dtype=int)
    plt.xticks(tick_positions + 0.5, [str(matrix_profiles[0].index[i].date()) for i in tick_positions], rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_matrix_profiles(matrix_profiles, labels=None, column="value", figsize=(14, 5)):
    """
    Plot one or several matrix profiles (value vs. index/timestamp).
    
    Args:
        matrix_profiles (pd.DataFrame or list of pd.DataFrame): Input(s) to plot.
        labels (list of str, optional): Legend labels for each series.
        column (str): The column to plot (default: 'value').
        figsize (tuple): Figure size.
    """
    # Handle single DataFrame input
    if isinstance(matrix_profiles, pd.DataFrame):
        dfs = [matrix_profiles]
    else:
        dfs = list(matrix_profiles)
    
    n = len(dfs)
    # Generate generic labels if none provided
    if labels is None:
        labels = [f"Profile {i+1}" for i in range(n)]
    elif len(labels) != n:
        raise ValueError("`labels` must match the number of matrix profiles.")
    
    color_list = plt.cm.get_cmap('tab20', n) if n > 10 else plt.get_cmap('tab10')
    color_iter = (color_list(i) for i in range(n))
    
    # Plot all matrix profiles on the same plot
    plt.figure(figsize=figsize)
    for df_, label, color in zip(dfs, labels, color_iter):
        df_ = df_.copy()
        if 'timestamp' in df_.columns:
            df_.index = pd.to_datetime(df_['timestamp'], errors='coerce')
            df_ = df_.drop(columns=['timestamp'])
        plt.plot(df_.index, df_[column], label=label, color=color)
    plt.title("Matrix Profile(s) Summary")
    plt.xlabel("Timestamp")
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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

def plot_aligned_motifs(
    aligned_segments: np.ndarray,
    title: str = "Aligned Motifs"
):
    window_size = aligned_segments.shape[1]
    t = np.arange(window_size)
    mean_curve = aligned_segments.mean(axis=0)
    # Trouve le segment le plus proche de la moyenne (méd-oïde)
    distances = np.linalg.norm(aligned_segments - mean_curve, axis=1)
    medoid_idx = np.argmin(distances)
    medoid_curve = aligned_segments[medoid_idx]

    plt.figure(figsize=(10, 5))
    for i, seg in enumerate(aligned_segments):
        plt.plot(t, seg, alpha=0.4, lw=1.5, label=f"Motif {i+1}" if i < 10 else None)
    # Affiche le motif méd-oïde en noir épais
    plt.plot(t, medoid_curve, color='black', linewidth=1, label="Most representative motif")
    plt.fill_between(
        t,
        aligned_segments.mean(axis=0) - aligned_segments.std(axis=0),
        aligned_segments.mean(axis=0) + aligned_segments.std(axis=0),
        color='gray', alpha=0.2, label='Std. dev.'
    )
    plt.title(title)
    plt.xlabel("Step in motif window (aligned)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_patterns_and_discords(df, result, column='value', figsize=(12, 6)):
    """
    Plot the original signal with identified motifs (affichés à partir de leur indice de début),
    les discordes en lignes verticales, et le Matrix Profile centré.
    En sous-figure : tous les motifs alignés par pattern avec la médoïde en noir.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    motif_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    n_patterns = len(result['patterns'])
    window_size = result['window_size']
    half = window_size // 2

    fig, axs = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*1.6), gridspec_kw={'height_ratios': [2, n_patterns]})

    # -- SUBPLOT 1 : Signal, motifs, discords, MP --
    ax1 = axs[0]

    # Signal original
    ax1.plot(df.index, df[column], label='Original Signal', color='black', alpha=0.5)
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel(column)

    # Matrix Profile (centré !)
    matrix_profile_values = result["matrix_profile"]['value']
    window_size = result["window_size"]
    mp_len = len(matrix_profile_values)
    # Indices X du MP centré
    center_indices = np.arange(mp_len)
    ax2 = ax1.twinx()
    ax2.plot(df.index[center_indices], matrix_profile_values, label='Matrix Profile', color='blue', alpha=0.6)
    ax2.set_ylabel('Matrix Profile')

    # Motifs détectés (indice de début !)
    for i, pattern in enumerate(result['patterns']):
        color = motif_colors[i % len(motif_colors)]
        # Ici, on prend les indices de début des motifs (PAS les indices centrés)
        motif_debuts = pattern["motif_indices_debut"]
        for motif_idx, idx_debut in enumerate(motif_debuts):
            start = idx_debut
            end = idx_debut + window_size
            if start >= 0 and end <= len(df):
                ax1.plot(df.index[start:end],
                         df[column].iloc[start:end],
                         color=color,
                         label=f"{pattern['pattern_label']} motif {motif_idx+1}" if motif_idx == 0 else "",
                         linewidth=2)
                # Début, centre, fin
                ax1.axvline(df.index[start], color='green', linestyle='--', linewidth=2)
                ax1.axvline(df.index[start + window_size // 2], color='purple', linestyle=':', linewidth=1)
                ax1.axvline(df.index[end-1], color='red', linestyle='--', linewidth=1)

    # Discords en lignes verticales
    for discord in result['discord_indices']:
        if discord < len(df):
            ax1.axvline(df.index[discord], color='red', linestyle='-', linewidth=2)

    ax1.set_title("Motifs (début de fenêtre), Discords, and Matrix Profile Detected")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    ax1.grid(True)

    # -- SUBPLOT 2 : Les motifs alignés par pattern --
    ax_motif = axs[1]
    motif_len = result['patterns'][0]['aligned_motifs'].shape[1] if n_patterns else 0
    motif_x = np.arange(motif_len)
    legend_shown = [False] * n_patterns

    for i, pattern in enumerate(result['patterns']):
        color = motif_colors[i % len(motif_colors)]
        motifs = pattern['aligned_motifs']
        medoid_idx = np.argmin([
            np.sum([np.linalg.norm(motifs[j] - motifs[k]) for k in range(len(motifs))])
            for j in range(len(motifs))
        ])
        for j, motif in enumerate(motifs):
            ax_motif.plot(motif_x, motif, color=color, alpha=0.4, linewidth=1,
                          label=f"{pattern['pattern_label']} motifs" if not legend_shown[i] and j == 0 else None)
            legend_shown[i] = True
        ax_motif.plot(motif_x, motifs[medoid_idx], color='black', linewidth=3, label=f"{pattern['pattern_label']} medoid")

    ax_motif.set_title("All motifs per pattern (médoïde en noir)")
    ax_motif.set_xlabel("Relative index in window")
    ax_motif.set_ylabel("value")
    ax_motif.legend()
    ax_motif.grid(True)

    plt.tight_layout(pad=3.0)
    plt.show()

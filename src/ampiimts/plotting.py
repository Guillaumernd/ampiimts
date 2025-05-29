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

    # 2) Separate subplots for each DataFrame
    fig2, axes = plt.subplots(nrows=n, ncols=1, figsize=(figsize_per_plot[0], figsize_per_plot[1]*n))
    if n == 1:
        axes = [axes]
    for ax, df_, label, color in zip(axes, dfs, labels, main_colors):
        if 'timestamp' in df_.columns:
            df_.index = pd.to_datetime(df_['timestamp'], errors='coerce')
            df_ = df_.drop(columns=['timestamp'])
        ax.plot(df_.index, df_[column], label=label, linewidth=1.5, color=color)
        ax.set_title(label)
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
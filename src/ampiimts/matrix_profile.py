"""Matrix profile computation helpers."""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import stumpy as sp
import os
os.environ["NUMBA_NUM_THREADS"] = "8"

from .motif_pattern import (
    discover_patterns_stumpy_mixed,
    discover_patterns_mstump_mixed,
)


def matrix_profile_process(
    df_o: pd.DataFrame,
    window_size: Optional[int] = None,
    column: str | None = None,
    max_motifs: int = 3,
    discord_top_pct: float = 0.04,
    max_matches: int = 10,
    cluster:bool = False,
    motif:bool=False,

) -> dict:
    """Return matrix profile related data for a single DataFrame."""

    # Copy to avoid mutating the original data
    df = df_o.copy()

    # Drop timestamp column if present, computation works on numeric values only
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    # Retrieve window size from attributes if not provided
    if window_size is None and "m" in df.attrs:
        window_size = df.attrs["m"]
    if window_size is None:
        raise RuntimeError("Critical error: window_size not provided")

    # Ensure all columns are numeric
    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
        raise ValueError(
            "All columns must be numeric for matrix profile computation."
        )
    
    # ==== UNIVARIATE ====
    if df.shape[1] == 1:
        try:
                
            # Delegate to motif discovery helper for one variable
            return discover_patterns_stumpy_mixed(
                df,
                window_size,
                max_motifs=max_motifs,
                discord_top_pct=discord_top_pct,
                max_matches=max_matches,
            )
        except ValueError as e:
            print(f"[MatrixProfile Warning] Échec du calcul avec fenêtre {window_size} → {e}")
            return None



    # ==== MULTIVARIATE ====
    try:
        return discover_patterns_mstump_mixed(
            df,
            window_size,
            max_motifs=max_motifs,
            discord_top_pct=discord_top_pct,
            max_matches=max_matches,
            cluster=cluster,
            motif=motif,
        )
    except ValueError as e:
        print(f"[MatrixProfile Warning] Échec du calcul avec fenêtre {window_size} → {e}")
        return None

def matrix_profile(
    df_o: Union[pd.DataFrame, List[pd.DataFrame], List[List[pd.DataFrame]]],
    window_size: Optional[int] = None,
    n_jobs: int = 4,
    column: str | None = None,
    max_motifs: int = 5,
    discord_top_pct: float = 0.04,
    max_matches: int = 10,
    cluster:bool = False,
    motif:bool =False,

) -> Union[dict, List[dict], List[List[dict]]]:
    """Compute the matrix profile for one or several DataFrames."""
        
    if df_o is None or (isinstance(df_o, list) and all(x is None for x in df_o)):
        return None

    if isinstance(df_o, pd.DataFrame):
        # Un seul DataFrame
        if column is None and len(df_o.columns) == 1:
            column = df_o.columns[0]
        if window_size is None and "m" in df_o.attrs:
            window_size = df_o.attrs["m"]

        return matrix_profile_process(
            df_o,
            window_size=window_size,
            column=column,
            max_motifs=max_motifs,
            discord_top_pct=discord_top_pct,
            max_matches=max_matches,
            cluster=cluster,
        )

    elif isinstance(df_o, list) and all(isinstance(x, pd.DataFrame) for x in df_o):
        # Liste plate de DataFrames
        return [
            matrix_profile_process(
                df,
                window_size=window_size,
                column=column,
                max_motifs=max_motifs,
                discord_top_pct=discord_top_pct,
                max_matches=max_matches,
                cluster=cluster,
                motif=motif,
            )
            for df in df_o
        ]

    elif isinstance(df_o, list) and all(isinstance(x, list) for x in df_o):
        # Liste de listes de DataFrames (cas avec cluster=True et plusieurs séries)
        return [
            [
                matrix_profile_process(
                    df,
                    window_size=window_size,
                    column=column,
                    max_motifs=max_motifs,
                    discord_top_pct=discord_top_pct,
                    max_matches=max_matches,
                    cluster=cluster,
                    motif=motif,
                )
                for df in sublist
            ]
            for sublist in df_o
        ]

    else:
        raise TypeError("df must be a DataFrame, a list of DataFrames, or a list of lists of DataFrames")

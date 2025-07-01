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
    df: pd.DataFrame,
    window_size: Optional[int] = None,
    max_motifs: int = 3,
    discord_top_pct: float = 0.04,
    max_matches: int = 10,
    cluster:bool = False,
    motif:bool=False,
    min_mdl_ratio: float = 0.25,
) -> dict:
    """Return matrix profile related data for a single DataFrame."""

    # Copy to avoid mutating the original data
    df = df.copy()

    # Drop timestamp column if present, computation works on numeric values only
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    # Retrieve window size from attributes if not provided
    window_size = df.attrs["m"]
 
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
            min_mdl_ratio=min_mdl_ratio,
        )
    except ValueError as e:
        print(f"[MatrixProfile Warning] Échec du calcul avec fenêtre {window_size} → {e}")
        return None

def matrix_profile(
    data: Union[pd.DataFrame, List[pd.DataFrame], List[List[pd.DataFrame]]],
    n_jobs: int = 4,
    max_motifs: int = 5,
    discord_top_pct: float = 0.04,
    max_matches: int = 10,
    cluster:bool = False,
    motif:bool =False,
    min_mdl_ratio: float = 0.25,
) -> Union[dict, List[dict], List[List[dict]]]:
    """Compute the matrix profile for one or several DataFrames."""
        
    if data is None or (isinstance(data, list) and all(x is None for x in data)):
        return None

    if isinstance(data, pd.DataFrame):
        # Un seul DataFrame
        df = data
        window_size = df.attrs["m"]

        return matrix_profile_process(
            df,
            window_size=window_size,
            max_motifs=max_motifs,
            discord_top_pct=discord_top_pct,
            max_matches=max_matches,
            cluster=cluster,
        )

    elif isinstance(data, list) and all(isinstance(x, pd.DataFrame) for x in data):
        pds = data
        # Liste plate de DataFrames
        return [
            matrix_profile_process(
                df,
                max_motifs=max_motifs,
                discord_top_pct=discord_top_pct,
                max_matches=max_matches,
                cluster=cluster,
                motif=motif,
                min_mdl_ratio=min_mdl_ratio,
            )
            for df in pds
        ]

    else:
        raise TypeError("df must be a DataFrame, a list of DataFrames, or a list of lists of DataFrames")

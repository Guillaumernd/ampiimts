import numpy as np
import pandas as pd
from numba import njit
import ruptures as rpt
import faiss
from tslearn.metrics import soft_dtw

def interpolate(
    df: pd.DataFrame, 
    gap_multiplier: float = 15
    ) -> pd.DataFrame:
    """
    Interpolates a multivariate, irregular time series DataFrame onto a regular time grid.
    - Infers base frequency.
    - Only fills small gaps (less than gap_multiplier * base frequency).
    - Leaves large gaps as NaN.
    
    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.
        gap_multiplier (float): Maximum gap (in multiples of base freq) to interpolate.
    
    Returns:
        pd.DataFrame: DataFrame resampled on a regular grid, with only small gaps interpolated.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.drop(columns=['timestamp'])
        else:
            df.index = pd.to_datetime(df.index, errors='coerce')
    df.index = df.index.tz_localize(None)
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    idx = df.index.unique().sort_values()
    if len(idx) < 2:
        raise ValueError("Not enough points to estimate frequency.")
    inferred = pd.infer_freq(idx)
    if inferred:
        # If inferred is a single character like "T", "S", "H", add "1" in front
        if len(inferred) == 1 or (len(inferred) == 2 and inferred[0] == "W"):
            inferred = "1" + inferred
        try:
            freq = pd.to_timedelta(inferred)
        except ValueError:
            # fallback to median diff in case of weird infer_freq
            freq = idx.to_series().diff().median()
    else:
        freq = idx.to_series().diff().median()

    max_gap = freq * gap_multiplier

    full_idx = pd.date_range(start=idx.min(), end=idx.max(), freq=freq)
    union_idx = idx.union(full_idx)
    df_union = df.reindex(union_idx)
    if 'timestamp' in df_union.columns:
        df_union = df_union.drop(columns=['timestamp'])
    df_union = df_union.infer_objects(copy=False)
    diffs = idx.to_series().diff()
    seg_ids = (diffs > max_gap).cumsum()
    seg_index = pd.Series(seg_ids.values, index=idx)
    seg_union = seg_index.reindex(union_idx, method='ffill').fillna(0).astype(int)
    seg_union = seg_union.infer_objects(copy=False)
    df_interp = (df_union.groupby(seg_union, group_keys=False).apply(lambda seg: seg.interpolate(method='time', limit_direction='both')))

    df_out = df_interp.reindex(full_idx)
    df_out.index.name = 'timestamp'

    gap_mask = pd.Series(False, index=df_out.index)
    for prev, nxt in zip(idx[:-1], idx[1:]):
        if nxt - prev > max_gap:
            # Marque tous les points entre prev (exclu) et nxt (inclus) comme gap
            in_gap = (df_out.index > prev) & (df_out.index <= nxt)
            gap_mask |= in_gap

    # Met à NaN partout où il y a un gap
    df_out.loc[gap_mask, :] = np.nan

    return df_out

@njit #comment njit for coverage
def _compute_aswn(
    values: np.ndarray, 
    window_size: int, 
    min_std: float, 
    min_valid_ratio: float
    ) -> np.ndarray:
    """
    Fast local window normalization (ASWN) with Numba.
    
    Args:
        values (np.ndarray): Input 1D array.
        window_size (int): Sliding window size.
        min_std (float): Minimum allowed standard deviation.
        min_valid_ratio (float): Min ratio of valid (non-NaN) values.
    
    Returns:
        np.ndarray: Normalized array, same shape as input.
    """
    n = len(values)
    half = window_size // 2
    result = np.empty(n)
    result[:] = np.nan
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        count_valid = 0
        sum_valid = 0.0
        sumsq_valid = 0.0
        for j in range(start, end):
            v = values[j]
            if not np.isnan(v):
                count_valid += 1
                sum_valid += v
                sumsq_valid += v * v
        total = end - start
        if count_valid / total >= min_valid_ratio and count_valid > 1:
            mean = sum_valid / count_valid
            var = (sumsq_valid / count_valid) - (mean * mean)
            std = np.sqrt(var)
            if std < min_std:
                std = min_std
            v = values[i]
            result[i] = (v - mean) / std if not np.isnan(v) else np.nan
        else:
            result[i] = np.nan
    return result

def find_optimal_m(
    signal: np.ndarray,
    model: str = "l2",
    ):
    """
    Finds the optimal window size m for a time series using changepoint detection and plateau detection.

    Args:
        signal (np.ndarray): The time series to segment.
        model (str): Ruptures model used for segmentation ("l2", "l1", etc.).
    Returns:
        m_opt (int): The optimal window size (median segment length on the first plateau).
    """

    # if pen is inputed by user
    duration_days = (signal.index.max() - signal.index.min()).total_seconds() / (24 * 3600)
    duration_days = max(duration_days, 1)
    pen = 3 * np.log(duration_days)
    bkps = rpt.Pelt(model=model).fit(signal.values).predict(pen=pen)   
    segment_lengths = np.diff([0] + bkps)
    print(f"window size pen = {round(np.median(segment_lengths))}")
    print(f"Window size per day = {pd.Timedelta('1D') / signal.index.to_series().diff().median()}")
    m_opt = max(int(np.median(segment_lengths)), pd.Timedelta('1D') / signal.index.to_series().diff().median())
    print(f"Optimal m = {m_opt}")
    return m_opt

def aswn_with_trend(
    series: pd.Series, 
    window_size: int = 50, 
    min_std: float = 1e-2,
    min_valid_ratio: float = 0.8, 
    alpha: float = 0.65,
    ) -> pd.Series:
    """
    ASWN normalization with optional trend blending.
    
    Args:
        series (pd.Series): Input time series.
        window_size (int): Sliding window size.
        min_std (float): Minimum allowed standard deviation.
        min_valid_ratio (float): Min ratio of valid (non-NaN) values.
        alpha (float): Trend blending coefficient (0-1).
    
    Returns:
        pd.Series: Normalized series (same index).
    """
    values = series.values.astype(np.float64)
    normed = _compute_aswn(values, window_size, min_std, min_valid_ratio)
    result = pd.Series(normed, index=series.index)
    if alpha > 0:
        print(window_size)
        trend = series.rolling(window=window_size * 10, center=True, min_periods=1).mean()
        return (1 - alpha) * result + alpha * (series - trend)
    return result

def normalization(
    df: pd.DataFrame, 
    min_std: float = 1e-2,
    min_valid_ratio: float = 0.8, 
    alpha: float = 0.65,
    window_size: str = None,
    ) -> pd.DataFrame:
    """
    Applies ASWN normalization (with trend) to all numeric columns in a DataFrame.
    The window size is automatically computed from the median time delta.
    
    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.
        min_std (float): Minimum allowed standard deviation.
        min_valid_ratio (float): Min ratio of valid (non-NaN) values.
        alpha (float): Trend blending coefficient (0-1).
    
    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.sort_index()

    if window_size == None:
        window_sizes = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                m = find_optimal_m(df[col].dropna())
                print(m)
                window_sizes.append(m)
        window_size = int(round(np.mean(window_sizes)))
    elif isinstance(window_size, str):
        window_size = int(pd.Timedelta(window_size) / df.index.to_series().diff().median())
    else: 
        raise RuntimeError("Window size isn't a str (e.g. : 1D, 1h, 1S etc...)")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = aswn_with_trend(df[col], window_size, min_std, min_valid_ratio, alpha)
    df.attrs['m'] = window_size 
    return df


def define_m_using_clustering( 
    df : pd.DataFrame,
    gap_multiplier : int = 15,
    gamma : int = 1,
    k : int = 3,
    window_sizes : list = ['0,001s', '0,01s', '0,1s', '1s', '15s', '30s',
                          '1m', '150s', '5m', '450s', '10m', '15m', '30m', '45m',
                          '1h', '2h', '3h', '4h', '5h', '6h', '7h',
                          '8h', '9h', '10h', '15h', '24h', '48h', '72h',
                          '168h'
                          ],
    ) -> list[tuple[int, str, float]]:
    """
    Determines the best window sizes for motif extraction in a time series using clustering.

    This function applies interpolation and normalization to the input DataFrame, then extracts sliding segments for a range of window sizes.
    Each segment is vectorized, and the FAISS library is used to accelerate nearest-neighbor searches. The similarity between segments is
    computed using soft-DTW. Clustering is then used to identify the window sizes that yield the most stable and repetitive motifs.

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.
        gap_multiplier (float): Threshold for maximum gap to interpolate (as a multiple of base frequency).
        gamma (float): Soft-DTW gamma parameter to control sensitivity to temporal distortion.
        k (int): Number of top window sizes to return.
        window_sizes (list of str): List of window sizes to test (e.g., ['1m', '1h', '24h']).

    Returns:
        results (list of tuples): Each tuple is (window_size_in_points, window_size_as_timedelta, stability_score), sorted by stability_score (ascending).
        The k best window sizes are returned.
    """
    #interpolate dataframe
    df = interpolate(df, gap_multiplier=gap_multiplier)
    #regular frequence of df
    freq = df.index.to_series().diff().median()
    #window_size in points, we filter the possible window_size
    window_sizes = [(int(pd.Timedelta(ws) / freq), ws) for ws in window_sizes if (pd.Timedelta(ws) >= freq and int(pd.Timedelta(ws) / freq) >= 10 and int(pd.Timedelta(ws) / freq) < len(df)//4)]

    # list of tuples to return
    result_nearest_neighbours = []

    # iteration on each numeric column of the dataframe
    for col in [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]:
        #serie to analyse
        values = df[col].values
        for ws_pts, ws_str in window_sizes:
            # Skip window sizes larger than the full series
            if ws_pts > len(values): continue

            # Extract all sliding window segments of size ws_pts from the time series
            segments = np.lib.stride_tricks.sliding_window_view(values, ws_pts)
            n = segments.shape[0]
            
            # Skip if there are less than 2 segments to compare
            if n < 2: continue

            # --- (Optional) Normalize each segment (z-score) to remove mean and scale to unit variance ---
            segments = (segments - segments.mean(axis=1, keepdims=True)) / (segments.std(axis=1, keepdims=True) + 1e-8)

            # --- FAISS nearest neighbor search ---
            # Create a FAISS index for L2 (Euclidean) distance, with dimension = window size
            index = faiss.IndexFlatL2(ws_pts)
            # Add all segments to the FAISS index as float32
            index.add(segments.astype(np.float32))
            # For each segment, find the 2 nearest neighbors (itself and its closest other segment)
            Distance_neighbours, Index_neighbours = index.search(segments.astype(np.float32), 2)
            # Take the distance to the nearest neighbor (excluding itself, which is at distance 0)
            nearest_distance = Distance_neighbours[:, 1]

            # --- Main score: median nearest neighbor distance, normalized by window size ---
            score_nearest_neighbour = np.median(nearest_distance) / ws_pts

            # --- Density score: fraction of segments with a very close neighbor ---
            # Threshold: 1.5x the normalized median neighbor distance
            threshold = score_nearest_neighbour * 1.5
            density_score = (nearest_distance < threshold).sum() / n

            # --- Store all scores for this window size ---
            result_nearest_neighbours.append(
                (ws_pts, ws_str, score_nearest_neighbour, density_score)
            )


            # # ---- Soft-DTW ----
            # # sampling if more of 50 segments
            # if n> 50 : 
            #     idx = np.random.choice(n, 50, replace=False)
            # else:
            #     #list of n
            #     idx =  np.arange(n)
            # sdtw_distance = []
            # for i in idx:
            #     #Get neighbours with soft_DTW to take in account temporel distorsion
            #     Distance_neighbours = [soft_dtw(segments[i], segments[j], gamma=gamma) for j in range(n) if j != i]

            #     if len(Distance_neighbours) > 10: 
            #         #add only ten best neighbours distance and calculate median of the segments
            #         sdtw_distance.append(np.mean(np.partition(Distance_neighbours, 10)[:10]))
            #     elif Distance_neighbours:
            #         sdtw_distance.append(np.mean(Distance_neighbours))
            # if sdtw_distance:
            #     #get score  of the median window_size
            #     result_nearest_neighbours = ((ws_pts, ws_str, np.median(sdtw_distance)))

    #return the best window_sizes
    final_result = sorted(result_nearest_neighbours, key=lambda x: x[2])[:k]

    return final_result

def pre_processed(
    df: pd.DataFrame,
    gap_multiplier: float = 15,
    min_std: float = 1e-2,
    min_valid_ratio: float = 0.8, 
    alpha: float = 0.65,
    window_size: str = None,
    ) -> pd.DataFrame:
    """
    Runs a full preprocessing pipeline on a pandas DataFrame time series:
      - Interpolates small gaps (large gaps remain as NaN)
      - Applies local ASWN normalization (with optional trend blending)
    
    Args:
        df (pd.DataFrame): Input data, must have a DatetimeIndex.
        gap_multiplier (float): Threshold for maximum gap to interpolate (as a multiple of base frequency).
        min_std (float): Minimum allowed standard deviation in the normalization window.
        min_valid_ratio (float): Minimum fraction of valid points in each window.
        alpha (float): Blending coefficient for trend removal (0=no trend, 1=only trend-removed).
    
    Returns:
        pd.DataFrame: The preprocessed DataFrame (regular time grid, interpolated, normalized).
    """
    # Ensure the input is a DataFrame; fail early with a clear error if not
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")
    
    # Defensive copy to avoid mutating user input
    df_preprocessed = df.copy()
    
    # Step 1: Interpolate small gaps on a regular grid, leave large gaps as NaN
    df_preprocessed = interpolate(df_preprocessed, gap_multiplier)
    
    # Step 2: Apply local normalization to all numeric columns (ASWN, with trend blending if alpha>0)
    df_preprocessed = normalization(df_preprocessed, min_std=min_std, min_valid_ratio=min_valid_ratio, alpha=alpha, window_size=window_size)  
    
    # Return the processed DataFrame
    return df_preprocessed


def missing_values(
    df: pd.DataFrame, 
    percent_missing: float = 0.2, 
    random_state: int = 1
    ) -> pd.DataFrame:
    """
    Simulate random missing data in all columns, including timestamp, in a vectorized way.

    Detects or creates a 'timestamp' column from the index or any datetime column.
    Applies a single boolean mask to assign NaN/NaT simultaneously across all columns.
    Sets 'timestamp' as the DataFrame index (as a DatetimeIndex), removing duplicate columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame; index may be DatetimeIndex or default.
    percent_missing : float, optional
        Fraction of rows to mark as missing (0 < percent_missing < 1). Default is 0.4.
    random_state : int, optional
        Seed for reproducibility. Default is 1.

    Returns
    -------
    pd.DataFrame
        Copy of input with missing values inserted, indexed by timestamp.
    """
    # 1. Input checks
    if not (0 < percent_missing < 1):
        raise ValueError("percent_missing must be between 0 and 1.")

    df_copy = df.copy()

    # 2. Ensure a 'timestamp' column of datetime type exists
    if isinstance(df_copy.index, pd.DatetimeIndex):
        name = df_copy.index.name or 'timestamp'
        df_copy = df_copy.reset_index().rename(columns={name: 'timestamp'})
    elif 'timestamp' in df_copy.columns and pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
        pass
    else:
        dt_cols = [c for c in df_copy.columns if pd.api.types.is_datetime64_any_dtype(df_copy[c])]
        if dt_cols:
            df_copy = df_copy.rename(columns={dt_cols[0]: 'timestamp'})
        else:
            df_copy = df_copy.reset_index().rename(columns={'index': 'timestamp'})

    # 3. Convert index to datetime (for safety)
    df_copy.index = pd.to_datetime(df_copy.index, errors='coerce')

    # 4. Prepare random mask for missingness
    rng = np.random.default_rng(random_state)
    mask = rng.random(df_copy.shape) < percent_missing

    # 5. Apply mask: pandas will choose NaN or NaT depending on dtype
    df_missing = df_copy.mask(mask)

    # 6. Remove duplicate columns, keeping the last 'timestamp'
    df_missing = df_missing.loc[:, ~df_missing.columns.duplicated(keep='last')]

    # 7. Set 'timestamp' as index and convert to DatetimeIndex
    df_missing = df_missing.set_index('timestamp')
    df_missing.index = pd.to_datetime(df_missing.index, errors='coerce')

    return df_missing
    

    

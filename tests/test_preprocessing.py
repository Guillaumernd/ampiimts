import numpy as np
import pandas as pd
import pytest

from swt import interpolate, _compute_aswn, aswn_with_trend, normalization, pre_processed


def test_interpolate_fills_small_gaps():
    """
    Test that interpolate fills small gaps and keeps original values.
    """
    idx = pd.date_range("2023-01-01", periods=10, freq="1min")
    df = pd.DataFrame({"value": np.arange(10)}, index=idx)
    df = df.drop(idx[5])  # Create a small gap

    result = interpolate(df, gap_multiplier=2)

    # The result index must be regular and complete
    assert len(result) == 10
    assert result.index.equals(idx)
    # Small gap is linearly interpolated
    expected = (df.loc[idx[4], "value"] + df.loc[idx[6], "value"]) / 2
    assert np.isclose(result.loc[idx[5], "value"], expected)
    # Values outside the gap are unchanged
    assert np.isclose(result.loc[idx[0], "value"], 0)
    assert np.isclose(result.loc[idx[9], "value"], 9)


def test_interpolate_preserves_large_gaps_as_nan():
    """
    Test that interpolate does not fill large gaps, leaving them as NaN (except possibly the gap's leftmost point).
    """
    idx = pd.date_range("2023-01-01", periods=12, freq="1min")
    df = pd.DataFrame({"value": np.arange(12)}, index=idx)
    df = df.drop(idx[5:9])  # Large gap: 4 min

    result = interpolate(df, gap_multiplier=2)

    # The leftmost point of the gap (idx[5]) may be interpolated (check docs/implementation!)
    # All *subsequent* points in the large gap must be NaN
    for i in range(6, 9):
        assert np.isnan(result.loc[idx[i], "value"])


def test_compute_aswn_handles_nan_and_normalization():
    """
    Test that _compute_aswn keeps NaNs and normalizes local windows.
    """
    values = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])
    window_size = 3
    min_std = 0.01
    min_valid_ratio = 0.5

    out = _compute_aswn(values, window_size, min_std, min_valid_ratio)
    # Output NaN at position 3
    assert np.isnan(out[3])
    # Output not NaN at position 1
    assert not np.isnan(out[1])
    # Normalized output (should be close to zero mean for central window)
    window = values[0:3]
    mean = np.nanmean(window)
    std = np.nanstd(window)
    expected = (values[1] - mean) / (std if std > min_std else min_std)
    assert np.isclose(out[1], expected, rtol=1e-5)


def test_aswn_with_trend_without_blending():
    """
    Test aswn_with_trend with alpha=0 returns local normalization only.
    """
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    window_size = 3

    normed = aswn_with_trend(series, window_size, alpha=0)

    # Compute manually for central value (index 2)
    window = series.iloc[1:4]
    mean = window.mean()
    std = window.std(ddof=0)
    expected = (series.iloc[2] - mean) / std
    assert np.isclose(normed.iloc[2], expected, rtol=1e-5)


def test_aswn_with_trend_with_blending():
    """
    Test aswn_with_trend blending with trend removal (alpha>0).
    """
    series = pd.Series(np.arange(10, dtype=float))
    window_size = 3
    alpha = 0.5

    out = aswn_with_trend(series, window_size, alpha=alpha)

    assert len(out) == len(series)
    # Output is a blend of normalized and detrended series
    normed = aswn_with_trend(series, window_size, alpha=0)
    trend = series.rolling(window=window_size * 10, center=True, min_periods=1).mean()
    manual = (1 - alpha) * normed + alpha * (series - trend)
    np.testing.assert_allclose(out, manual, rtol=1e-5, atol=1e-5)


def test_normalization_on_dataframe():
    """
    Test normalization applies to all numeric columns and leaves non-numeric untouched.
    """
    idx = pd.date_range("2023-01-01", periods=20, freq="1min")
    df = pd.DataFrame({
        "a": np.arange(20),
        "b": np.arange(20, 0, -1),
        "c": ["x"] * 20
    }, index=idx)

    normed = normalization(df)

    assert np.all(normed["a"].notna())
    assert np.all(normed["b"].notna())
    assert normed["c"].equals(df["c"])
    # Output columns 'a' and 'b' are still floats
    assert normed["a"].dtype.kind == "f"
    assert normed["b"].dtype.kind == "f"

def test_pre_processed_typical_case():
    """
    Test pre_processed on a simple DataFrame with a small gap.
    Should interpolate the gap and normalize the values.
    """
    idx = pd.date_range("2023-01-01", periods=6, freq="1min")
    df = pd.DataFrame({"signal": [1, 2, 3, np.nan, 5, 6]}, index=idx)
    df_out = pre_processed(df, gap_multiplier=2, min_valid_ratio=0.5, alpha=0)
    # Index is still complete and sorted
    assert isinstance(df_out, pd.DataFrame)
    assert df_out.index.equals(idx)
    # The previously missing value is now filled (not NaN)
    assert not np.isnan(df_out.loc[idx[3], "signal"])
    # Normalization applied: mean ~0 at center
    assert np.abs(df_out["signal"].mean()) < 1.0  # Should be roughly centered

def test_pre_processed_large_gap_stays_nan():
    """
    Test that a large gap is not interpolated (remains NaN after pre_processed).
    """
    idx = pd.date_range("2023-01-01", periods=7, freq="1min")
    df = pd.DataFrame({"s": [1, 2, 3, 6, 7]}, index=idx[[0,1,2,5,6]])
    df_out = pre_processed(df, gap_multiplier=1, min_valid_ratio=0.5, alpha=0)
    assert np.isnan(df_out.loc[idx[3], "s"])
    assert np.isnan(df_out.loc[idx[4], "s"])



def test_pre_processed_non_dataframe_input():
    """
    Test that a TypeError is raised if input is not a DataFrame.
    """
    with pytest.raises(TypeError):
        pre_processed([1, 2, 3])  # List, not DataFrame

def test_pre_processed_preserves_non_numeric():
    """
    Test that non-numeric columns are preserved (unchanged) in output.
    """
    idx = pd.date_range("2023-01-01", periods=5, freq="1min")
    df = pd.DataFrame({
        "num": [1, 2, np.nan, 4, 5],
        "txt": ["a", "b", "c", "d", "e"]
    }, index=idx)
    df_out = pre_processed(df, gap_multiplier=2, min_valid_ratio=0.5, alpha=0)
    # Non-numeric column is identical
    assert (df_out["txt"] == df["txt"]).all()

def test_pre_processed_empty_dataframe():
    """
    Test that an empty DataFrame raises a ValueError.
    """
    df = pd.DataFrame(columns=["x", "y"])
    with pytest.raises(ValueError, match="Not enough points"):
        pre_processed(df)

def test_interpolate_multicolumn_large_gap():
    """
    Test interpolate on a DataFrame with several columns and a large gap in the index.
    All columns should get NaN in the gap, regardless of their values.
    """
    idx = pd.date_range("2023-01-01", periods=8, freq="1min")
    # On enlève idx[4] et idx[5] pour faire un trou de 2 min
    used_idx = [0,1,2,3,6,7]
    df = pd.DataFrame({
        "a": [10, 20, 30, 40, 70, 80],
        "b": [1, np.nan, 3, 4, 7, 8],
        "c": [0, 0, 1, 1, 2, 2]
    }, index=idx[used_idx])
    df_out = interpolate(df, gap_multiplier=1)
    # Les points dans le gap doivent être NaN pour toutes les colonnes
    assert df_out.loc[idx[4], "a"] != df_out.loc[idx[4], "a"]  # NaN check
    assert df_out.loc[idx[4], "b"] != df_out.loc[idx[4], "b"]
    assert df_out.loc[idx[5], "c"] != df_out.loc[idx[5], "c"]

def test_normalization_multicolumn():
    """
    Test that normalization runs and produces output of the correct shape and index on multi-column DataFrame.
    """
    idx = pd.date_range("2023-01-01", periods=10, freq="1min")
    df = pd.DataFrame({
        "x": np.arange(10),
        "y": np.arange(10, 20)[::-1]
    }, index=idx)
    df_norm = normalization(df)
    assert df_norm.shape == df.shape
    assert np.all(df_norm.index == df.index)
    # Les colonnes doivent être normalisées, donc la somme des moyennes doit être proche de zéro
    assert abs(df_norm["x"].mean()) < 1e-8
    assert abs(df_norm["y"].mean()) < 1e-8

def test_pre_processed_multicolumn_gap():
    """
    Test pre_processed pipeline on a DataFrame with several columns and a gap in the index.
    """
    idx = pd.date_range("2023-01-01", periods=8, freq="1min")
    used_idx = [0,1,2,3,6,7]
    df = pd.DataFrame({
        "a": [10, 20, 30, 40, 70, 80],
        "b": [1, 2, 3, 4, 7, 8]
    }, index=idx[used_idx])
    df_out = pre_processed(df, gap_multiplier=1)
    # Les points dans le gap doivent être NaN dans toutes les colonnes
    assert np.isnan(df_out.loc[idx[4], "a"])
    assert np.isnan(df_out.loc[idx[4], "b"])

import numpy as np
import pytest
from ampiimts.pre_processed import compute_variances, normalize_segments, _compute_aswn


def test_compute_variances_basic():
    segs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    expected = np.array([2/3, 2/3, 2/3])
    result = compute_variances(segs)
    np.testing.assert_allclose(result, expected)

def test_compute_variances_with_nan():
    segs = np.array([[1, 2, 3], [np.nan, np.nan, np.nan], [7, 8, 9]], dtype=np.float32)
    result = compute_variances(segs)
    assert np.isnan(result[1])
    np.testing.assert_allclose(result[[0, 2]], [2/3, 2/3])

def test_normalize_segments_basic():
    segs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    result = normalize_segments(segs)
    assert result.shape == segs.shape
    assert np.allclose(np.mean(result, axis=1), 0, atol=1e-6)
    assert np.allclose(np.std(result, axis=1), 1, atol=1e-6)

def test_normalize_segments_constant_rows():
    segs = np.array([[5, 5, 5], [10, 10, 10]], dtype=np.float32)
    result = normalize_segments(segs)
    assert result.shape == segs.shape
    # Tolère les NaN ou 0 si std == 0
    assert np.all((np.isnan(result)) | (result == 0))

def test_compute_aswn_linear_trend():
    series = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    window_size = 3
    min_std = 1e-3
    min_valid_ratio = 0.5
    result = _compute_aswn(series, window_size, min_std, min_valid_ratio)
    assert result.shape == series.shape
    assert not np.isnan(result).any()

def test_compute_aswn_constant():
    series = np.ones(10, dtype=np.float32)
    window_size = 3
    min_std = 1e-6
    min_valid_ratio = 0.8
    result = _compute_aswn(series, window_size, min_std, min_valid_ratio)
    assert result.shape == series.shape
    # Le comportement actuel retourne 0 dans ce cas
    assert np.all(result == 0)


def test_compute_aswn_with_nan():
    series = np.array([1, 2, np.nan, 4, 5], dtype=np.float32)
    window_size = 3
    min_std = 1e-6
    min_valid_ratio = 0.3
    result = _compute_aswn(series, window_size, min_std, min_valid_ratio)
    assert result.shape == series.shape
    assert np.isnan(result[2])

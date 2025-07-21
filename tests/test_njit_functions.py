import pytest
import numpy as np

@pytest.fixture(autouse=True)
def disable_njit_locally(monkeypatch):
    import numba

    # Redéfinit njit localement dans ce fichier uniquement
    monkeypatch.setattr(numba, "njit", lambda *args, **kwargs: (lambda f: f) if not args else args[0])

    # Réimporte ici les fonctions à tester APRÈS le patch
    global compute_variances, normalize_segments, _compute_aswn
    import importlib
    pre_processed = importlib.import_module("src.ampiimts.pre_processed")
    importlib.reload(pre_processed)
    from src.ampiimts.pre_processed import compute_variances, normalize_segments, _compute_aswn



@pytest.fixture(autouse=True)
def disable_njit_locally(monkeypatch):
    """
    Fixture that disables `numba.njit` locally for this test file.

    This prevents JIT compilation for tested functions so they can be debugged and
    tested more easily in a pure Python environment. Other test files are unaffected.
    """
    import numba
    import importlib

    # Patch numba.njit to be a no-op decorator
    monkeypatch.setattr(numba, "njit", lambda *args, **kwargs: (lambda f: f) if not args else args[0])

    # Reload the target module after patching njit
    pre_processed = importlib.import_module("src.ampiimts.pre_processed")
    importlib.reload(pre_processed)

    # Re-import patched functions globally
    global compute_variances, normalize_segments, _compute_aswn
    from src.ampiimts.pre_processed import compute_variances, normalize_segments, _compute_aswn


def test_compute_variances_basic():
    """
    Test that `compute_variances` returns correct variance values
    for normal input with no missing data.
    """
    segs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    expected = np.array([2/3, 2/3, 2/3])
    result = compute_variances(segs)
    np.testing.assert_allclose(result, expected)


def test_compute_variances_with_nan():
    """
    Test that `compute_variances` returns NaN for rows containing only NaNs,
    and correct variances for the rest.
    """
    segs = np.array([[1, 2, 3], [np.nan, np.nan, np.nan], [7, 8, 9]], dtype=np.float32)
    result = compute_variances(segs)
    assert np.isnan(result[1])
    np.testing.assert_allclose(result[[0, 2]], [2/3, 2/3])


def test_normalize_segments_basic():
    """
    Test that `normalize_segments` returns segments with zero mean and unit variance
    when input contains standard numerical values.
    """
    segs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    result = normalize_segments(segs)
    assert result.shape == segs.shape
    assert np.allclose(np.mean(result, axis=1), 0, atol=1e-6)
    assert np.allclose(np.std(result, axis=1), 1, atol=1e-6)


def test_normalize_segments_constant_rows():
    """
    Test that `normalize_segments` returns NaNs or zeros for rows with constant values,
    which have zero standard deviation.
    """
    segs = np.array([[5, 5, 5], [10, 10, 10]], dtype=np.float32)
    result = normalize_segments(segs)
    assert result.shape == segs.shape
    assert np.all((np.isnan(result)) | (result == 0))


def test_compute_aswn_linear_trend():
    """
    Test that `_compute_aswn` returns a transformed signal of same shape,
    and handles a simple increasing trend correctly without NaNs.
    """
    series = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    window_size = 3
    min_std = 1e-3
    min_valid_ratio = 0.5
    result = _compute_aswn(series, window_size, min_std, min_valid_ratio)
    assert result.shape == series.shape
    assert not np.isnan(result).any()


def test_compute_aswn_constant():
    """
    Test that `_compute_aswn` handles a flat signal by returning an array of zeros.
    """
    series = np.ones(10, dtype=np.float32)
    window_size = 3
    min_std = 1e-6
    min_valid_ratio = 0.8
    result = _compute_aswn(series, window_size, min_std, min_valid_ratio)
    assert result.shape == series.shape
    assert np.all(result == 0)


def test_compute_aswn_with_nan():
    """
    Test that `_compute_aswn` preserves NaNs in the output where input contains NaNs.
    """
    series = np.array([1, 2, np.nan, 4, 5], dtype=np.float32)
    window_size = 3
    min_std = 1e-6
    min_valid_ratio = 0.3
    result = _compute_aswn(series, window_size, min_std, min_valid_ratio)
    assert result.shape == series.shape
    assert np.isnan(result[2])
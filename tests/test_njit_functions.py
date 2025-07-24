import pytest
import numpy as np

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
    global _compute_aswn
    from src.ampiimts.pre_processed import _compute_aswn


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
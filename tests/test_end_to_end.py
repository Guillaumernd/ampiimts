import pytest
import pandas as pd
import numpy as np
from ampiimts import ampiimts, pre_processed

def reference_unidimensional_dataframe(window_size):
    np.random.seed(23)  # reproductibilité

    # Crée un motif de taille fixe
    motif = np.random.randint(0, 10, size=window_size).tolist()
    repeated_motifs = motif * 29

    n_discordes = (len(repeated_motifs)) // 25
    discordes = []
    discordes = np.random.randint(200, 300, size=n_discordes).tolist()
    full_signal = repeated_motifs.copy()
    actual_discordes = 0
    for i, val in enumerate(discordes):
        insert_index = i * 25
        if insert_index != 0 and insert_index < len(full_signal):
            full_signal[insert_index] = val
            actual_discordes += 1

    timestamps = pd.date_range("2022-01-01", periods=len(full_signal), freq="min")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "value": full_signal
    })
    return df, actual_discordes


def test_univariate_with_nan():
    """Unidimensional DataFrame without NaN values."""
    window_size = 60
    df, n_discordes = reference_unidimensional_dataframe(window_size)
    # Appel de ampiimts sans valeurs manquantes (cluster désactivé, impression unidimensionnelle activée)
    interpolated, normalized, result = ampiimts(df, 
                                                cluster=False, 
                                                window_size=window_size, 
                                                visualize=False, 
                                                )
    # Vérifications de base sur le type de sortie
    assert isinstance(interpolated, pd.DataFrame)
    assert isinstance(normalized, pd.DataFrame)
    assert isinstance(result, dict) and result is not None
    assert isinstance(result["window_size"][1], int) and result["window_size"] is not None
    print(result["window_size"][0])
    assert result["window_size"][1] == window_size
    # Le DataFrame interpolé doit conserver le timestamp en index et contenir la colonne de valeur
    assert interpolated.index.name == "timestamp"
    assert "value" in interpolated.columns
    # n_discorde doit être égale aux nombres de Nan insérés 
    assert interpolated.isna().sum().sum() == n_discordes
    # Il ne doit plus rester aucun Nan avec la réinterpolation courte de normalization
    assert normalized.isna().sum().sum() == 0


def test_univariate_window_too_large():
    df, _ = reference_unidimensional_dataframe(60)
    with pytest.raises(Exception):
        ampiimts(df, cluster=False, window_size=len(df) // 2 + 1)

def test_univariate_quasi_constant():
    timestamps = pd.date_range("2022-01-01", periods=500, freq="min")
    df = pd.DataFrame({"timestamp": timestamps, "value": [1.001] * 500})
    interpolated, normalized, _ = ampiimts(df, cluster=False, window_size=30)
    assert normalized is None



def test_univariate_all_nan():
    timestamps = pd.date_range("2022-01-01", periods=100, freq="min")
    df = pd.DataFrame({"timestamp": timestamps, "value": [np.nan]*100})
    interpolated, normalized, result = ampiimts(df, cluster=False, window_size=30)
    assert interpolated is None
    assert normalized is None


def test_univariate_not_enough_valid_windows():
    df, _ = reference_unidimensional_dataframe(60)
    # Injecte 20 NaN toutes les 50 lignes
    for start in range(0, len(df), 50):
        df.loc[df.index[start:start + 45], "value"] = np.nan
    interpolated, normalized, _ = ampiimts(df, cluster=False, window_size=60)
    assert normalized is None


def test_univariate_without_trend_blending():
    df, _ = reference_unidimensional_dataframe(60)
    interpolated, normalized, _ = pre_processed(df, alpha=0.0)
    mean_val = normalized["value"].mean()
    assert abs(mean_val) < 0.1  # la normalisation doit centrer


def test_univariate_no_smart_interpolation():
    df, _ = reference_unidimensional_dataframe(60)
    interpolated, normalized, _ = pre_processed(df, smart_interpolation=False)
    assert isinstance(normalized, pd.DataFrame)
    assert not normalized.isna().any().any()


def test_univariate_auto_window_selection():
    df, _ = reference_unidimensional_dataframe(60)
    interpolated, normalized, _ = pre_processed(df, window_size=None)
    assert "m" in normalized.attrs
    assert isinstance(normalized.attrs["m"][1], int)



def reference_multivariate_dataframe(window_size=60, n_sensors=3):
    np.random.seed(42)
    motif = np.random.randint(0, 10, size=window_size).tolist()
    base_signal = motif * 30
    timestamps = pd.date_range("2022-01-01", periods=len(base_signal), freq="min")

    df = pd.DataFrame({"timestamp": timestamps})
    for i in range(n_sensors):
        signal = base_signal.copy()
        noise = np.random.normal(scale=0.5, size=len(signal))
        if i == 0:
            discordes = np.random.randint(200, 300, size=len(signal) // 25)
            for j, val in enumerate(discordes):
                if j * 25 < len(signal):
                    signal[j * 25] = val
        signal = np.array(signal) + noise
        df[f"sensor_{i}"] = signal
    return df


def test_multivariate_basic_pipeline():
    df = reference_multivariate_dataframe()
    interpolated, normalized, _ = ampiimts(df, cluster=False, window_size=60)
    assert isinstance(interpolated, pd.DataFrame)
    assert isinstance(normalized, pd.DataFrame)
    assert interpolated.index.name == "timestamp"
    assert all(col.startswith("sensor_") for col in interpolated.columns)
    assert normalized.isna().sum().sum() == 0


def test_multivariate_all_nan_column():
    df = reference_multivariate_dataframe()
    df["sensor_1"] = np.nan
    interpolated, normalized, _ = ampiimts(df, cluster=False, window_size=60)
    assert "sensor_1" not in interpolated.columns or interpolated["sensor_1"].isna().all()
    assert "sensor_1" not in normalized.columns


def test_multivariate_constant_column():
    df = reference_multivariate_dataframe()
    df["sensor_2"] = 1.234  # constante
    interpolated, normalized, _ = ampiimts(df, cluster=False, window_size=60)
    assert "sensor_2" not in normalized.columns  # doit être retirée


def test_multivariate_not_enough_valid_windows():
    df = reference_multivariate_dataframe()
    for col in df.columns:
        if col.startswith("sensor"):
            for start in range(0, len(df), 50):
                df.loc[df.index[start:start + 45], col] = np.nan
    interpolated, normalized, _ = ampiimts(df, cluster=False, window_size=60)
    assert normalized is None


def test_multivariate_auto_window_selection():
    df = reference_multivariate_dataframe()
    interpolated, normalized, _ = pre_processed(df, window_size=None)
    assert "m" in normalized.attrs
    assert isinstance(normalized.attrs["m"][1], int)


def test_multivariate_without_trend_blending():
    df = reference_multivariate_dataframe()
    interpolated, normalized, _ = pre_processed(df, alpha=0.0)
    for col in normalized.columns:
        mean_val = normalized[col].mean()
        assert abs(mean_val) < 0.1


def test_multivariate_no_smart_interpolation():
    df = reference_multivariate_dataframe()
    interpolated, normalized, _ = pre_processed(df, smart_interpolation=False)
    assert isinstance(normalized, pd.DataFrame)
    assert not normalized.isna().any().any()
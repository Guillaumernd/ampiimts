import pytest
import pandas as pd
from ampiimts import ampiimts
from ampiimts import define_m, discover_patterns_mstump_mixed
from pathlib import Path
import numpy as np 
import os 

# All possible boolean combinations for parameter testing
boolean_combinations = [
    (c, m, s, i)
    for c in [False, True]               # cluster
    for m in [False, True]               # motif
    for s in [False, True]               # most_stable_only
    for i in [False, True]               # smart_interpolation
]

boolean_combinations_2d = [
    (c, m)
    for c in [False, True]
    for m in [False, True]
]

@pytest.mark.parametrize("csv_path", [
    "tests/data/art_daily_jumpsup.csv",
])
@pytest.mark.parametrize("cluster, motif", boolean_combinations_2d)
@pytest.mark.parametrize("as_str_input", [True, False])  # True = passer la str, False = passer le DataFrame
def test_univariate_from_csv_or_df(csv_path, as_str_input, cluster, motif):
    """
    Test the ampiimts pipeline on both CSV paths (str) and preloaded DataFrame inputs.

    Parameters
    ----------
    csv_path : str
        Path to a CSV file containing a univariate time series.
    as_str_input : bool
        Whether to pass the path as str (to trigger internal CSV loading) or a preloaded DataFrame.
    cluster, motif, most_stable_only, smart_interpolation : bool
        Parameters passed to the ampiimts pipeline.
    """
    assert os.path.isfile(csv_path), f"Le fichier {csv_path} est introuvable"

    if as_str_input:
        data = csv_path  # ← déclenche le bloc "if isinstance(data, str) and os.path.isfile(data):"
    else:
        data = pd.read_csv(csv_path, parse_dates=["timestamp"])

    interpolated, normalized, result = ampiimts(
        data,
        cluster=cluster,
        motif=motif,
        most_stable_only=False,
        smart_interpolation=False,
        visualize=True,
        only_heat_map=False,
    )

    if interpolated is not None:
        assert isinstance(interpolated, pd.DataFrame)
    if normalized is not None:
        assert isinstance(normalized, pd.DataFrame)
    assert isinstance(result, (dict, type(None)))


def load_and_concat_csvs_from_folder(folder_path):
    """
    Load and concatenate all multivariate CSV files from a folder.

    Parameters
    ----------
    folder_path : str or Path
        Folder containing CSV files with a 'timestamp' column.

    Returns
    -------
    List[pd.DataFrame]
        A list of time-indexed DataFrames.
    """
    dfs = []
    for file in sorted(Path(folder_path).glob("*.csv")):
        df = pd.read_csv(file, parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        dfs.append(df)
    return dfs


@pytest.mark.parametrize("cluster, motif, most_stable_only, smart_interpolation", boolean_combinations)
def test_multivariate_from_air_bejin(cluster, motif, most_stable_only, smart_interpolation):
    """
    Test the ampiimts pipeline on multivariate sensor data (as folder input).

    Parameters
    ----------
    cluster, motif, most_stable_only, smart_interpolation : bool
        Parameters passed to the pipeline.

    Assertions
    ----------
    - If cluster is True, all returned values must be lists of DataFrames/dicts.
    - If cluster is False, returned values must be single DataFrames/dict.
    """
    folder = "tests/data/air_bejin"

    interpolated, normalized, result = ampiimts(
        folder,
        top_k_cluster=1,
        cluster=cluster,
        motif=motif,
        most_stable_only=most_stable_only,
        smart_interpolation=smart_interpolation,
        max_len=750,
        group_size=5,
        visualize=True,
        only_heat_map=False
    )

    if cluster:
        assert isinstance(interpolated, list)
        assert all(isinstance(df, pd.DataFrame) for df in interpolated)
        assert isinstance(normalized, list)
        assert all(isinstance(df, pd.DataFrame) for df in normalized)
        assert isinstance(result, list)
    else:
        assert isinstance(interpolated, pd.DataFrame)
        assert isinstance(normalized, pd.DataFrame)
        assert isinstance(result, dict)


@pytest.mark.parametrize("window_size", ["24h", 24])
def test_multivariate_with_dataframe_input(window_size):
    """
    Test the ampiimts pipeline on a list of multivariate DataFrames
    with different formats for the window size ("24h" vs 24).

    Parameters
    ----------
    window_size : str or int
        Window size passed to the pipeline, tested as both duration string and integer.

    Assertions
    ----------
    - Each output must be a list of DataFrames or dictionaries.
    - All lists must have the same length.
    - Window size is correctly parsed and applied.
    """
    folder = "tests/data/air_bejin"
    input_data = load_and_concat_csvs_from_folder(folder)

    interpolated, normalized, result = ampiimts(
        input_data,
        top_k_cluster=1,
        cluster=True,
        motif=True,
        window_size=window_size,
        most_stable_only=False,
        smart_interpolation=True,
        max_len=1000,
        visualize=False,
        display_info=True,
        only_heat_map=False,
    )

    assert isinstance(interpolated, list)
    assert all(isinstance(df, pd.DataFrame) for df in interpolated)
    assert isinstance(normalized, list)
    assert all(isinstance(df, pd.DataFrame) for df in normalized)
    assert isinstance(result, list)
    assert len(result) == len(interpolated) == len(normalized)
    assert result[0]["window_size"][1] == 24

@pytest.mark.parametrize("data", [False, True])
def test_multivariate_with_dataframe_nan_input(data):
    """
    Test the ampiimts pipeline on a list of multivariate DataFrames or one dataframes
    with NaNs injected every 30 points in each DataFrame.

    Parameters
    ----------
    window_size : str or int
        Window size passed to the pipeline.

    Assertions
    ----------
    - Each output must be a list (if clustering is enabled).
    - NaN injection doesn't crash the pipeline.
    """

    folder = "tests/data/air_bejin"
    input_data = load_and_concat_csvs_from_folder(folder)

    input_data_nan = []
    for df in input_data:
        df_copy = df.copy()
        for col in df_copy.columns:
            if col != "timestamp":
                i = 0
                while i + 20 <= len(df_copy):
                    df_copy.iloc[i:i+20, df_copy.columns.get_loc(col)] = np.nan
                    i += 30
        input_data_nan.append(df_copy)
    if not data:
        input_data_nan = input_data_nan[0]

    with pytest.raises(ValueError, match="too many Nans or are too noisy"):
        ampiimts(
            input_data_nan,
            top_k_cluster=1,
            cluster=True,
            motif=False,
            most_stable_only=False,
            smart_interpolation=True,
            max_len=1000,
            window_size=24,
            visualize=False,
            display_info=False,
            only_heat_map=False,
        )


def test_clustering_with_numeric_column_names():
    """
    Test that `ampiimts` works correctly when input columns are renamed to integers (1, 2, 3, ...)
    with clustering enabled. This simulates anonymized sensor data.

    Assertions
    ----------
    - The pipeline does not fail when columns are purely numeric.
    - Interpolated and normalized outputs are lists of DataFrames (due to clustering).
    - Result is also a list matching the number of clusters.
    """
    folder = "tests/data/air_bejin"
    input_data = load_and_concat_csvs_from_folder(folder)
    # Rename columns (except timestamp) to 1, 2, 3, ...
    i = 1
    for df in input_data:
        nb_cols = len(df.columns)
        new_columns = [str(j) for j in range(i, i + nb_cols)]
        df.columns = new_columns
        i += nb_cols

    interpolated, normalized, result = ampiimts(
        input_data,
        top_k_cluster=1,
        cluster=True,
        motif=False,
        most_stable_only=False,
        smart_interpolation=True,
        max_len=750,
        visualize=False,
        display_info=False,
    )

    # Assertions
    assert isinstance(interpolated, list)
    assert isinstance(normalized, list)
    assert isinstance(result, list)
    assert all(isinstance(df, pd.DataFrame) for df in interpolated)
    assert all(isinstance(df, pd.DataFrame) for df in normalized)
    assert len(result) == len(interpolated) == len(normalized)


def test_ampii_invalid_data_type():
    with pytest.raises(TypeError, match="`data` must be a path to a folder"):
        ampiimts(
            data=42,  # ← Type non valide
            cluster=False,
            motif=False,
            most_stable_only=False,
            smart_interpolation=True,
            window_size=60,
            visualize=False,
        )

def test_ampii_empty_folder_raises(tmp_path):
    # Folder empty
    with pytest.raises(ValueError, match="No .csv files found"):
        ampiimts(str(tmp_path))

def test_ampii_folder_with_non_csv_ignored(tmp_path):

    (tmp_path / "note.txt").write_text("don't process this file")

    df = pd.DataFrame({"timestamp": [1, 2, 3], "value": [10, 20, 30]})
    df.to_csv(tmp_path / "valid.csv", index=False)
    with pytest.raises(ValueError, match="Unable to compute common frequency: time gaps are zero."):
        ampiimts(str(tmp_path))

def test_ampii_folder_with_directory_ignored(tmp_path):

    os.mkdir(tmp_path / "subdir")

    df = pd.DataFrame({"timestamp": [1, 2], "value": [1, 2]})
    df.to_csv(tmp_path / "ok.csv", index=False)
    with pytest.raises(ValueError, match="Not enough points to estimate frequency."):
        ampiimts(str(tmp_path))


@pytest.mark.parametrize("freq_str", [
    "1ns", "100ns", "5us", "5ms", "50ms", "500ms",
    "5s", "1min", "5min", "30min", "1h", "12h", "1d", "2d"
])
def test_define_m_frequency_ranges(freq_str):
    def generate_df(freq: str, n: int = 50) -> pd.DataFrame:
        index = pd.date_range("2020-01-01", periods=n, freq=freq)
        data = np.random.randn(n, 2)
        return pd.DataFrame(data, index=index, columns=["sensor_1", "sensor_2"])
    df = generate_df(freq_str)
    with pytest.raises(ValueError, match="Dataframe too small"):
        define_m(df, k=3)

def test_force_faiss_with_realistic_motif():
    """
    Teste discover_patterns_mstump_mixed() avec un motif réaliste :
    montée → plateau → descente, répété 3 fois.
    """
    window_size = 15

    # Motif : montée linéaire (5), plateau (5), descente linéaire (5)
    ramp_up = np.linspace(0, 1, 5)
    plateau = np.ones(5)
    ramp_down = np.linspace(1, 0, 5)
    motif = np.concatenate([ramp_up, plateau, ramp_down])  # total 15

    # Signal = motifs espacés par du bruit faible
    signal = np.concatenate([
        np.random.normal(0, 0.01, 10),
        motif,
        np.random.normal(0, 0.01, 20),
        motif,
        np.random.normal(0, 0.01, 20),
        motif,
        np.random.normal(0, 0.01, 10),
    ])

    # DataFrame multivarié : deux capteurs identiques (comme des jumeaux)
    df = pd.DataFrame({
        "sensor_1": signal,
        "sensor_2": signal
    }, index=pd.date_range("2020-01-01", periods=len(signal), freq="1s"))

    df.attrs["m"] = window_size

    result = discover_patterns_mstump_mixed(
        df,
        window_size=window_size,
        motif=True,
        cluster=False,
        smart_interpolation=False,
        most_stable_only=False,
        max_motifs=3,
        max_matches=10,
        printunidimensional=False,
        discord_top_pct=0.01  # pour éviter de tout filtrer
    )

    print("[TEST] motifs détectés :", result["patterns"])
    assert isinstance(result["patterns"], list)
    assert len(result["patterns"]) >= 1

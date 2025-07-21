import pytest
import pandas as pd
from ampiimts import ampiimts
from pathlib import Path

# All possible boolean combinations for parameter testing
boolean_combinations = [
    (c, m, s, i)
    for c in [False, True]               # cluster
    for m in [False, True]               # motif
    for s in [False, True]               # most_stable_only
    for i in [False, True]               # smart_interpolation
]

@pytest.mark.parametrize("csv_path", [
    "tests/data/art_daily_jumpsup.csv",
    "tests/data/cpu_utilization_asg_misconfiguration.csv",
    "tests/data/nyc_taxi.csv"
])
@pytest.mark.parametrize("cluster, motif, most_stable_only, smart_interpolation", boolean_combinations)
def test_univariate_from_csv(csv_path, cluster, motif, most_stable_only, smart_interpolation):
    """
    Test the ampiimts pipeline on univariate CSV inputs with all boolean parameter combinations.

    Parameters
    ----------
    csv_path : str
        Path to a univariate time series CSV file.
    cluster, motif, most_stable_only, smart_interpolation : bool
        Parameters passed to the ampiimts pipeline.

    Assertions
    ----------
    - If result is returned, it must be a dictionary or None.
    - If interpolation/normalization succeed, their outputs must be pandas DataFrames.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    interpolated, normalized, result = ampiimts(
        df,
        cluster=cluster,
        motif=motif,
        most_stable_only=most_stable_only,
        smart_interpolation=smart_interpolation,
        window_size=60,
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
        group_size=10,
        visualize=False
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
        motif=False,
        window_size=window_size,
        most_stable_only=False,
        smart_interpolation=True,
        max_len=1000,
        visualize=True,
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

import pytest
import pandas as pd
from ampiimts import ampiimts
from pathlib import Path

# Toutes les combinaisons de paramètres booléens
boolean_combinations = [
    (c, m, s, i)
    for c in [False, True]
    for m in [False, True]
    for s in [False, True]
    for i in [False, True]
]

# Test unidimensionnel
@pytest.mark.parametrize("csv_path", [
    "tests/data/art_daily_jumpsup.csv",
    "tests/data/cpu_utilization_asg_misconfiguration.csv",
    "tests/data/nyc_taxi.csv"
])
@pytest.mark.parametrize("cluster, motif, most_stable_only, smart_interpolation", boolean_combinations)
def test_univariate_from_csv(csv_path, cluster, motif, most_stable_only, smart_interpolation):
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

# Utilitaire pour charger tous les CSV multivariés
def load_and_concat_csvs_from_folder(folder_path):
    dfs = []
    for file in sorted(Path(folder_path).glob("*.csv")):
        df = pd.read_csv(file, parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        dfs.append(df)
    return dfs

@pytest.mark.parametrize("cluster, motif, most_stable_only, smart_interpolation", boolean_combinations)
def test_multivariate_from_air_bejin(cluster, motif, most_stable_only, smart_interpolation):
    folder = "tests/data/air_bejin"

    interpolated, normalized, result = ampiimts(
        folder,
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

def test_multivariate_with_dataframe_input():
    folder = "tests/data/air_bejin"
    input_data = load_and_concat_csvs_from_folder(folder)

    interpolated, normalized, result = ampiimts(
        input_data,
        cluster=True,
        motif=False,
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

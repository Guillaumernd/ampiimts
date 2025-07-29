# AMPIIMTS

[![codecov](https://codecov.io/gh/Guillaumernd/ampiimts/graph/badge.svg?token=6JEX3JUSAB)](https://codecov.io/gh/Guillaumernd/ampiimts)
![Tests](https://github.com/Guillaumernd/ampiimts/actions/workflows/python-tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Licence](https://img.shields.io/github/license/Guillaumernd/ampiimts)

Adaptive Matrix Profile Indexing for irregular multivariate time series.

## Features

- Data Preprocessing
- Automatic Window Selection
- Clustering of Dimensions
- Matrix Profile Computation
- Motif and Discord Discovery
- Iterative Refinement (Smart Interpolation)
- Visualization Utilities


## Documentation

- [Documentation](documentation/README.md)
- [Examples](documentation/ampiimts_example.ipynb) with air bejin [dataset](tests/data/air_bejin/)
- [Diagram](documentation/Features_ampiimts_package.drawio.png)

## Installation

```bash
pip install ampiimts
```

To install the latest development version:

```bash
pip install git+https://github.com/your-username/ampiimts.git
```

For local development clone the repository and install with the testing extras:

```bash
git clone https://github.com/your-username/ampiimts.git
cd ampiimts
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[test]
```

## Quick start

```python
import pandas as pd
from ampiimts import ampiimts

# Either a path to a folder of CSV files or a DataFrame
results = ampiimts("/path/to/csv/folder", visualize=False)
```

`ampiimts` returns the interpolated data, the normalized data and a
dictionary with motif and discord information.

Depending on whether `cluster==True` or `cluster==False`, the parameters in result can be lists or non-lists.

`cluster==False`
- `result[0]` = `df_interpolate`

- `result[1]` = `df_normalize`

- `result[2]` = `dic_matrix_profile`
  - `result[2]['matrix_profile']` = `matrix_profile`
  - `result[2]['window_size']` = `window_size`
  - `result[2]['discord_indices']` = `discord_indices`
  - `result[2]['patterns']` = `list_dic_pattern` : 
    - `result[2]['patterns][pattern_label]` = `pattern_label`
    - `result[2]['patterns][medoid_idx_start]` = `medoid_idx_start`
    - `result[2]['patterns][motif_indice_start][0...n-1]` = `motif_indice_start`

`cluster==True`

- `result[0][0...n-1]` = `df_interpolate_0`

- `result[1][0...n-1]` = `df_normalize_0`

- `result[2][0...n-1]` = `dic_matrix_profile_0`


## Running tests

```
pytest -vv
```

## License

This project is released under the MIT license. See [LICENCE](LICENSE) for
details.

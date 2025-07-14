# AMPIIMTS

Adaptive Matrix Profile Indexing for irregular multivariate time series.

## Features

- Preprocessing helpers to interpolate irregular data and normalize values
  (ASWN normalization, trend removal, etc.).
- Automatic window size selection based on clustering or user provided sizes.
- Motif and discord detection through STUMPY matrix profile computations.
- Optional dimensionality reduction and FAISS based nearest neighbour search.
- Plotting utilities to visualise patterns and discords.

## Documentation

- [Documentation](documentation/README.md)
- [Examples](documentation/ampiimts_example.ipynb)

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

## Running tests

```
pytest -vv
```

## Contributing

Issues and pull requests are welcome. Please run the test suite before
submitting changes.

## License

This project is released under the MIT license. See [LICENCE](LICENCE) for
details.

# firstsearch
RT and M/z alignment from FirstSearches

## Installation
To install the firstsearch package and its dependencies, clone the repository and run:

`python3 -m pip install [path_to_firstsearch_dir]`

## Usage
Run an alignment on a search results CSV file, with default parameters and plotting:
```
from firstsearch import Alignment
alignment = Alignment()
alignment.fit_from_csv('path/to/csv.csv', plot=True)
```

After fitting the model with `fit_from_csv`, transform an array-like sequence of retention times `a` with:

`alignment.predict(a)`

Additional post-fit metrics are stored in:

`alignment.stats`

## Parameters
Parameters to initialize Alignment class:
- `model` (str): may be `"centered_isotonic"` (default) for `CenteredIsotonicRegression` or `"isotonic"` for `IsotonicRgression`.
- `model_kwargs` (dict): optional keyword arguments with which to initialize the chosen model
- `func` (str): (optional) if provided, a function will be used for curve fitting instead of the sklearn regression model. Options are `"sigmoid"`.

Parameters to Alignment.fit_from_csv:
- `csv` (str or file buffer): file path or buffer object, passed to `pandas.read_csv`
- `lib_rt` (str): name of library RT column header (default `"library_rt"`)
- `obs_rt` (str): name of observed RT column header (default `"rt"`)
- `score` (str): name of score column header (default `"hyperscore"`)
- `bin_ax` (str): axis on which to bin datapoints. Options are `"y"` (default) for lib RT, or `"x"` for obs RT.
- `bin_width` (float): bin width in minutes along selected `bin_ax` (default `0.75`)
- `std_tol_factor` (float): factor by which to multiply stdev to determine RT tolerance (default `0.5`)
- `csv_kwargs` (dict): optional keyword arguments to pass to `pandas.read_csv`
- `plot` (bool): if `True`, display plots (default `False`)

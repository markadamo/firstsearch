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


# Tables

Heavylight is design to be modular, you can pick and choose which functionality you use.

A core feature of actuarial models is the use of a variety of assumption tables, for example yield curves, mortality tables, lapse rates and expenses tables.

Heavylight includes the `Table` class which provides both individual and vectorised lookups, and an opinionated format for loading tables from pandas dataframes, csv files and excel files.

```python
from heavylight import Table
import pandas as pd
import numpy as np
# typically you would import the table from a csv or other source

df = pd.DataFrame(
  {'age|int': np.arange(18, 65)}
)
df['values'] = df['age|int'] * 0.01

tab = Table(df)

# we can query a single value
print(tab[20])   # prints(0.20)

# we can query multiple values at once using numpy arrays.
rng = np.random.default_rng(seed=42)   # set up random number generator
ages = rng.integers(low=18, high=65, size=100_000, endpoint=True)   # sample 100k ages
vals = tab[ages]    # query the table and store values
```

## Defining Column Types

`Table` determines the type of key column based on the column name suffix, the following are supported:

- `|int`: integers (...0, 1, 2, 3...), can start and end anywhere, but must be consecutive
- `|int_bound`: as `|int` but any values are constrained to the lowest and highest values.
- `|str`: keys are interpreted as strings, e.g. 'M' and 'F'
- `|band`: key is numeric and treated as the upper bound on a lookup.


## Mult-factor tables

Tables aren't limited to a single key, and keys can take any of the above types.

For examples, if a table `tab2` had headers `age|int` and `sex|str`, to look up the value for age `20` and sex `F`:

```python
tab2[20, 'F']
```

This also works with by passing in numpy arrays of keys.
```python
ages = np.array([20, 25, 30, 22])
sexs = np.array(['F', 'M', 'M', F])
tab2[ages, sexs]
```
# Tables

Heavylight is design to be modular, you can pick and choose which functionality you use.

A core feature of actuarial models is the use of a variety of assumption tables, for example yield curves, mortality tables, lapse rates and expesnse tables.

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


```

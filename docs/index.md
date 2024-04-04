# Getting started

Heavylight is a lightweight Python library that allows you to run heavy modeling workloads using a familiar recursive syntax.

## Installation

```bash
pip install heavylight
```

Requires Python 3.8+

## Quick example

Models are defined by subclassing from `heavylight.Model`

```python
from heavylight import Model

class Annuity(Model):
    def t(self, t):
        return t

    def expected_claim(self, t):
        return self.number_alive(t) * self.data["annuity_per_period"]

    def number_alive(self, t):
        if t == 0:
            return self.data["initial_policies"]
        else:
            return self.number_alive(t - 1) - self.deaths(t - 1)
    
    def deaths(self, t):
        return self.number_alive(t) * self.mortality_rate(t)

    def mortality_rate(self, t):
        return self.basis["q_x"](t)
```

We can define data as a dictionary (heavylight can use any storage mechanism)

```python
policy_data = {
    "initial_policies": 10,
    "annuity_per_period": 55,
}
```

We likely need a basis too, heavylight includes a `Table` class for storing tables, but for this example we will define a mortality function directly, and store it in a basis:

```python
def q_x(t):
    return 0.02*2.64**(0.04 * (t + 30)) + 0.002

basis = {
    'q_x':q_x,
}
```

Pulling it all together, we run a projection and store it in a variable `model`

```python
model = Annuity(data = policy_data,
                basis = basis,
                proj_len = 40,
                )
```

We can query individual results:

```python
model.expected_claim(5)
```

Output: `379.7484060121289`

We can view results as a DataFrame:

```python
model_cashflows = model.df
model_cashflows.head()
```

Output:

|    |   t |   deaths |   expected_claim |   mortality_rate |   number_alive |
|---:|----:|---------:|-----------------:|-----------------:|---------------:|
|  0 |   0 | 0.661143 |          550     |        0.0661143 |       10       |
|  1 |   1 | 0.641139 |          513.637 |        0.0686529 |        9.33886 |
|  2 |   2 | 0.620078 |          478.374 |        0.071292  |        8.69772 |
|  3 |   3 | 0.598033 |          444.27  |        0.0740356 |        8.07764 |
|  4 |   4 | 0.575091 |          411.378 |        0.0768878 |        7.47961 |

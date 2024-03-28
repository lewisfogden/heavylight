```python
import numpy as np
import heavylight

class SimpleModel(heavylight.Model):

    def __init__(self, initial_pols_if: np.ndarray, mortality_rate: float):
        # storage_function determines what gets stored in the results DataFrame
        super().__init__(storage_function=lambda results: np.round(np.sum(results), 3)) # customize how you aggregate results
        self.initial_pols_if = initial_pols_if
        self.mortality_rate = mortality_rate

    def t(self, t):
        return t

    def num_pols_if(self, t):
        if t == 0:
            return self.initial_pols_if
        return self.num_pols_if(t - 1) - self.pols_death(t - 1) # causes exponential time complexity if uncached
        
    def pols_death(self, t):
        return self.num_pols_if(t) * self.mortality_rate

    def cashflow(self, t):
        return self.num_pols_if(t) * 100
    
    def v(self, t):
        if t == 0:
            return 1
        return self.v(t - 1) / (1 + self.forward_rate(t))
    
    def forward_rate(self, t):
        return 0.04
    
    def pv_cashflow(self, t):
        return self.cashflow(t) * self.v(t)
    
# start with 10 policies and constant mortality rate of .01
simple_model = SimpleModel(initial_pols_if=np.ones((10,)), mortality_rate=.01)
# run the model for 5 timesteps
simple_model.RunModel(proj_len=5)
# create a dataframe to store results
results = simple_model.ToDataFrame()
```

Results are a Pandas dataframe:

|   t |   num_pols_if |   cashflow |   forward_rate |   pols_death |     v |   pv_cashflow |
|----:|--------------:|-----------:|---------------:|-------------:|------:|--------------:|
|   0 |        10     |   1000     |           0.04 |        0.1   | 1     |      1000     |
|   1 |         9.9   |    990     |           0.04 |        0.099 | 0.962 |       951.923 |
|   2 |         9.801 |    980.1   |           0.04 |        0.098 | 0.925 |       906.158 |
|   3 |         9.703 |    970.299 |           0.04 |        0.097 | 0.889 |       862.592 |
|   4 |         9.606 |    960.596 |           0.04 |        0.096 | 0.855 |       821.121 |
|   5 |         9.51  |    950.99  |           0.04 |        0.095 | 0.822 |       781.645 |


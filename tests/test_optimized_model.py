import numpy as np
from heavylight import LightModel
from typing import Union, Callable
import pytest

class SimpleModel(LightModel):

    def __init__(self, initial_pols_if: np.ndarray, storage_function: Union[Callable, None] = None, mortality_rate = .01):
        super().__init__(storage_function=storage_function)
        self.initial_pols_if = initial_pols_if
        self.mortality_rate = mortality_rate

    def t(self, t):
        return t

    def num_pols_if(self, t):
        if t == 0:
            return self.initial_pols_if
        else:
            return self.num_pols_if(t - 1) - self.pols_death(t - 1) # causes exponential time complexity if uncached
        
    def pols_death(self, t):
        return self.num_pols_if(t) * self.mortality_rate

    def cashflow(self, t):
        return self.num_pols_if(t) * 100
    
    def v(self, t):
        """discount factor for time t --> time 0"""
        if t == 0:
            return 1
        else:
            return self.v(t - 1) / (1 + self.forward_rate(t))
    
    def forward_rate(self, t):
        return 0.04
    
    def pv_cashflow(self, t):
        """present value of the cashflow occuring at time t"""
        return self.cashflow(t) * self.v(t)
    
    def fib(self, t, mult_factor):
        if t <= 1:
            return t
        return mult_factor * self.fib(t - 1, mult_factor) + self.fib(t - 2, mult_factor)

def test_forward_rate():
    sm = SimpleModel(np.linspace(.1, 1, 10), np.sum)
    sm.forward_rate(0)
    assert sm.forward_rate.cache[((0,), frozenset())] == .04
    assert len(sm.StoredResults()) == 1
    assert len(sm.StoredResults()['forward_rate']) == 1
    assert sm.StoredResults()['forward_rate'][0] == .04

    sm.forward_rate(5)
    assert len(sm.forward_rate.cache) == 2


def test_timestep_functions():
    sm = SimpleModel(np.linspace(.1, 1, 10), np.sum)
    timestep_functions = sm.TimestepFunctions()
    expected_functions = ['t', 'num_pols_if', 'pols_death', 'cashflow', 'v', 'pv_cashflow', 'forward_rate']
    assert set(timestep_functions) == set(expected_functions) # no duplicates
    sm.RunModel(5)
    # The timestep functions are the ones displayed in StoredResults()
    assert set(sm.StoredResults().keys()) == set(expected_functions)

@pytest.mark.timeout(4)
def test_caching():
    sm = SimpleModel(np.linspace(.1, 1, 10), np.sum)
    assert len(sm.num_pols_if.cache) == 0
    sm.RunModel(200)
    assert len(sm.num_pols_if.cache) == 201
    assert type(sm.StoredResults()['num_pols_if'][200]) == np.float64
    # fib did not get called because it is not single parameter time function
    assert len(sm.fib.cache) == 0
    sm.fib(200, 1.1)
    # but it is still cached
    assert len(sm.fib.cache) == 201

def test_dataframe():
    sm = SimpleModel(np.linspace(.1, 1, 10), np.sum)
    sm.RunModel(5)
    df = sm.ToDataFrame()
    assert df.shape == (6, 7)
    assert df['t'].tolist() == [0, 1, 2, 3, 4, 5]

def test_reset_cache():
    sm = SimpleModel(np.linspace(.1, 1, 10), storage_function=np.sum, mortality_rate=.01)
    sm.RunModel(5)
    assert round(sm.StoredResults()['pols_death'][0], 10) == .055
    sm.mortality_rate = .02
    sm.RunModel(5)
    assert round(sm.StoredResults()['pols_death'][0], 10) == .055
    sm.ResetCache()
    sm.RunModel(5)
    assert round(sm.StoredResults()['pols_death'][0], 10) == .11
    


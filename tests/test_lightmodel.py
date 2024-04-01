import numpy as np
from heavylight import LightModel
from heavylight.memory_optimized_model import store
from typing import Union, Callable
import pytest

class SimpleModel(LightModel):

    def __init__(self, initial_pols_if: np.ndarray, mortality_rate = .01):
        super().__init__()
        self.initial_pols_if = initial_pols_if
        self.mortality_rate = mortality_rate

    def t(self, t):
        return t
    
    @store(np.sum)
    def num_pols_if(self, t):
        if t == 0:
            return self.initial_pols_if
        else:
            return self.num_pols_if(t - 1) - self.pols_death(t - 1) # causes exponential time complexity if uncached
    
    @store(np.sum)
    def pols_death(self, t):
        return self.num_pols_if(t) * self.mortality_rate

    @store(np.sum)
    def cashflow(self, t):
        return self.num_pols_if(t) * 100
    
    def v(self, t):
        """discount factor for time t --> time 0"""
        if t == 0:
            return 1
        else:
            return self.v(t - 1) / (1 + self.forward_rate(t))
    
    @store(lambda x: x * 2)
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
    sm = SimpleModel(np.linspace(.1, 1, 10))
    sm.forward_rate(0)
    assert sm.forward_rate._cache[((0,), frozenset())] == .04
    assert len(sm.StoredResults()) == 1
    assert len(sm.StoredResults()['forward_rate']) == 1
    assert sm.StoredResults()['forward_rate'][0] == .08

    sm.forward_rate(5)
    assert len(sm.forward_rate._cache) == 2


def test_timestep_functions():
    sm = SimpleModel(np.linspace(.1, 1, 10))
    timestep_functions = sm.TimestepFunctions()
    expected_functions = ['t', 'num_pols_if', 'pols_death', 'cashflow', 'v', 'pv_cashflow', 'forward_rate']
    assert set(timestep_functions) == set(expected_functions) # no duplicates
    sm.RunModel(5)
    # The timestep functions are the ones displayed in StoredResults()
    assert set(sm.StoredResults().keys()) == set(["cashflow", "forward_rate", "num_pols_if", "pols_death"])

@pytest.mark.timeout(4)
def test_caching():
    sm = SimpleModel(np.linspace(.1, 1, 10))
    assert len(sm.num_pols_if._cache) == 0
    sm.RunModel(200)
    assert len(sm.num_pols_if._cache) == 201
    assert type(sm.StoredResults()['num_pols_if'][200]) == np.float64
    # fib did not get called because it is not single parameter time function
    assert len(sm.fib._cache) == 0
    sm.fib(200, 1.1)
    # but it is still cached
    assert len(sm.fib._cache) == 201

def test_reset_cache():
    sm = SimpleModel(np.linspace(.1, 1, 10), mortality_rate=.01)
    sm.RunModel(5)
    assert round(sm.StoredResults()['pols_death'][0], 10) == .055
    sm.mortality_rate = .02
    sm.RunModel(5)
    assert round(sm.StoredResults()['pols_death'][0], 10) == .055
    sm.ResetCache()
    sm.RunModel(5)
    assert round(sm.StoredResults()['pols_death'][0], 10) == .11
    
class TestPrettyCacheModel(LightModel):
    def __init__(self):
        super().__init__()
    def t(self, t):
        return self.wowee(t, 1) + self.zowee(t, x=1)
    def wowee(self, t, x):
        return 1
    def zowee(self, t, x):
        return 2
    
def test_pretty_cache():
    pretty_model = TestPrettyCacheModel()
    pretty_model.ResetCache()
    pretty_model.RunModel(0)
    assert pretty_model.cache_graph.caches['wowee'][(0,1)] == 1
    assert pretty_model.wowee.cache[(0,1)] == 1
    assert pretty_model.cache_graph.caches['zowee'][((0,),frozenset({'x': 1}.items()))] == 2
    assert pretty_model.zowee.cache[((0,),frozenset({'x': 1}.items()))] == 2
    assert pretty_model.cache_graph.caches['t'][0] == 3
    assert pretty_model.t.cache[0] == 3
    # can inject into the cache
    pretty_model.zowee[1] = 'hello cache'
    assert pretty_model.zowee.cache[1] == 'hello cache'

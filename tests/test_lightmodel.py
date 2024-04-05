import numpy as np
from heavylight import LightModel
from heavylight.memory_optimized_model import agg
from typing import Union, Callable
import pytest

class SimpleModel(LightModel):

    def __init__(self, initial_pols_if: np.ndarray, mortality_rate = .01):
        super().__init__()
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
    
    @agg(lambda x: x * 2)
    def forward_rate(self, t):
        return 0.04
    
    def pv_cashflow(self, t):
        """present value of the cashflow occuring at time t"""
        return self.cashflow(t) * self.v(t)
    
    def fib(self, t, mult_factor):
        if t <= 1:
            return t
        return mult_factor * self.fib(t - 1, mult_factor) + self.fib(t - 2, mult_factor)

def test_method_call_and_cache_retrievals():
    sm = SimpleModel(np.linspace(.1, 1, 10))
    sm.forward_rate(0)
    assert sm.forward_rate.cache[0] == .04
    assert sm._cache_graph._caches['forward_rate'][((0,), frozenset())] == .04
    assert sm._cache_graph.cache['forward_rate'][0] == .04
    assert len(sm._cache_graph.cache) == 1
    assert len(sm.forward_rate.cache) == 1
    sm.forward_rate(5)
    assert len(sm.forward_rate.cache) == 2
    assert len(sm._cache_graph.cache) == 1

def test_timestep_functions():
    sm = SimpleModel(np.linspace(.1, 1, 10))
    timestep_functions = sm._timestep_functions
    expected_functions = ['t', 'num_pols_if', 'pols_death', 'cashflow', 'v', 'pv_cashflow', 'forward_rate']
    assert set(timestep_functions) == set(expected_functions) # no duplicates

@pytest.mark.timeout(4)
def test_caching_speedups():
    sm = SimpleModel(np.linspace(.1, 1, 10))
    assert len(sm.num_pols_if.cache) == 0
    sm.RunModel(200)
    assert len(sm.num_pols_if.cache) == 201
    # fib did not get called because it is not single parameter time function
    assert len(sm.fib.cache) == 0
    sm.fib(200, 1.1)
    # but it is still cached
    assert len(sm.fib.cache) == 201

def test_reset_cache():
    sm = SimpleModel(np.linspace(.1, 1, 10))
    sm.RunModel(5)
    assert round(np.sum(sm.pols_death.cache[0]), 10) == .055
    sm.mortality_rate = .02
    sm.RunModel(5)
    assert round(np.sum(sm.pols_death.cache[0]), 10) == .055
    sm.Clear()
    sm.RunModel(5)
    assert round(np.sum(sm.pols_death.cache[0]), 10) == .11

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
    pretty_model.Clear()
    pretty_model.RunModel(0)
    assert pretty_model._cache_graph.cache['wowee'][(0,1)] == 1
    assert pretty_model.wowee.cache[(0,1)] == 1
    assert pretty_model._cache_graph.cache['zowee'][((0,),frozenset({'x': 1}.items()))] == 2
    assert pretty_model.zowee.cache[((0,),frozenset({'x': 1}.items()))] == 2
    assert pretty_model._cache_graph.cache['t'][0] == 3
    assert pretty_model.t.cache[0] == 3
    # can inject into the cache
    pretty_model.zowee[1] = 'hello cache'
    assert pretty_model.zowee.cache[1] == 'hello cache'

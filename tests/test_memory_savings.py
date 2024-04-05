import numpy as np
from heavylight import LightModel
import numpy as np
import pytest

class SimpleModel(LightModel):

    def __init__(self, initial_pols_if: np.ndarray, mortality_rate = .01):
        super().__init__()
        self.initial_pols_if = initial_pols_if
        self.mortality_rate = mortality_rate

    def t(self, t):
        return np.int64(t) # to calculate nbytes for memory consumption

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
        if t == 0:
            return np.int64(1)
        else:
            return self.v(t - 1) / (1 + self.forward_rate(t))
    
    def forward_rate(self, t):
        return np.float64(0.04)
    
    def pv_cashflow(self, t):
        return self.cashflow(t) * self.v(t)

def calculate_cache_graph_size(model: LightModel):
    cg = model._cache_graph
    return sum(val.nbytes for cache in cg._caches.values() for val in cache.values())

def run_model_calculate_max_cache(model: SimpleModel, max_time: int):
    max_cache_size = 0
    for t in range(max_time+1):
        model.pv_cashflow(t)
        cache_size = calculate_cache_graph_size(model)
        max_cache_size = max(max_cache_size, cache_size)
    return max_cache_size

def get_memory_savings_ratio(model: SimpleModel, max_time: int):
    model.Clear()
    max_cache_size = run_model_calculate_max_cache(model, max_time)
    model.ClearOptimize()
    optimized_max_cache_graph_size = run_model_calculate_max_cache(model, max_time)
    return optimized_max_cache_graph_size / max_cache_size

@pytest.mark.timeout(4)
def test_memory_savings():
    model = SimpleModel(np.ones((1000,)))
    assert get_memory_savings_ratio(model, 100) < .01
    assert sum(model._cache_graph.cache_misses.values()) > 0 # verify that misses are being tracked
    assert all(misses == 1 for misses in model._cache_graph.cache_misses.values())

def test_cache_misses():
    sm = SimpleModel(np.linspace(.1, 1, 10))
    sm.RunModel(5)
    assert len(sm._cache_graph.cache_misses.values()) > 0
    assert all(x == 1 for x in sm._cache_graph.cache_misses.values())
    sm.ClearOptimize()
    assert len(sm._cache_graph.cache_misses.values()) == 0
    sm.RunModel(5)
    assert len(sm._cache_graph.cache_misses.values()) > 0
    assert all(x == 1 for x in sm._cache_graph.cache_misses.values())

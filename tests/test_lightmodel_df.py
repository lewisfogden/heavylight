"""Testing the dataframe and cache presentation for LightModel class."""
from heavylight import LightModel, agg
from heavylight.memory_optimized_model import default_agg_function
import numpy as np
from typing import Callable, Union
import pandas as pd

class TestModel(LightModel):
    def __init__(self, agg_function: Union[Callable, None]):
        super().__init__(agg_function)
        self.size = 2

    def pols_if(self, t):
        if t == 0:
            return np.ones((self.size,))
        return self.pols_if(t - 1) - self.pols_death(t - 1)
    
    def pols_death(self, t):
        return .01 * self.pols_if(t)
    
    @agg(None)
    def test_agg_none(self, t):
        return np.ones((self.size,))
    
    @agg(len)
    def t(self, t):
        self.multi_param(t, t)
        return np.ones((self.size,))
    
    def multi_param(self, t, t2):
        return np.ones((self.size,))
    
expected_cache = {'pols_death': {0: np.array([0.01, 0.01]),
    1: np.array([0.0099, 0.0099])},
    'pols_if': {0: np.array([1., 1.]), 1: np.array([0.99, 0.99])},
    'test_agg_none': {0: np.array([1., 1.]), 1: np.array([1., 1.])},
    't': {0: np.array([1., 1.]), 1: np.array([1., 1.])},
    'multi_param': {(0, 0): np.array([1., 1.]), (1, 1): np.array([1., 1.])}}

expected_cache_no_multi = { k: v for k, v in expected_cache.items() if k != 'multi_param' }

expected_cache_agg = {'pols_if': {0: 2.0, 1: 1.98},
    'pols_death': {0: 0.02, 1: 0.0198},
    't': {0: 2, 1: 2},
    'multi_param': {(0, 0): 2, (1, 1): 2}}

expected_cache_agg_no_multi = { k: v for k, v in expected_cache_agg.items() if k != 'multi_param' }

expected_cache_agg_none_aggfunc = {'t': {0: 2, 1: 2}}

def test_model_df_before_run():
    tm = TestModel(default_agg_function)
    assert set(tm.df_agg.columns) == set(['pols_if', 'pols_death', 't'])
    assert set(tm.df.columns) == set(['pols_if', 'pols_death', 'test_agg_none', 't'])
    assert len(tm.df) == 0
    assert len(tm.df_agg) == 0

def test_method_df():
    tm = TestModel(default_agg_function)
    assert len(tm.t.cache) == 0
    assert tm.t.df.columns == ['t'] and len(tm.t.df) == 0
    tm.RunModel(1)
    assert len(tm.t.cache) == 2
    compare_structures(tm.t.cache, expected_cache['t'])
    assert tm.t.df.equals(pd.DataFrame({"t": tm.t.cache}))
    assert tm.t.df_agg.equals(pd.DataFrame({"t": tm.t.cache_agg}))

def test_model_cache_df_after_run():
    tm = TestModel(default_agg_function)
    tm.RunModel(1)
    compare_structures(tm.cache_agg, expected_cache_agg)
    compare_structures(tm.cache, expected_cache)
    assert tm.df.sort_index(axis=1).equals(pd.DataFrame(expected_cache_no_multi).sort_index(axis=1))
    assert tm.df_agg.sort_index(axis=1).equals(pd.DataFrame(expected_cache_agg_no_multi).sort_index(axis=1))

def test_model_cache_df_after_run_agg_none():
    tm = TestModel(None)
    assert set(tm.df.columns) == set(['pols_if', 'pols_death', 'test_agg_none', 't'])
    assert set(tm.df_agg.columns) == set(['t'])
    tm.RunModel(1)
    compare_structures(tm.cache_agg, expected_cache_agg_none_aggfunc)
    compare_structures(tm.cache, expected_cache)
    assert tm.df.sort_index(axis=1).equals(pd.DataFrame(expected_cache_no_multi).sort_index(axis=1))
    assert tm.df_agg.sort_index(axis=1).equals(pd.DataFrame(expected_cache_agg_none_aggfunc).sort_index(axis=1))

def test_model_t_first_column():
    tm = TestModel(default_agg_function)
    assert tm.df.columns[0] == 't'
    assert tm.df_agg.columns[0] == 't'
    tm.RunModel(1)
    assert tm.df.columns[0] == 't'
    assert tm.df_agg.columns[0] == 't'

def compare_structures(obj1, obj2):
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        assert set(obj1.keys()) == set(obj2.keys()), "Dictionary keys do not match."
        for key in obj1:
            compare_structures(obj1[key], obj2[key])
    elif isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
        assert np.array_equal(obj1, obj2), "Numpy arrays are not equal."
    else:
        assert obj1 == obj2, f"Values do not match: {obj1} != {obj2}"
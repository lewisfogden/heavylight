import numpy as np
from heavylight import LightModel
from heavylight.memory_optimized_model import agg
from typing import Union, Callable, Any
import pytest

class AggModel(LightModel):

    def __init__(self, agg_function: Union[Callable, None] = None):
        super().__init__(agg_function)

    @agg(None)
    def not_agg(self, t):
        return np.array([t, t])
    
    # @agg(lambda x: 2*x)
    # def doubled(self, t):
    #     self.default_applied(t, t)
    #     return np.array([t, t])
    
    # def default_applied(self, w, x):
    #     return np.array([w, x])
    
def default_agg_function(x: Any):
    if isinstance(x, (int, float, np.number)):
        return x    
    if isinstance(x, np.ndarray) and issubclass(x.dtype.type, np.number):
        return np.sum(x)
    
# @pytest.fixture()
# def no_agg():
#     agg_model = AggModel()
#     agg_model.RunModel(1)
#     return agg_model

@pytest.fixture()
def sum_agg():
    agg_model = AggModel(np.sum)
    agg_model.RunModel(1)
    return agg_model

def test_agg_none_functions(sum_agg):
    # assert no_agg.not_agg.cache[1].tolist() == [1, 1]
    # assert 1 not in no_agg.not_agg.cache_agg
    assert sum_agg.not_agg.cache[1].tolist() == [1, 1]
    assert 1 not in sum_agg.not_agg.cache_agg
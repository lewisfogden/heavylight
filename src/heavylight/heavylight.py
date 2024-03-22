from inspect import getmembers, signature
from typing import Callable, List, Union
from types import MethodType
from heavylight.cache_graph import CacheGraph, _Cache
import pandas as pd

def check_if_single_parameter_t(func: Callable):
    sig = signature(func)
    return 't' in sig.parameters and len(sig.parameters) == 1

class Model:
    def __init__(self, storage_function: Union[Callable, None] = None):
        self.cache_graph = CacheGraph()
        self._single_param_timestep_funcs: List[_Cache]  = []
        self._storage_function = storage_function
        # happens after setting up attributes
        for method_name, method in getmembers(self):
            if not method_name[0].islower() or method_name.startswith("_") or not isinstance(method, MethodType):
                continue
            is_single_param_t = check_if_single_parameter_t(method)
            if is_single_param_t:
                cached_method = self.cache_graph(self._storage_function)(method)
                self._single_param_timestep_funcs.append(cached_method)
            else:
                cached_method = self.cache_graph()(method)
            setattr(self, method_name, cached_method)

    def ResetCache(self):
        self.cache_graph.reset()

    def RunModel(self, proj_len: int):
        for t in range(proj_len+1):
            for func in self._single_param_timestep_funcs:
                func(t)

    def OptimizeMemoryAndReset(self):
        self.cache_graph.optimize_and_reset()

    def TimestepFunctions(self):
        return [cache._func.__name__ for cache in self._single_param_timestep_funcs]
    
    def StoredResults(self):
        return self.cache_graph.stored_results
    
    def ToDataFrame(self):
        """return a pandas dataframe of all single parameter columns"""
        return pd.DataFrame(self.StoredResults())
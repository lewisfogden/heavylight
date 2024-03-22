from inspect import getmembers, signature
from typing import Callable, List, Union
from types import MethodType
from heavylight.cache_graph import CacheGraph, _Cache

def no_cache(func):
    """Decorator to remove caching from a function"""
    setattr(func, "no_cache", True)
    return func

def check_if_single_parameter_t(func: Callable):
    sig = signature(func)
    return 't' in sig.parameters and len(sig.parameters) == 1

class Model:
    def __init__(self, aggregation_function: Union[Callable, None] = None):
        self.cache_graph = CacheGraph()
        self._single_param_timestep_funcs: List[_Cache]  = []
        self._aggregation_function = aggregation_function
        # happens after setting up attributes
        for method_name, method in getmembers(self):
            if hasattr(method, "no_cache") or method_name.startswith("_") or not isinstance(method, MethodType):
                continue
            is_single_param_t = check_if_single_parameter_t(method)
            if is_single_param_t:
                cached_method = self.cache_graph(self._aggregation_function)(method)
                self._single_param_timestep_funcs.append(cached_method)
            else:
                cached_method = self.cache_graph()(method)
            setattr(self, method_name, cached_method)

    @no_cache
    def reset_cache(self):
        self.cache_graph.reset()

    @no_cache
    def run(self, proj_len: int):
        for t in range(proj_len):
            for func in self._single_param_timestep_funcs:
                func(t)

    @no_cache
    def optimize_memory(self):
        self.cache_graph.optimize_and_reset()

    @property
    @no_cache
    def timestep_functions(self):
        return [cache._func.__name__ for cache in self._single_param_timestep_funcs]
    
    @property
    @no_cache
    def aggregated_results(self):
        return self.cache_graph.stored_results
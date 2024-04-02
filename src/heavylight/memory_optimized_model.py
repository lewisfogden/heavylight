from inspect import getmembers, signature
from typing import Callable, List, Union, Any
from types import MethodType
from heavylight.memory_optimized_cache import CacheGraph, _Cache, FunctionCall
import pandas as pd
import numpy as np

class LightModel:
    """Base class to subclass from for recursive actuarial models.

    Inheriting from this class causes functions starting with a lowercase letter to be cached.
    A cached function will only be called once for each unique set of arguments.
    Caching is necessary for recursive actuarial calculations to be efficient.

    This means that the function will only be called once for each unique set of arguments, important for recursive actuarial models.

    Parameters
    ----------
    agg_function: Callable[[int], Any], optional
        This function is applied to the results of methods starting with a lowercase letter.

    Class level methods:
        RunModel(proj_len):
        if the model has not been auto-run at initialisation, run it for projection length.

    methods/variables to avoid:
    methods/variables starting with an underscore `_` are treated as internal.  You may break functionality if you create your own.
    """
    
    def __init__(self, agg_function: Union[Callable, None] = None):
        self.cache_graph = CacheGraph()
        self._single_param_timestep_funcs: List[_Cache]  = []
        # happens after setting up attributes
        for method_name, method in getmembers(self):
            if not method_name[0].islower() or method_name.startswith("_") or not isinstance(method, MethodType):
                continue
            method_agg_function = getattr(method, "agg_function", None)
            if not hasattr(method, "agg_function"):
                method_agg_function = agg_function
            cached_method = self.cache_graph(method_agg_function)(method)
            is_single_param_t = check_if_single_parameter_t(method)
            if is_single_param_t:
                self._single_param_timestep_funcs.append(cached_method)
            setattr(self, method_name, cached_method)

    def RunModel(self, proj_len: int):
        """
        Run the model for a projection length.
        All single parameter timestep functions will be run for each timestep.
        """
        for t in range(proj_len+1):
            for func in self._single_param_timestep_funcs:
                # We avoid recalling any functions that have already been cached, resolves issue #15 lewisfogden/heavylight
                if not FunctionCall(func._func.__name__, (t,), frozenset()) in self.cache_graph.all_calls:
                    func(t)

    def ResetCache(self):
        """Reset the cache, useful if you want to run the model again with different parameters."""
        self.cache_graph.reset()

    def OptimizeMemoryAndReset(self):
        """
        Calling this clears the cache, and causes the cache to be optimized.
        Optimizing the cache means that only needed results are stored, and intermediate results are cleared.
        """
        self.cache_graph.optimize_and_reset()

    def TimestepFunctions(self):
        """List the cached functions that have a single parameter `t`."""
        return [cache._func.__name__ for cache in self._single_param_timestep_funcs]
    
    def AggResults(self):
        """
        Return a dictionary of all stored results. 
        The dictionary is of the form:
        {function_name: {timestep: stored_result}}
        """
        return self.cache_graph.caches_agg
    
    def ToDataFrame(self):
        """Return a pandas dataframe of all single parameter timestep results."""
        df = pd.DataFrame(self.AggResults())
        # if t is in the dataframe, move it to first position
        if "t" in df.columns:
            df.insert(0, "t", df.pop("t"))
        return df


def check_if_single_parameter_t(func: Callable):
    sig = signature(func)
    return 't' in sig.parameters and len(sig.parameters) == 1

def agg(agg_function: Union[Callable, None]):
    """
    Register the storage function on a function. 
    Used for storing aggregated results before cache eviction to reduce memory consumption.
    """
    def decorator(func: Callable):
        func.agg_function = agg_function
        return func
    return decorator


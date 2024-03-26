from inspect import getmembers, signature
from typing import Callable, List, Union
from types import MethodType
from heavylight.memory_optimized_cache import CacheGraph, _Cache
import pandas as pd


class LightModel:
    """Base class to subclass from for recursive actuarial models.

    Inheriting from this class causes functions starting with a lowercase letter to be cached.
    A cached function will only be called once for each unique set of arguments.
    Caching is necessary for recursive actuarial calculations to be efficient.

    This means that the function will only be called once for each unique set of arguments, important for recursive actuarial models.

    Parameters
    ----------
    storage_function: Callable[[int], Any], optional
        This function is applied to the results of cached methods that have a single parameter `t`.
        (e.g. `def pols_if(self, t):`).  The default is None, no values will be stored.
        `storage_function=np.sum` will cause the sum of the results to be stored.
        `storage_function=lambda x: x[0] if isinstance(x, np.ndarray) else x` will store the first element.
        `storage_function=lambda x: x` will store the raw results.

    Class level methods:
        RunModel(proj_len):
        if the model has not been auto-run at initialisation, run it for projection length.

    methods/variables to avoid:
    methods/variables starting with an underscore `_` are treated as internal.  You may break functionality if you create your own.
    """
    
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

    def RunModel(self, proj_len: int):
        """
        Run the model for a projection length.
        All single parameter timestep functions will be run for each timestep.
        """
        for t in range(proj_len+1):
            for func in self._single_param_timestep_funcs:
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
    
    def StoredResults(self):
        """
        Return a dictionary of all stored results. 
        The dictionary is of the form:
        {function_name: {timestep: stored_result}}
        """
        return self.cache_graph.stored_results
    
    def ToDataFrame(self):
        """Return a pandas dataframe of all single parameter timestep results."""
        df = pd.DataFrame(self.StoredResults())
        # if t is in the dataframe, move it to first position
        if "t" in df.columns:
            df.insert(0, "t", df.pop("t"))
        return df


def check_if_single_parameter_t(func: Callable):
    sig = signature(func)
    return 't' in sig.parameters and len(sig.parameters) == 1
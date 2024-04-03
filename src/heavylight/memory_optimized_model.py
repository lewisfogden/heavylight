from inspect import getmembers, signature
from typing import Callable, Dict, List, Union, Any
from types import MethodType
from heavylight.memory_optimized_cache import CacheGraph, CacheMethod, FunctionCall
import pandas as pd
import numpy as np


def default_agg_function(x: Any):
    """Default aggregation function for storing results."""
    if isinstance(x, np.ndarray) and issubclass(x.dtype.type, np.number):
        return np.sum(x)
    return x


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

    def __init__(self, agg_function: Union[Callable, None] = default_agg_function):
        self.cache_graph = CacheGraph()
        self._single_param_timestep_funcs: List[CacheMethod] = []
        self._funcs: Dict[MethodType, CacheMethod] = {}
        # happens after setting up attributes
        for method_name, method in getmembers(self):
            if (
                not method_name[0].islower()
                or method_name.startswith("_")
                or not isinstance(method, MethodType)
            ):
                continue
            method_agg_function = getattr(method, "agg_function", None)
            if not hasattr(method, "agg_function"): # only provide class-level agg_function if method-level is not provided
                method_agg_function = agg_function
            cached_method = self.cache_graph(method_agg_function)(method)
            self._funcs[method] = cached_method
            is_single_param_t = check_single_parameter_name(method, "t")
            if is_single_param_t:
                self._single_param_timestep_funcs.append(cached_method)
            setattr(self, method_name, cached_method)

    def RunModel(self, proj_len: int):
        """
        Run the model for a projection length.
        All single parameter timestep functions will be run for each timestep.
        """
        for t in range(proj_len + 1):
            for func in self._single_param_timestep_funcs:
                # We avoid recalling any functions that have already been cached, resolves issue #15 lewisfogden/heavylight
                if (
                    not FunctionCall(func.func.__name__, (t,), frozenset())
                    in self.cache_graph.all_calls
                ):
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

    @property
    def timestep_functions(self):
        """List the cached functions that have a single parameter `t`."""
        return [cache.func.__name__ for cache in self._single_param_timestep_funcs]

    @property
    def cache(self):
        return self.cache_graph.cache

    @property
    def cache_agg(self):
        return self.cache_graph.cache_agg

    @property
    def df(self):
        return self.ToDataFrame(is_agg=False)

    @property
    def df_agg(self):
        return self.ToDataFrame(is_agg=True)

    def ToDataFrame(self, param_name="t", is_agg=False):
        """Return a pandas dataframe of all single parameter timestep results."""
        get_method_cache = _get_method_cache_factory(is_agg)
        filtered_cache = {}
        for method, cache_method in self._funcs.items():
            if not check_single_parameter_name(method, param_name):
                continue
            if is_agg and cache_method.agg_func is None:
                continue
            filtered_cache[method.__name__] = get_method_cache(cache_method)
        df = pd.DataFrame(filtered_cache)
        # if t is in the dataframe, move it to first position
        if "t" in df.columns:
            df.insert(0, "t", df.pop("t"))
        return df


def check_single_parameter_name(func: Callable, name: str):
    sig = signature(func)
    return len(sig.parameters) == 1 and name in sig.parameters


def agg(agg_function: Union[Callable, None]):
    """
    Register the storage function on a function.
    Used for storing aggregated results before cache eviction to reduce memory consumption.
    """

    def decorator(func: Callable):
        func.agg_function = agg_function # type: ignore
        return func

    return decorator


def _get_method_cache_factory(is_agg: bool):
    if is_agg:
        accessor: Callable[[CacheMethod], Dict] = lambda method: method.cache_agg
    else:
        accessor: Callable[[CacheMethod], Dict] = lambda method: method.cache
    return accessor

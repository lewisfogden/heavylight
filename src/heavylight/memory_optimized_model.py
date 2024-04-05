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
    """Inheriting from this class causes functions starting with a lowercase letter to be cached. 
    The caches can be cleared to rerun the same instance with different data. 
    This model can be memory optimized to reduce the memory requirements of the model without impacting results or performance. 
    """

    def __init__(self, agg_function: Union[Callable, None] = default_agg_function):
        """**Arguments**

        `agg_function`: Aggregation function for storing results. This function is applied to the return values of each method.
         In memory optimized models the cache is cleared but the aggregated results are stored for reporting. 
         If `agg_function` is not provided, the default aggregation function is used.
         If `agg_function` is `None`, no aggregated results will be provided unless overridden at a method level with `agg`.

        The implementation of the default aggregation function is as follows:

        ```python
        def default_agg_function(x: Any):
            if isinstance(x, np.ndarray) and issubclass(x.dtype.type, np.number):
                return np.sum(x)
            return x
        ```
        """
        self._cache_graph = CacheGraph()
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
            cached_method = self._cache_graph(method_agg_function)(method)
            self._funcs[method] = cached_method
            is_single_param_t = check_single_parameter_name(method, "t")
            if is_single_param_t:
                self._single_param_timestep_funcs.append(cached_method)
            setattr(self, method_name, cached_method)

    def RunModel(self, proj_len: int):
        """**Arguments**

        - `proj_len`: Projection length. All single parameter timestep functions will be run for each timestep in `range(proj_len + 1)`.
        """
        for t in range(proj_len + 1):
            for func in self._single_param_timestep_funcs:
                # We avoid recalling any functions that have already been cached, resolves issue #15 lewisfogden/heavylight
                if (
                    not FunctionCall(func.func.__name__, (t,), frozenset())
                    in self._cache_graph.all_calls
                ):
                    func(t)

    def Clear(self):
        """Clears the cache. If the model was memory optimized, it is no longer memory optimized."""
        self._cache_graph.reset()

    def ClearOptimize(self):
        """Clears the cache. The model is memory optimized after this function is called."""
        self._cache_graph.optimize_and_reset()

    @property
    def _timestep_functions(self):
        """List the cached functions that have a single parameter `t`."""
        return [cache.func.__name__ for cache in self._single_param_timestep_funcs]

    @property
    def cache(self):
        """
        This is the cache of the model. It is a dictionary of dictionaries. 
        The outer dictionary is keyed by the function name and the inner dictionary is keyed by the arguments.
        """
        return self._cache_graph.cache

    @property
    def cache_agg(self):
        """
        The `cache` property with the `agg_function` applied to the results.
        """
        return self._cache_graph.cache_agg

    @property
    def df(self):
        """
        A dataframe with the `cache` values of all functions that have a single parameter `t`.
        """
        return self._ToDataFrame(is_agg=False)

    @property
    def df_agg(self):
        """
        The `df` property dataframe with the `agg_function` applied to the results.
        """
        return self._ToDataFrame(is_agg=True)

    def _ToDataFrame(self, param_name="t", is_agg=False):
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
    """Register the storage function of a method.
    
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

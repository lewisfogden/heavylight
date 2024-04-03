from typing import Callable
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Tuple, Union, FrozenSet
import pandas as pd

@dataclass(frozen=True)
class FunctionCall:
    func_name: str
    args: tuple
    kwargs: FrozenSet[Tuple[str, Any]]

    def __repr__(self):
        if len(self.kwargs) == 0:
            if len(self.args) == 1:
                return f"{self.func_name}({self.args[0]})"
            return f"{self.func_name}{self.args}"
        return f"{self.func_name}({', '.join(map(str, self.args))}, {', '.join(f'{k}={v}' for k, v in self.kwargs)})"

ArgsHash = Tuple[Tuple, frozenset]

class CacheGraph:
    """
    The cache graph maintains data structures necessary for caching and memory-optimizing collections of recursive functions.
    It is applied to functions as a decorator
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Clear all internal state of the cache graph.
        """
        self.stack: list[FunctionCall] = [] # what function is currently being called
        self._caches: defaultdict[str, dict[ArgsHash, Any]] = defaultdict(dict) # Results of function calls, ugly keys like ((1, 2), frozenset([('a', 1)]))
        self._caches_agg: defaultdict[str, dict[ArgsHash, Any]] = defaultdict(dict)
        self.graph: defaultdict[FunctionCall, set[FunctionCall]] = defaultdict(set) # Call graph, graph[caller] = [callee1, callee2, ...]
        # Typically aggregated results for a function at a timestep.
        # What is the last function that needs the result of a function? Used to help in clearing the cache
        self.last_needed_by: dict[FunctionCall, FunctionCall] = {}
        # can_clear[caller] = [callee1, callee2, ...] means that caller can clear the cache of callee1 and callee2
        self.can_clear: dict[FunctionCall, list[FunctionCall]] = defaultdict(list)
        self.all_calls: set[FunctionCall] = set()
        self.cache_misses: defaultdict[FunctionCall, int] = defaultdict(int)

    def check_if_cached(self, function_call: FunctionCall):
        name_in_cache = function_call.func_name in self._caches
        return name_in_cache and (function_call.args, function_call.kwargs) in self._caches[function_call.func_name]
    
    def optimize(self):
        self.can_clear = defaultdict(list)
        for callee, caller in self.last_needed_by.items():
            self.can_clear[caller].append(callee)
        uncleared_calls = self.all_calls - set(self.last_needed_by.keys())
        for call in uncleared_calls:
            self.can_clear[call].append(call)

    def optimize_and_reset(self):
        self.optimize()
        can_clear = self.can_clear
        self.reset()
        self.can_clear = can_clear

    def __call__(self, storage_func: Union[Callable[[int], Any], None] = None):
        def custom_cache_decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                frozen_kwargs = frozenset(kwargs.items())
                function_call = FunctionCall(func.__name__, args, frozen_kwargs)
                if self.stack:
                    self.graph[self.stack[-1]].add(function_call)
                    self.last_needed_by[function_call] = self.stack[-1]
                if not self.check_if_cached(function_call):
                    self.all_calls.add(function_call)
                    self.cache_misses[function_call] += 1
                    self.stack.append(function_call)
                    result = func(*args, **kwargs)
                    self._caches[func.__name__][(args, frozen_kwargs)] = result
                    for clearable_call in self.can_clear[function_call]:
                        del self._caches[clearable_call.func_name][(clearable_call.args, clearable_call.kwargs)]
                    self.stack.pop()
                    self._store_result(storage_func, func, (args, frozen_kwargs), result)
                    return result
                return self._caches[func.__name__][(args, frozen_kwargs)]
            decorator = CacheMethod(self, wrapper, storage_func)
            return decorator
        return custom_cache_decorator
    
    def _store_result(self, storage_func: Union[Callable, None], func: Callable, args_hash: ArgsHash, raw_result: Any):
        """We might want to store an intermediate result"""
        if storage_func is None:
            return
        stored_result = storage_func(raw_result)
        self._caches_agg[func.__name__][args_hash] = stored_result

    def size(self):
        return sum(len(cache) for cache in self._caches.values())
    
    @property
    def cache(self):
        caches = defaultdict(dict, {func_name: {get_pretty_key(k): v for k, v in cache.items()} for func_name, cache in self._caches.items()})
        return caches
    
    @property
    def cache_agg(self):
        caches = defaultdict(dict, {func_name: {get_pretty_key(k): v for k, v in cache.items()} for func_name, cache in self._caches_agg.items()})
        return caches

class CacheMethod:
    def __init__(self, cache_graph: CacheGraph, func: Callable, agg_func: Union[Callable, None] = None):
        self.cache_graph = cache_graph
        self.func = func
        self.agg_func = agg_func

    @property
    def df(self):
        return pd.DataFrame({self.func.__name__: self.cache})

    @property
    def df_agg(self):
        return pd.DataFrame({self.func.__name__: self.cache_agg})

    # only run the dictionary comprehension for the particular method we want to access
    # simply returning self.cache_graph.caches[self.func.__name__] would run the dictionary comprehension for all methods
    @property
    def cache(self):
        return {get_pretty_key(k): v for k, v in self._cache.items()}
    
    @property
    def cache_agg(self):
        return {get_pretty_key(k): v for k, v in self._cache_agg.items()}

    @property
    def _cache(self):
        return self.cache_graph._caches[self.func.__name__]

    @property
    def _cache_agg(self):
        return self.cache_graph._caches_agg[self.func.__name__]

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self._cache[((key,), frozenset())] = value
        else:
            self._cache[(key, frozenset())] = value

    def __repr__(self):
        return f"<Cache Function: {self.func.__name__}, Size: {len(self._cache)}>"
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)
    
def get_pretty_key(key: ArgsHash):
    if len(key[0]) == 1 and len(key[1]) == 0:
        return key[0][0]
    elif len(key[1]) == 0:
        return key[0]
    else:
        return key
    
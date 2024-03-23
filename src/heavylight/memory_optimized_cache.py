from typing import Callable
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Tuple, Union, FrozenSet

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
        self.caches: defaultdict[str, dict[ArgsHash, Any]] = defaultdict(dict) # Results of function calls
        self.graph: defaultdict[FunctionCall, set[FunctionCall]] = defaultdict(set) # Call graph, graph[caller] = [callee1, callee2, ...]
        # Typically aggregated results for a function at a timestep.
        self.stored_results: defaultdict[str, dict[int, Any]] = defaultdict(dict)
        # What is the last function that needs the result of a function? Used to help in clearing the cache
        self.last_needed_by: dict[FunctionCall, FunctionCall] = {}
        # can_clear[caller] = [callee1, callee2, ...] means that caller can clear the cache of callee1 and callee2
        self.can_clear: dict[FunctionCall, list[FunctionCall]] = defaultdict(list)
        self.all_calls: set[FunctionCall] = set()
        self.cache_misses: defaultdict[FunctionCall, int] = defaultdict(int)

    def check_if_cached(self, function_call: FunctionCall):
        name_in_cache = function_call.func_name in self.caches
        return name_in_cache and (function_call.args, function_call.kwargs) in self.caches[function_call.func_name]
    
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
                    self.caches[func.__name__][(args, frozen_kwargs)] = result
                    for clearable_call in self.can_clear[function_call]:
                        del self.caches[clearable_call.func_name][(clearable_call.args, clearable_call.kwargs)]
                    self.stack.pop()
                    self._store_result(storage_func, func, args, kwargs, result)
                    return result
                return self.caches[func.__name__][(args, frozen_kwargs)]
            decorator = _Cache(self, wrapper)
            return decorator
        return custom_cache_decorator
    
    def _store_result(self, storage_func: Union[Callable, None], func: Callable, args: tuple, kwargs: dict, raw_result: Any):
        """We might want to store an intermediate result"""
        if storage_func is None:
            return
        # These conditions should not trigger, why we assert and not throw an exception
        assert len(args) == 1 and isinstance(args[0], int)
        assert len(kwargs) == 0
        # store the processed result
        timestep = args[0]
        stored_result = storage_func(raw_result)
        self.stored_results[func.__name__][timestep] = stored_result

    def size(self):
        return sum(len(cache) for cache in self.caches.values())

class _Cache:
    def __init__(self, cache_graph: CacheGraph, func: Callable):
        self.cache = cache_graph.caches[func.__name__]
        self._func = func

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.cache[((key,), frozenset())] = value
        else:
            self.cache[(key, frozenset())] = value

    def __repr__(self):
        return f"<Cache Function: {self._func.__name__}, Size: {len(self.cache)}>"
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._func(*args, **kwds)
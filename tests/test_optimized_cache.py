from heavylight.memory_optimized_cache import FunctionCall, CacheGraph
import pytest

def test_function_call():
    fc_args_kwargs = FunctionCall("func", (1, 2), frozenset([('a', 1)]))
    assert fc_args_kwargs.func_name == "func"
    assert fc_args_kwargs.args == (1, 2)
    assert fc_args_kwargs.kwargs == frozenset([('a', 1)])
    assert repr(fc_args_kwargs) == "func(1, 2, a=1)"
    fc_single_arg_no_kwargs = FunctionCall("func", (1,), frozenset())
    assert repr(fc_single_arg_no_kwargs) == "func(1)"
    fc_multiple_args_no_kwargs = FunctionCall("func", (1, "hello"), frozenset())
    assert repr(fc_multiple_args_no_kwargs) == "func(1, 'hello')"

def test_cache_graph_storage_function():
    cg = CacheGraph()
    @cg(lambda x: x**2)
    def fib(n):
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)
    fib(5)
    assert len(cg.cache_agg["fib"]) == 6
    for k, v in cg.cache_agg["fib"].items():
        assert v == fib(k)**2

def test_cache_dunders():
    cg = CacheGraph()
    @cg()
    def fib(n):
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)
    fib(5)
    assert repr(fib) == "<Cache Function: fib, Size: 6>"
    assert len(fib.cache) == 6
    test_key = 5
    assert fib.cache[test_key] == cg.cache['fib'][test_key] == 5
    assert fib.cache[5] == 5 # prettified keys
    fib[5] = 10
    assert fib.cache[test_key] == cg.cache['fib'][test_key] == 10
    assert fib.cache[5] == 10
    fib[(5,)] = 100
    assert fib.cache[test_key] == cg.cache['fib'][test_key] == 100

@pytest.mark.timeout(4)
def test_cache_graph_memory_optimization():
    cg = CacheGraph()

    @cg()
    def fib(n: int):
        if n <= 1:
            return n
        return fib(n-1) + fib(n-2)

    def get_max_memory_fib(n: int, cg: CacheGraph):
        max_memory_size = 0
        for i in range(n+1):
            fib(i)
            max_memory_size = max(max_memory_size, cg.size())
        return max_memory_size

    assert get_max_memory_fib(200, cg) == 201
    cg.optimize_and_reset()
    assert get_max_memory_fib(200, cg) == 2
from heavylight.cache_graph import FunctionCall, CacheGraph
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
    assert len(cg.stored_results["fib"]) == 6
    for k, v in cg.stored_results["fib"].items():
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
    assert fib.cache[((5,), frozenset())] == 5
    fib[5] = 10
    assert fib.cache[((5,), frozenset())] == 10
    fib[(5,)] = 100
    assert fib.cache[((5,), frozenset())] == 100

@pytest.mark.timeout(4)
def test_cache_graph_memoization_speeedup():
    cg = CacheGraph()
    @cg()
    def fib(n):
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)
    fib(200)
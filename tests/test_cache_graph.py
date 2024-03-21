from heavylight.cache_graph import FunctionCall

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
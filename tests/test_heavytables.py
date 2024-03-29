import numpy as np
import pandas as pd
import heavylight.heavytables as ht

def test_integer_lookup():
    lookup = ht.IntegerLookup()
    arr = np.array([1,2,3])
    assert np.all(arr==lookup.get(arr))

def test_bound_integer_lookup():
    lookup = ht.BoundIntLookup(0, 10)
    arr = np.array([-1, 0, 1, 10, 11])
    assert np.all(np.array([0, 0, 1, 10, 10])==lookup.get(arr))

def test_rectify():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [1, 2], 'c': [1, 2]})
    expected_rectified_df1 = pd.DataFrame({'a': [1, 1, 2, 2], 'b': [1, 2, 1, 2], 'c': [1, np.nan, np.nan, 2]})
    assert ht.Table.rectify(df1).equals(expected_rectified_df1)
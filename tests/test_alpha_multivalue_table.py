import numpy as np
import pandas as pd
import heavylight.heavytables as ht
import pytest
import heavylight.alpha_multivalue_table as amt

def test_multi_key_table():
    df = pd.DataFrame({'key1|int_cat': [2, 4, 2, 4], 'key2|int_bound': [1, 1, 2, 2]})
    df["val_a|value_int"] = df['key1|int_cat'] * df['key2|int_bound']
    df["val_b|value_float"] = df["val_a|value_int"] * 3.5
    table = amt.MultiColTable(df)

    assert table['val_a'][4, 2] == 4 * 2
    assert table['val_b'][4, 2] == 4 * 2 * 3.5

    assert table.val_a[4, 2] == 4 * 2
    assert table.val_b[4, 2] == 4 * 2 * 3.5

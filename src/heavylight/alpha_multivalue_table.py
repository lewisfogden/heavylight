from .heavytables import Table
import pandas as pd
import numpy as np

class MultiColTable:
    value_col_options = 'value_int', 'value_float', 'value_str'

    def __init__(self, df:pd.DataFrame):
        self.df = df
        self._tables = {}
        self.key_col_names = []
        self.value_col_names = []
        for col_idx, col_name_full in enumerate(self.df.columns):
            col_name, col_type = col_name_full.split("|", maxsplit=1)
            is_value_col = col_type in self.value_col_options
            if not is_value_col:
                self.key_col_names.append(col_name_full)
            else:
                sub_df = self.df.filter([*self.key_col_names, col_name_full])
                sub_table = Table(df=sub_df)
                self._tables[col_name] = sub_table
                self.value_col_names.append(col_name)
                setattr(self, col_name, sub_table)

    def __getitem__(self, key):
        return self._tables[key]
    
    def __repr__(self):
        col_names = "', '".join(self.value_col_names)
        key_names = "', '".join(self.key_col_names)
        return f"<MultiColTable columns=['{col_names}'], keys=['{key_names}']>"
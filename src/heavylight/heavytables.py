# Heavytables
# Provides a high performance multiple-index -> single index table

import itertools
import pandas as pd
import numpy as np

# Lookup classes
# Each key column requires an input class which maps from the source datatype to an integer

class IntegerLookup:
    """An integer lookup - unsafe as doesn't check bounds"""
    def get(self, values):
        return values

class IntegerLookupSafe:
    """An integer lookup that checks bounds"""
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
    
    def get(self, values):
        # TODO: replace np.all with np.any for faster failing (probably doesn't improve performance)
        if not np.all((self.lower <= values) & (values <= self.upper)):
            raise KeyError(f'values are not between table minimum ({self.lower}) and maximum ({self.upper})')
        return values

class BoundIntLookup:
    """An integer lookup that clips to the lowest and highest keys"""
    def __init__(self, lower, upper):
        self.lower = int(lower)
        self.upper = int(upper)
    
    def get(self, numpy_array):
        return np.clip(numpy_array, self.lower, self.upper)

class StringLookup:
    """A string lookup that include validation that the string match is exact"""
    def __init__(self, string_vals):
        self.string_vals = np.array(string_vals)
    
    def get(self, keys):
        if isinstance(keys, np.ndarray):
            if not all(np.isin(keys, self.string_vals)):
                raise KeyError("invalid string key(s) passed into table lookup.")
        else:
            if not keys in self.string_vals:
                raise KeyError("invalid string key(s) passed into table lookup.")   
        return np.searchsorted(self.string_vals, keys)          

class BandLookup:
    """A lookup that matches on bands, using the np.searchsorted function"""
    def __init__(self, upper_bounds, labels):
        """Inputs must be sorted"""
        self.upper_bounds = np.array(upper_bounds)
        self.labels = np.array(labels)

    def get(self, numpy_array):
        """get the labels for vector"""
        indices = np.searchsorted(self.upper_bounds, numpy_array)
        return self.labels[indices]
    
    def get_max(self, numpy_array):
        """get the upper band limit"""
        indices = np.searchsorted(self.upper_bounds, numpy_array)
        return self.upper_bounds[indices]
    
    @classmethod
    def from_dataframe(cls, df:pd.DataFrame, band_column_name, integer_column_name):
        df_copy = df.sort_values(band_column_name)
        band_arr = df_copy[band_column_name].values
        band_labels = df_copy[integer_column_name].values
        return cls(band_arr, band_labels)
    
    @classmethod
    def from_excel_sheet(cls, workbook_path, sheet_name):
        """Extract from an excel sheet, there must be two columns, the band column and the integer column"""
        df = pd.read_excel(workbook_path, sheet_name=sheet_name)
        assert len(df.columns) == 2, "must have 2 columns"
        band_col_name, int_col_name = df.columns[0], df.columns[1]
        return cls.from_dataframe(df, band_col_name, int_col_name)

class IntKeyTable:
    """Vectorised Table Allowing for multiple integer keys"""
    def __init__(self, df: pd.DataFrame):
        self.key_cols = list(df.columns[:-1])
        self.value_col = df.columns[-1]

        # ensure all key_cols are integers
        for col in self.key_cols:
            df[col] = df[col].astype(int)

        self.df = df.sort_values(by=list(reversed(self.key_cols))) # TODO: don't need to store this in self, this but handy for testing.
        self.bases = [min(self.df[col]) for col in self.key_cols]
        self.ranges = [max(self.df[col]) - min(self.df[col]) + 1 for col in self.key_cols]

        # set up index scalars
        total_scalar = 1
        self.scalars = []
        for col in self.ranges:
            self.scalars.append(total_scalar)
            total_scalar *= col

        self.values = self.df[self.value_col].values

        expected_rows = np.prod(self.ranges)
        if expected_rows != len(self.values):
            raise ValueError(f'Input `df` is not rectangular, {expected_rows=} != {len(self.values)=}')

    def get_index(self, *keys):
        index = 0
        for key, base, scalar in zip(keys, self.bases, self.scalars):
            index += (key - base) * scalar
        return index
    
    def get_value(self, *keys):
        indices = self.get_index(*keys)
        return self.values[indices]
    
    def __getitem__(self, keys):
        return self.get_value(*keys)
    
class Table:
    """Table provides multi-key compatible high performance table lookup.
    """
    col_types = "int", "int_bound", "str", "band", "float"

    def __init__(self, df:pd.DataFrame, rectify=False, safe=True):
        """Initialise a table from a dataframe.
        
        parameters:
         - `rectify`: force table to be rectangular (default False)
         - `safe`: validates that integers are between bounds (default True)

        Tables should be in long format:
         - the final column containing the values to look up
         - all other columns contain keys to lookup
         - tables should be contingous, i.e. no gaps in integer keys.
         - tables should be complete if viewed as square matrixes (i.e. all combinations of keys are input).  If not, you should fill any gaps with np.nan or a suitable value.

        The type of key is determined by the suffix on the column name:
        `|int`: integers (...0, 1, 2, 3...), can start and end anywhere, but must be consecutive
        `|int_bound`: as `|int` but any values are constrained to the lowest and highest values.
        `|str': keys are interpreted as strings, e.g. 'M' and 'F'
        `|band`: key is numeric and treated as the upper bound on a lookup.
        `|float`: not currently available due to floating point equality, use int or band depending on use case.
        """
        if rectify:
            df = Table.rectify(df)

        key_cols = list(df.columns[:-1])
        df = df.sort_values(key_cols[::-1]) # sort by reverse order

        df_int_keys = df.copy() # this will have keys overridden as we work through mappers

        # prepare the mappers
        self.mappers = []
        for col in key_cols:
            col_type = col.split("|")[1] # "int", "str" etc
            if col_type == "int":
                if safe:
                    lower = df[col].min()
                    upper = df[col].max()                    
                    self.mappers.append(IntegerLookupSafe(lower=lower, upper=upper))
                else:
                    self.mappers.append(IntegerLookup())
            elif col_type == "int_bound":
                # bound integer forces values to be between the lowest and highest value in the table, for example maximum durations in mortality tables.
                lower = df[col].min()
                upper = df[col].max()
                self.mappers.append(BoundIntLookup(lower=lower, upper=upper))
            elif col_type == "str":
                cols = df[col].unique()
                string_mapper = StringLookup(cols)
                self.mappers.append(string_mapper)
                df_int_keys[col] = string_mapper.get(df_int_keys[col])

            elif col_type in ["str_unsafe", "band"]:
                df_col = pd.DataFrame(df[col].unique(), columns=["band_name"]).reset_index().sort_values("band_name")

                # add a nan on the end so we get errors if the lookup fails
                # as by default the BandLookup will return last item if no earlier matches
                # commented out as it converts the datatype - we need these to be integers
                # df_col.loc[len(df_col)] = len(df_col), np.nan
                band_mapper = BandLookup.from_dataframe(df_col, "band_name", "index")
                self.mappers.append(band_mapper)
                df_int_keys[col] = band_mapper.get(df_int_keys[col])
            else:
                raise NotImplementedError(f"{col_type} not implemented on {col}")
        
        # create an intkeytable

        self._int_key_table = IntKeyTable(df_int_keys)
        self.df = df # store for referencing TODO: decide if this should be removed.
    
    def get(self, *keys):
        assert len(keys) == len(self.mappers)
        # TODO: if just one key then this doesn't work?  (needs fixed throughout)
        int_keys = [mapper.get(key) for key, mapper in zip(keys, self.mappers)]
        return self._int_key_table.get_value(*int_keys)

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = keys,    #force to be a tuple
        return self.get(*keys)

    def __repr__(self):
        # TODO: return a nice representation of the table, e.g. head/keys etc.
        return self.df.__repr__()
    
    @staticmethod
    def rectify(df: pd.DataFrame, fill=np.nan) -> pd.DataFrame:
        """Convert a triangular (incomplete) dataframe into a valid rectangular dataframe
        
        any missing points will be filled with `fill`, default: np.nan
        """
        key_cols = list(df.columns[:-1])
        val_col = df.columns[-1]

        df_unique_keys = []
        for col_name in key_cols:
            if col_name.endswith('|int') or col_name.endswith('|int_bound'):
                # make sure all integers are covered (no gaps)
                full_keys = list(range(df[col_name].min(), df[col_name].max() + 1))
            else:
                # otherwise, grab the unique values.
                full_keys = list(df[col_name].unique())
            df_unique_keys.append(full_keys)

        # construct a dataframe from the cartesian product of these
        df_rect_keys = pd.DataFrame(itertools.product(*df_unique_keys), columns=key_cols)

        # fill out the table
        df_rect = df_rect_keys.merge(right=df, how='left', on = key_cols)

        # replace gaps with the fill value.
        df_rect[val_col] = df_rect[val_col] .fillna(fill)
        return df_rect

    @classmethod
    def read_excel(cls, spreadsheet_path, sheet_name):
        """Read in a table from an excel sheet, for more control pass in the dataframe using `__init__`"""
        df = pd.read_excel(spreadsheet_path, sheet_name=sheet_name)
        return cls(df)
    
    @classmethod
    def read_csv(cls, csv_path):
        """Read in a table from an csv file, for more control pass in the dataframe using `__init__`"""
        df = pd.read_csv(csv_path)
        return cls(df)

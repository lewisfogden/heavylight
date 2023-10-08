import types
from typing import List
import warnings
from inspect import signature
import pandas as pd

class Table:
    """A Table has one or more keys, and a single column of values"""

    def __init__(self, series):
        """Initialise a table with a series - for multiple keys use a multikey index"""
        self.series = series

    def __getitem__(self, key):
        return self.series.at[key]
    
    def __call__(self, key):
        return self.series.at[key]
    
    def values(self, key):
        return self.series.loc[key].values

    @staticmethod
    def from_csv(filename, sep=",", type_identifier = "|"):
        """Read in a table from a csv file, type encoding is via the header:
        
        <keyname1>|<type1>,<keyname2>|<type2>....<value>|<typeN>
        where type is one of "str", "int", "float"
        """
        column_types = {"str":str, "int": int, "float": float}
        with open(filename, "r") as csv_file:
            header = next(csv_file).strip("\n").split(sep)
        tid = type_identifier
        header_mapper_str = {item:item.split(tid)[1] for item in header}
        header_mapper_types = {col:column_types[val] for col, val in header_mapper_str.items()}
        df = pd.read_csv(filename, sep=sep, dtype=header_mapper_types)
        # strip `tid` from column names
        df.columns = [col.split("|")[0] for col in df.columns]
        df.set_index(list(df.columns[:-1]), inplace=True)
        series = df[df.columns[0]]
        return Table(series=series)


class _Cache:
    """Cache provides controllable memoization for model methods"""

    def __init__(self, func: types.MethodType, param_len: int):
        self.func = func
        self.param_len = param_len
        self.has_one_param = self.param_len == 1
        self.values = dict()
        self.__name__ = "Cache: " + func.__name__

    def __call__(self, *arg:list):
        if arg in self.values:
            return self.values[arg]
        else:
            result = self.func(*arg)
            self.values[arg] = result
            return result

    def __repr__(self) -> str:
        return f"<Cache Function: {self.func.__name__} Size: {len(self.values)}>"

class Model:
    def __init__(self, *, do_run: bool = False, verbose=False, proj_len: int|types.NoneType = None, **kwargs,):
        """Base Class to subclass for user models.

        All variables/methods in user models should be lower case, using underscore as spaces.

        Class level methods:
          RunModel(proj_len):
            if the model has not been auto-run at initialisation, run it for projection length.

        Special user methods:
          BeforeRun(self):
            If this is specified in the user model it called before the projection starts, e.g. to set up some specific variables

          AfterRun(self):
            user method, called after Run is completed, e.g. can use to calculate NPVs of variables

        methods/variables to avoid:
        methods/variables starting with an underscore `_` are treated as internal.  You may break functionality if you create your own.

        """

        self._cached = False
        self._is_run = False
        if verbose:
            print("== Run Parameters ==")
            print("    do_run:", do_run)
            print("    proj_len:", proj_len)
            print()
        
        if do_run:
            if not isinstance(proj_len, int):
                raise ValueError("proj_len must be an integer")
            elif proj_len <= 0:
                raise ValueError("proj_len must have value greater than 0")
            self.proj_len = proj_len
        else:
            if verbose: print("== Not Running - call Run() manually ==")
       
        if verbose: print("== Storing Arguments ==")
        for k, v in kwargs.items():
            if k in dir(self):
                warnings.warn("Warning: Duplicate Item: "+str(k))
            setattr(self, k, v)
            if verbose:
                print("    Updated: ", k, " : ", v)

        # cacheify
        if verbose: print("== Caching Functions ==")
        self._cache_funcs(verbose)

        if do_run and proj_len > 0:
            self.RunModel(proj_len, verbose)
        if verbose: print("== Run complete == ")


    def RunModel(self, proj_len, verbose=False):
        if self._is_run:
            raise ValueError("Run has already been completed.")

        if verbose: print(f"== Running Projection | length: {proj_len} ==")
        
        if hasattr(self, "BeforeRun"):
            if verbose: print("    Calling BeforeRun")
            self.BeforeRun()

        if not self._cached:
            raise ValueError("Functions have not been cached")  # NB: this shouldn't occur as now caching in instance
        for t in range(proj_len):
            for var in self._funcs.keys():
                func = getattr(self, var)
                if func.has_one_param:   # skip functions with more than one parameter
                    func(t)   #call each function in turn, starting from t==0
        self._is_run = True
        if hasattr(self, "AfterRun"):
            if verbose: print("    Calling AfterRun")
            return self.AfterRun()
    
    def _cache_funcs(self, verbose : bool=False):
        if self._cached:
            raise ValueError("Cache has already been set-up, please create a new instance")

        self._funcs = {}

        for method_name in dir(self):
            method = getattr(self, method_name)

            if method_name[0] != "_" and method_name[0].islower() and isinstance(method, types.MethodType):
                param_count = len(signature(method).parameters) # count the parameters in the function.
                cached_method = _Cache(method, param_count)
                setattr(self, method_name, cached_method)
                self._funcs[method_name] = cached_method
                if verbose: print(f"    Cached: {method_name}")

        
        self._cached = True
        
    def ToDataFrame(self):
        """return a pandas dataframe of all single parameter columns"""
        df = pd.DataFrame()
        for func in self._funcs:
            if self._funcs[func].has_one_param:
                df[func] = pd.Series(self._funcs[func].values)
        return df
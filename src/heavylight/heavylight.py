import warnings
import types
from inspect import signature, getmembers
import pandas as pd

class _Cache:
    """Cache provides controllable memoization for model methods"""

    def __init__(self, func, param_len, param_names):
        self.func = func
        self.param_len = param_len
        self.param_names = param_names
        self.has_one_param = self.param_len == 1
        self._store = dict()
        self.__name__ = func.__name__

    def __call__(self, *arg):
        if arg in self._store:
            return self._store[arg]
        else:
            result = self.func(*arg)
            self._store[arg] = result
            return result

    def __repr__(self):
        return f"<Cache Function: {self.func.__name__} Params: {self.param_names} Size: {len(self._store)}>"
    
    def sum(self):
        """return the sum of all values in the Cache Function"""
        return sum(self._store.values())
    
    @property
    def keys(self):
        return list(self._store.keys())

    @property
    def values(self):
        return list(self._store.values())
    
    @property
    def df(self):
        """return the cache as a pandas dataframe"""
        df = pd.DataFrame(data=self.keys, columns=self.param_names)
        df[self.__name__] = self.values
        # df = df.set_index(list(self.param_names))  # TODO: decide if keys should be indexes or not.
        return df

class Model:
    def __init__(self, *, do_run = None, proj_len:int = 0, **kwargs,):
        """Base Class to subclass for user models.

        When the model is instanced it is run to proj_len, if non-zero.

        Parameters
        ----------
        - proj_len: length of projection to run
        

        All variables/methods in user models should be lower case, using underscore as spaces.

        Class level methods:
          RunModel(proj_len):
            Run the model to proj_len.

        Special user methods:
          BeforeRun(self):
            If this is specified in the user model it called before the projection starts, e.g. to set up some specific variables

          AfterRun(self):
            user method, called after Run is completed, e.g. can use to calculate NPVs of variables

        methods/variables to avoid:
        methods/variables starting with an underscore `_` are treated as internal.  You may break functionality if you create your own.

        """

        if do_run is not None:
            warnings.warn("Warning: `do_run` will be removed in a future version, use `proj_len` to control projection")
        else:
            do_run = proj_len > 0

        self._cached = False
        self._is_run = False
        
        if do_run:
            # TODO: this can be cleaned up once do_run is removed
            if not isinstance(proj_len, int):
                raise ValueError("proj_len must be an integer")
            elif proj_len < 0:
                raise ValueError("proj_len must be non-negative")
            self.proj_len = proj_len

        # store all keyword arguments as attributes
        for k, v in kwargs.items():
            if k in dir(self):
                # TODO: include a strict mode that raises on duplicates?
                warnings.warn("Warning: Duplicate Item: "+str(k))
            setattr(self, k, v)

        # cacheify
        self._cache_funcs()

        if do_run and proj_len > 0:
            self.RunModel(proj_len)

    def RunModel(self, proj_len: int):
        """
        Run the model if not already run.

        Parameters
        ----------
        - proj_len: length of projection to run"""
        if self._is_run:
            # TODO: replace this with ability to run further, but warn that earlier values not recalculated?
            raise ValueError("Run has already been completed.")
        
        if hasattr(self, "BeforeRun"):
            self.BeforeRun()

        if not self._cached:
            raise ValueError("Functions have not been cached")  # NB: this shouldn't occur as now caching in instance
        for t in range(proj_len):
            for name, func in self._funcs.items():
                if func.has_one_param and func.param_names[0] == 't':   # skip functions with more than one parameter
                    func(t)   #call each function in turn, starting from t==0
        self._is_run = True

        if hasattr(self, "AfterRun"):
            return self.AfterRun()
    
    def _cache_funcs(self):
        if self._cached:
            raise ValueError("Cache has already been set-up, please create a new instance")

        self._funcs = {}

        for method_name, method in getmembers(self):
            if method_name[0] != "_" and method_name[0].islower() and isinstance(method, types.MethodType):
                param_count = len(signature(method).parameters) # count the parameters in the function.
                param_names = tuple(signature(method).parameters)
                cached_method = _Cache(method, param_count, param_names)
                setattr(self, method_name, cached_method)
                self._funcs[method_name] = cached_method
       
        self._cached = True
    
    def _info(self):
        """Print info about the model"""
        for name, func in self._funcs.items():
            print(f"{name}: {func}")
        
    def ToDataFrame(self, param = 't'):
        """return a pandas dataframe of all single parameter columns
        
        Parameters
        ----------
        - param: parameter to filter on.  Default: `t`
        """
        df = pd.DataFrame()
        for func in self._funcs:
            if self._funcs[func].has_one_param and self._funcs[func].param_names[0] == param:
                df[func] = pd.Series(self._funcs[func].values)

        # if t is in the dataframe, move it to first position
        if "t" in df.columns:
            df.insert(0, "t", df.pop("t"))
        return df
    
    @property
    def df(self):
        """return a pandas dataframe of all single parameter columns parameterised with `t`"""
        return self.ToDataFrame()
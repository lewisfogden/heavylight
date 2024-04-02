# I would really rather just put this in the docs instead of trying to make it part of the framework. users should know how the api works so this forces them to.

def default_agg_function(x: Any):
    if isinstance(x, (int, float, np.number)):
        return x    
    if isinstance(x, np.ndarray) and issubclass(x.dtype.type, np.number):
        return np.sum(x)
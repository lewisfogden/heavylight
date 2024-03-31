# %%
import heavylight
import pandas as pd
import numpy as np

# A Vectorised Actuarial Contingency Model

# This code is out of date/unmaintained but includes some examples of usage.

# %%
class Life(heavylight.Model):
    time_step = 1/12

    def t(self, t):
        return np.ones(self.data["data_count"]) * t
    
    def num_pols_if(self, t):
        if t == 0:
            return self.data["initial_policies"]
        else:
            return self.num_pols_if(t - 1) - self.num_deaths(t - 1)
    
    def num_deaths(self, t):
        if t == 0:
            return np.zeros(self.data["data_count"])
        else:
            return self.num_pols_if(t) * self.q_x_m(t)
    
    def age(self, t):
        """age at time t"""
        if t == 0:
            return self.data["initial_age"] # floating point
        else:
            return self.age(t - 1) + Life.time_step
    
    def age_rounded(self, t):
        return np.round(self.age(t)).astype(int)
    
    def q_x_m(self, t):
        """monthly mortality rate"""
        return self.q_x(t) ** (Life.time_step)
    
    def q_x(self, t):
        """annual mortality rate"""
        return self.basis["mortality"][self.age_rounded(t)]
    
    def premium(self, t):
        #print(self.data["premium"])
        #print(self.num_pols_if(t))
        return self.data["premium"] * self.num_pols_if(t)
    
    def claim(self, t):
        #print(f"claim: {len(self.num_deaths(t))=}")
        #print(f"claim: {len(self.data['sum_assured'])=}")
        return self.num_deaths(t) * self.data["sum_assured"]

    def net_cashflow(self, t):
        return self.premium(t) - self.claim(t)
    
    def mpnum(self, t):
        return self.data["mp_num"]


# %%
mortality = heavylight.Table.read_csv("sample_q_x_table.csv")

# %%
basis = {"mortality": mortality}

# %%


def make_random_data(num_pols):
    rng = np.random.default_rng(seed=42)
    return_data = {}
    return_data["data_count"] = num_pols
    return_data["initial_policies"] = np.ones(num_pols)
    return_data["mp_num"] = np.arange(num_pols)
    return_data["initial_age"] = rng.uniform(low=20.0, high=25.0, size=num_pols)
    return_data["premium"] = rng.beta(a=2, b=5, size=num_pols) * 100 + 50
    return_data["sum_assured"] = return_data["premium"] * 100 # rng.uniform(low=10.0, high=20, size=num_pols)
    return return_data

"""
data = {
    "initial_policies": np.ones(num_model_points),
    "initial_age" : np.array([32+1/12, 42, 25+7/12]),
    "premium": np.array([100.65, 500, 220.34]),
    "sum_assured": np.array([10000, 30000, 25000]),
    }
"""

# num_model_points = 3


# %%
data = make_random_data(1000)
model = Life(data=data, basis=basis, proj_len=10) #, do_run=True, proj_len = 5)



# %%
def expand(model, variable):
    """return a variable from the model as a dataframe
    rows = time
    columns = variables
    """
    temp_df = pd.DataFrame(getattr(model, variable).values).T
    temp_df.index.name = "t"
    return temp_df
# %%
def get_single_result(model, index):
    temp_df = model.ToDataFrame()
    # TODO: check type of each column
    # TODO: check that index is not out of bounds
    for column in temp_df:
        if isinstance(temp_df[column].iloc[0], np.ndarray):
            temp_df[column] = temp_df[column][index]
    return temp_df
    #return temp_df.applymap(lambda x: x[index])

# %%

def dfize(model):
    columns = {}
    for func in model._funcs:
        func_values = getattr(model,func).values
        if isinstance(func_values.values[0], np.ndarray):
            # multi-index
            temp_df = pd.DataFrame(func_values).T.stack()
            columns[func] = temp_df
        else:
            print(f"Still to deal with {func}")
    return pd.concat(columns, axis=1)


    # convert 

# %%
df = dfize(model)
df[df.mpnum==500].plot(x="t", y="net_cashflow")  # can filter on a model point
# %%
def npv(rate, values):
    values = np.atleast_2d(values)
    timestep_array = np.arange(0, values.shape[1])
    npv = (values / (1 + rate) ** timestep_array).sum(axis=1)
    try:
        # If size of array is one, return scalar
        return npv.item()
    except ValueError:
        # Otherwise, return entire array
        return npv
# %%
print("npv", npv(0.08, df[df.mpnum==500]["net_cashflow"]))
# %%
npvs_4pc = df.groupby("mpnum").agg({"net_cashflow": lambda x: npv(0.04, x)})
# %%

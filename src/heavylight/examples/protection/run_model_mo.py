# -*- coding: utf-8 -*-
# required libraries:
# heavylight
# pip install heavylight

# %%
import heavylight
from heavylight import Table
from protection_model_mo import TermAssurance
import numpy as np
import pandas as pd


# %%

basis = {
    "cost_inflation_pa": 0.02,
    "initial_expense": 500,
    "expense_pp": 10,
    "lapse_rate_pa": 0.1,
    "mort_table": Table.read_csv(r"tables/q_x_generic.csv"),
    "forward_rates": Table.read_csv(r"tables/forward_rates.csv"),
}

def create_data(pols, seed=42):

    rng = np.random.default_rng(seed)

    # override the single datapoint `pols`
    data = dict(
        sum_assured = rng.integers(10_000, 250_000, pols),
        age_at_entry = rng.integers(20, 50, pols),
        term_y = rng.integers(10, 30, pols),
        smoker_status = rng.choice(['S', 'N'], pols),
        shape = rng.choice(['level', 'decreasing'], pols),
        annual_premium = np.ones(pols),
        init_pols_if = np.ones(pols),
        extra_mortality = np.zeros(pols),
        sex = rng.choice(['F', 'M'], pols),
    )
    return data

## with LightModel we can run unoptimised, then run optimised.

opt_data = create_data(5) # small data set to run the optimiser
data = create_data(100_000) # full data set

proj = TermAssurance()
proj.data = opt_data
proj.basis = basis
proj.RunModel(240) # run unoptimised
proj.data = data # replace data
proj.RunOptimized() # run optimised
proj.df_agg
# %%
    

# -*- coding: utf-8 -*-
# required libraries:
# heavylight
# pip install heavylight

# %%
import heavylight
from heavylight import Table
from protection_model import TermAssurance

def solve_prot_premium(model: heavylight.Model, data: dict, basis: dict):
    
    data = data.copy()

    pricing_entries = {
            "annual_premium":1,
            "init_pols_if":1,
            }
    
    data.update(pricing_entries)

    proj_len = data["term_y"] * 12 + 1

    model_inst = model(data=data, basis=basis, proj_len=proj_len)
            
    # extract npvs
    npv_claims = model_inst.pv_claims.sum()
    npv_expenses = model_inst.pv_expenses.sum()
    npv_premiums = model_inst.pv_premiums.sum()
            
    # calculate premium
    
    annual_risk_premium = (npv_claims + npv_expenses) / npv_premiums
    monthly_risk_premium = annual_risk_premium / 12
    return round(monthly_risk_premium, 2)

# %%
if __name__=='__main__':
    
    basis = {
        "cost_inflation_pa": 0.02,
        "initial_expense": 500,
        "expense_pp": 10,
        "lapse_rate_pa": 0.1,
        "mort_table": Table.read_csv(r"tables/q_x_generic.csv"),
        "forward_rates": Table.read_csv(r"tables/forward_rates.csv"),
    }

    data = {
        "sum_assured": 100_000,
        "age_at_entry": 49,
        "term_y": 35,
        "smoker_status": "N",
        "shape": "level",
        "annual_premium": 1,
        "init_pols_if": 1,
        "extra_mortality": 0,
        "sex": "F"
    }

    model = TermAssurance(basis=basis, data=data, proj_len=data["term_y"]*12 + 12)
  
    monthly_premium = solve_prot_premium(TermAssurance, data, basis)
    print("Premium: ", monthly_premium)
   

    

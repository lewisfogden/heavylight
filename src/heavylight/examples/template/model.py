# a minimal example
# %%
import heavylight

class ModelName(heavylight.Model):
    def t(self, t):
        return t
    
    # define user methods here
    def net_cashflow(self, t):
        """expected net cashflow occuring at time t"""
        if t == 0:
            return -100
        else:
            return self.monthly_premium(t) - 5
    
    def monthly_premium(self, t):
        return self.data['annual_premium'] / 12
    
    def forward_rate_pm(self, t):
        """monthly forward rate applying from time t-1 to time t (or however you define)"""
        return (1 + self.forward_rate_pa) ** (1/12) - 1
    
    def v(self, t):
        """discount factor from time t to time 0"""
        if t == 0:
            return 1.0
        else:
            return self.v(t - 1) / (1 + self.forward_rate_pm(t))
    
    def pv_net_cashflow(self, t):
        """present value (at t=0) of the cashflow occuring at time t"""
        return self.net_cashflow(t) * self.v(t)

# data for model.

data = {
    'annual_premium': 84,
}
# run model, we could store forward rate in another dictionary (like data), but for simplicity we can pass it in directly.
proj = ModelName(proj_len=5 * 12, data=data, forward_rate_pa=0.04)

# output the present value
print(f"Present Value of Net Cashflows: {proj.pv_net_cashflow.sum():=.2f}\n")

# view a dataframe of the results
print(proj.df.head())

# %%

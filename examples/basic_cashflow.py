# %%

from heavylight import Model

# %%

class SimpleModel(Model):
    def t(self, t):
        return t

    def num_pols_if(self, t):
        if t == 0:
            return 1
        else:
            return self.num_pols_if(t - 1) * 0.98

    def cashflow(self, t):
        return self.num_pols_if(t) * 100
    
    def v(self, t):
        """discount factor for time t --> time 0"""
        if t == 0:
            return 1
        else:
            return self.v(t - 1) / (1 + self.forward_rate(t))
    
    def forward_rate(self, t):
        return 0.04
    
    def pv_cashflow(self, t):
        """present value of the cashflow occuring at time t"""
        return self.cashflow(t) * self.v(t)
    
# %%

if __name__ == "__main__":
    model = SimpleModel(proj_len = 10)
    print(model.pv_cashflow.sum())

    print(model.v.values)

    print(model.df)

    print(model.pv_cashflow.df)
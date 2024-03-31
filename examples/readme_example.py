# %%

import heavylight

class Annuity(heavylight.Model):
    def t(self, t):
        return t

    def expected_claim(self, t):
        return self.number_alive(t) * self.data["annuity_per_period"]

    def number_alive(self, t):
        if t == 0:
            return self.data["initial_policies"]
        else:
            return self.number_alive(t - 1) - self.deaths(t - 1)
    
    def deaths(self, t):
        return self.number_alive(t) * self.mortality_rate(t)

    def mortality_rate(self, t):
        return 0.02

    def v(self, t):
        """discount factor from time t to time 0"""
        if t == 0:
            return 1
        else:
            return self.v(t - 1) / (1 + self.forward_rate(t))
    
    def forward_rate(self, t):
        return 0.04

    def pv_expected_claim(self, t):
        return self.expected_claim(t) * self.v(t)
    
# %%

policy_data = {
    "initial_policies": 10,
    "annuity_per_period": 55,
}

# %%

model = Annuity(data = policy_data,
                do_run = True,
                proj_len = 20,
                )

print(model.pv_expected_claim.sum())

model_cashflows = model.df
print(model_cashflows)
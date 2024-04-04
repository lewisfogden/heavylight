import heavylight

class TermAssurance(heavylight.Model):
    def t(self, t):
        return t

    def net_cf(self, t):
        return self.premiums(t) - self.claims(t) - self.expenses(t)
    
    def premium_pp(self, t):
        """monthly premium"""
        return self.data["annual_premium"] / 12
    
    def claim_pp(self, t):
        if t == 0:
            return self.data["sum_assured"]
        elif t > self.data["term_y"] * 12:
            return 0
        elif self.data["shape"] == "level":
            return self.data["sum_assured"]
        elif self.data["shape"] == "decreasing":
            r = (1 + 0.07)**(1/12)-1
            S = self.data["sum_assured"]
            T = self.data["term_y"] * 12
            outstanding = S * ((1 + r)**T - (1 + r)**t)/((1 + r)**T - 1)
            return outstanding
        else:
            raise ValueError("Parameter 'shape' must be 'level' or 'decreasing'")
    
    def inflation_factor(self, t):
        """annual"""
        return (1 + self.basis["cost_inflation_pa"])**(t//12)

    def v(self, t):
        """present value of 1 discounted from time t to time 0"""
        if t == 0:
            return 1
        else:
            return self.v(t - 1) / (1 +self.basis["forward_rates"][t])

    def premiums(self, t):
        return self.premium_pp(t) * self.num_pols_if(t)
    
    def pv_premiums(self, t):
        return self.v(t) * self.premiums(t)
    
    def duration(self, t):
        """duration in force in years"""
        return t//12
    
    def claims(self, t):
        return self.claim_pp(t) * self.num_deaths(t)
    
    def pv_claims(self, t):
        return self.v(t) * self.claims(t)
      
    def expenses(self, t):
        if t == 0:
            return self.basis["initial_expense"]
        else:
            return self.num_pols_if(t) * self.basis["expense_pp"]/12 * self.inflation_factor(t)
      
    def pv_expenses(self, t):
        return self.v(t) * self.expenses(t)

    def num_pols_if(self, t):
        """number of policies in force"""
        if t == 0:
            return self.data["init_pols_if"]
        elif t > self.data["term_y"] * 12:
            return 0
        else:
            return self.num_pols_if(t-1) - self.num_exits(t-1) - self.num_deaths(t-1)
    
    def num_exits(self, t):
        """exits occurring at time t"""
        return self.num_pols_if(t) * (1-(1 - self.basis["lapse_rate_pa"])**(1/12))
      
    def num_deaths(self, t):
        """deaths occurring at time t"""
        return self.num_pols_if(t) * self.q_x_12(t)
    
    def age(self, t):
        return self.data["age_at_entry"] + t//12
    
    
    def q_x_12(self, t):
        return 1-(1- self.q_x_rated(t))**(1/12)
    
    def q_x(self, t):
        return self.basis["mort_table"][self.age(t), self.duration(t), self.data["smoker_status"]]
            
    def q_x_rated(self, t):
        return max(0, min(1 , self.q_x(t) * (1 + self.data["extra_mortality"]) ) )
        
    def commission(self, t):
        return 0

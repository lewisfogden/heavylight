from heavylight import Model

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

def test_heavylight():
    """Tests be improved upon later"""
    model = SimpleModel(do_run = True, proj_len = 10)
    assert model.pv_cashflow.sum() > 0

class UnitModel(Model):
    def t(self, t):
        return t

    def func_t(self, t):
        return 100 * t
    
    def func_a(self, a):
        return 100 * a

def test_UnitModel():
    proj_len = 10
    model = UnitModel(proj_len = proj_len)
    assert model.func_t.sum() == sum(100 * t for t in range(proj_len))
    assert model.func_a.values == []
    assert model.t.values == list(range(proj_len))
    model.func_a(5)
    model.func_a(10)
    model.func_a(7)
    assert model.func_a.values == [5, 10, 7]  # values are ordered by insertion.
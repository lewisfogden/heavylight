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
    assert model.func_a.values == [500, 1000, 700]  # values are ordered by insertion.
    assert [k[0] for k in model.t.keys] == model.t.values  # keys are stored as tuples, need to grab the first one.

class DataInput(Model):
    b = 4
    def t(self, t):
        return t
    def a_times_t(self, t):
        return self.a * self.b * t
    
def test_data_input():
    model = DataInput(proj_len=10, a=100)
    assert model.a_times_t(0) == 0
    assert model.a_times_t(10) == 10 * 100 * 4

class RecursiveModel(Model):
    """recursive model, including data inputs"""
    start_time = 14
    init_delta: int # = 2
    def t(self, t):
        return t

    def time_left(self, t):
        if t == 0:
            return self.start_time
        else:
            new_time = self.time_left(t - 1) - self.time_delta(t)
            return max(new_time, 0)
    
    def time_delta(self, t):
        return self.init_delta
        
def test_recursive():
    model = RecursiveModel(proj_len = 7, init_delta = 3)  # note overriding init_delta, this will warn
    assert model.time_left.values == [14, 11, 8, 5, 2, 0, 0]

class MultipleParamModel(Model):
    def t(self, t):
        return t
    
    def func_2_params(self, t, dur):
        """function with 2 parameters"""
        return t * dur * self.param_a(5)
    
    def sum_f2(self, t):
        return sum(self.func_2_params(t, dur) for dur in range(t + 1))
    
    def square_f2(self, t):
        return self.func_2_params(t, t)
    
    def param_a(self, a):
        return 3
    
def test_multi_param():
    model = MultipleParamModel(proj_len = 5)
    assert model.sum_f2(3) == 3 * 3 *(0 + 1 + 2 + 3)
    assert model.func_2_params(2, 5) == 2 * 5 * 3

class DataFrameTest(Model):
    def t(self, t):
        return t
    
    def ones(self, t):
        return 1
    
    def excluded(self, blah):
        """not included in the dataframe as t is not a parameter"""
        return 99
    
    def ninety_nines(self, t):
        return self.excluded(t)

def test_dataframe_extracts():
    model = DataFrameTest(proj_len=5)
    df = model.df
    assert len(df) == 5
    assert df['ones'].sum() == 5 
    assert df['ninety_nines'].sum() == 5 * 99
    assert sorted(df.columns) == sorted(['t', 'ones', 'ninety_nines'])

    # check dataframe for one function
    df_ones = model.ones.df
    assert df_ones.shape == (5, 2)  # 5 rows, 2 columns (t and ones)
    assert sorted(df_ones.columns) == sorted(['t', 'ones'])

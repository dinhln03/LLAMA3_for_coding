import numpy as np
import pandas as pd
import matlibplot.pyplot as plt


'''
Simulating Solow-Swan model, which attempts to model the long-run economic growth
by looking at capital accumulation (K), population growth (L) and technological 
progress, which results in increase in productivity. It models the total production
of the economy using the constant-returns-to-scale Cobb-Douglas production function

    Y(t) = K(t)^{alpha} * (A(t)L(t))^{1-alpha}, where
    
    Y(t): a single good output at time t
    K(t): the amount of capital at time t
    L(t): population at time t
    A(t): total factor productivity at time t
    alpha: output elasticity of capital

with a law of motion:
    I(t) = sY(t)
    C(t) = (1-s)Y(t)
    K(t+1) = (1-delta)K(t) + I(t)
    L(t+1) = (1+n)L(t)

we can derive the law of motion for k(t) capital per capita:
    k(t+1) = K(t+1)/N(t+1)
           = ((1-delta)K(t) + I(t))/ (1+n)N(t)
           = (1-delta)/(1+n) * k(t) + s/(1+n) A*K_t^alpha

as well as per capita output:
    y(t) = Y(t)/N(t)
         = Ak_t^alpha

where, I(t): total investment at time t
       C(t): total consumption at time t
       K(t): total capital at time t
       L(t): total population at time t
          s: the saving rate
      delta: rate of capital depreciation
          n: rate of population growth

This simulation allows user to take controls of those parameters and plot the simulated
total output growth. The program also enables user to query data from the Federal Reserve
Economic Data
'''
class solow:
    '''
    A: total factor productivity
    k0: the initial amount of capital
    delta: rate of depreciation of cpiatal
    s: the saving rate
    n: the population growth rate
    alpha: output elasticity of capital
    starting_year: 
    '''
    def __init__(self, A=2.87, k0=3.5, delta = 0.08, s = 0.1, n = 0.015, alpha = 0.36, t0 = 1956, tmax = 2060):
        self._A = A
        self._k0 = k0
        self._k = k0
        self._delta = delta
        self._s = s
        self._n = n
        self._alpha = alpha
        self._t0 = t0
        self._tmax = tmax
        self._t = range(t0, tmax + 1)
        self._y = np.zeros(len(self._t))
        self._y[0] = self._A * (self._k0 ** self._alpha)
        self._time_passed = 0

    '''
    this method returns all the variables in this model, which includes A, k0,
    delta, s, n, alpha, t0, tax, Y, and t as a dictionary
    '''
    def get_variables(self):
        return {
            'A' : self._A, 
            'k0': self._k0, 
            'delta': self._delta,
            's' : self._s,
            'n' : self._n,
            'alpha': self._alpha,
            't0' : self._t0,
            'tmax': self._tmax,
            'y' : self._y,
            't' : self._t }
    
    '''
    this method takes a list or dictionary as input and set the variables based on
    the user's input. If the user inputs a list, it will treats the entries of list
    as the values of A, k0, delta, s, n, alpha, t0, tmax, Y, t the user wants to 
    change into. If the user inputs a dictionary, the fields will be set according
    to the keys.

    Example:
    set_variables({A: 2.87, k0: 3.5, delta:0.08, s:0.1, n:0.015, alpha:0.36, t0:1956, tmax:2060})
    set_variables(2.87,3.5,0.08,0.1,0.015,0.36,1956,2060)
    both achieve the same output
    '''
    def set_variables(self, vars):
        if (type(vars) != type([]) or type(vars) != type({})):
            raise ValueError('arguments must be either a dictionary or a list')
        if (type(vars) == type([])):
            if (len(vars) != 8):
                raise ValueError('You must enter the following arguments: A, k0, delta, s, n, alpha, t0, tmax')
            else:
                self.setA(vars[0])
                self.setK0(vars[1])
                self.setDelta(vars[2])
                self.setS(vars[3])
                self.setN(vars[4])
                self.setAlpha(vars[5])
                self.setTRange(vars[6], vars[7])
        if (type(vars) == type({})):
            try:
                self.setA(vars['A'])
                self.setK0(vars['k0'])
                self.setDelta(vars['delta'])
                self.setS(vars['s'])
                self.setN(vars['n'])
                self.setAlpha(vars['alpha'])
                self.setTRange(vars['t0'], vars['tmax'])
            except KeyError:
                raise ValueError("Your dictionary must have the keys A, k0, delta, s, n, alpha, t0, and tmax")
    '''
    setter for the field A (total factor productivity)
    '''
    def setA(self, A):
        if (A < 0):
            raise ValueError("A must be positive")
        self._A = A
    
    '''
    setter for the field k0 (the initial amount of capital)
    '''
    def setK0(self,k0):
        if(k0 < 0):
            raise ValueError("k0 must be positive")
    
    '''
    setter for Delta (rate of depreciation of cpiatal)
    '''
    def setDelta(self, delta):
        if (delta > 1 or delta < 0):
            raise ValueError("depreciation rate must be in between 0 and 1")
        self._delta = delta
    
    '''
    setter for S (saving rate)
    '''
    def setS(self, s):
        if (s > 1 or s < 0):
            raise ValueError("saving rate must be in between 0 and 1")
        self.S = S
    
    '''
    setter for N (population growth rate)
    '''
    def setN(self,n):
        self._n = n
    
    '''
    setter for alpha (output elasticity of capital)
    '''
    def setAlpha(self, alpha):
        if (alpha < 0 or alpha > 1):
            raise ValueError("alpha must be in between 0 and 1")
        self._alpha = alpha
    
    '''
    setter for the time range

    Example:
    setTRange(1956, 2060): set the time range starting from 1956 to 2060
    '''
    def setTRange(self, start, end):
        if (end < start):
            raise ValueError("tmax must be greater than t0")
        self._t0 = start
        self._tmax = end
        self._t = range(start, end+1)

    '''
    Start the simulation, and return the predicted value of Y 
    from the start period to the end period

    TO BE IMPLEMENTED
    '''
    def simulate(self):
        for t in self._t:
            self._update()
        return [self._y, self._t]

    '''
    Plot the prediction using matlibplot. x-axis would be year, y-axis would 
    the predicted GDP

    TO BE IMPLEMENTED
    '''
    def plot(self):
        pass

    '''
    store the output as a pandas dataframe
    '''
    def to_df(self):
        return pd.DataFrame({'year' : self._t, 'gdp_per_capita' : self._y})

    '''
    export the output as a csv file to the user-provided location

    TO BE IMPLEMENTED
    '''
    def to_csv(self, dir):
        pass
    
    '''
    lunch the GUI, that enables more user-friendly interaction with the software
    
    TO BE IMPLEMENTED
    '''
    def gui(self):
        pass

    '''
    update all the fields according to the law of motion

    TO BE IMPLEMENTED
    '''
    def _update(self):
        #update k
        self._k = (1-self._delta)/(1+self._n) * self._k + (self._s)/(1+n) * self._A * (self._k ** self._alpha)
        # update t
        self._time_passed += 1
        #update y
        self._y[self._time_passed] = self._A * (self._k ** self._alpha)

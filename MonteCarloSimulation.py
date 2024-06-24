import numpy as np
import matplotlib.pyplot as plt
from ad import adnumber
from ad.admath import *

def sign_func(x, y):
    if x >= y:
        return 1
    else:
        return 0

class MonteCarloPricer():
    def __init__(self, random_seed, S_0, K, T, r, sigma, num_sims, option_type):
        self.S_0 = S_0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.num_sims = num_sims
        self.option_type = option_type

        self.num_steps = int(T * 252)
        self.dt = self.T / self.num_steps

        self.random_seed = random_seed


    """
    calculate option price at each time step t by averaging and discounting to present value.
    """
    def calc_option_price(self):
        np.random.seed(self.random_seed)
        
        underlying_prices_array = np.zeros((self.num_steps + 1, self.num_sims))
        payoff_array = np.zeros((self.num_steps + 1, self.num_sims))
        
        z_array = np.random.normal(0, 1, size=(self.num_steps, self.num_sims))

        S_t = self.S_0 * np.exp(np.cumsum((self.r - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * z_array, axis=0))

        underlying_prices_array[1:, :] = S_t
        
        if self.option_type == 'call':
            payoff_array[1:, :] = np.maximum(S_t - self.K, 0)
        elif self.option_type == 'put':
            payoff_array[1:, :] = np.maximum(self.K - S_t, 0)
        else:
            raise ValueError(f"{self.option_type} is not defined.")
        
        average_option_prices = np.mean(payoff_array, axis=1)

        discounted_option_prices = average_option_prices * np.exp(-self.r * np.arange(self.num_steps + 1) * self.dt)
        option_price = discounted_option_prices[-1]
        
        return underlying_prices_array, discounted_option_prices, option_price
    

    """
    Estimate delta, vega, rho using automatic differentiation method.
    """
    def calc_option_greeks(self):
        S_0 = adnumber(self.S_0)
        sigma = adnumber(self.sigma)
        r = adnumber(self.r)

        sum_payoff = 0
        for sim in range(self.num_sims):
            S_t = S_0
            z_array = np.random.normal(0, 1, size=(self.num_steps, 1))
            
            for step in range(1, self.num_steps+1):
                S_t = S_t * exp((r - 0.5 * sigma ** 2) * self.dt + sigma * sqrt(self.dt) * float(z_array[step-1]))
            
            if self.option_type == 'call':
                payoff = max(S_t - self.K, 0)
            elif self.option_type == 'put':
                payoff = max(self.K - S_t, 0)
            else:
                raise ValueError(f"{self.option_type} is not defined.")
            sum_payoff = sum_payoff + payoff
        
        option_price = (sum_payoff / self.num_sims) * exp(-r * self.T)
        
        delta = option_price.d(S_0)
        vega = option_price.d(sigma) / 100
        rho = option_price.d(r) / 100

        return option_price, delta, vega, rho
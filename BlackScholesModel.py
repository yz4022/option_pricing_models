import numpy as np
from scipy.stats import norm

class BlackScholesPricer():
    def __init__(self, S_0, K, T, r, sigma, num_sims, option_type):
        self.S_0 = S_0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.num_sims = num_sims
        self.option_type = option_type

        self.num_steps = int(T * 252)
        self.dt = self.T / self.num_steps

    
    def calc_option_price(self):
        d1 = (np.log(self.S_0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == 'call':
            option_price = self.S_0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.option_type == 'put':
            option_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S_0 * norm.cdf(-d1)
        else:
            raise ValueError(f"{self.option_type} is not defined.")
        
        return option_price
    

    def calc_greeks(self):
        d1 = (np.log(self.S_0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == 'call':
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (self.S_0 * self.sigma * np.sqrt(self.T))
            theta = (-self.S_0 * self.sigma * norm.pdf(d1) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
            vega = self.S_0 * np.sqrt(self.T) * norm.pdf(d1)
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)

        elif self.option_type == 'put':
            delta = norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (self.S_0 * self.sigma * np.sqrt(self.T))
            theta = (-self.S_0 * self.sigma * norm.pdf(d1) / (2 * np.sqrt(self.T))+ self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
            vega = self.S_0 * np.sqrt(self.T) * norm.pdf(d1)
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)

        else:
            raise ValueError(f"{self.option_type} is not defined.")
        
        return delta, gamma, theta, vega, rho
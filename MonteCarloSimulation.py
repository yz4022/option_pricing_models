import numpy as np
import matplotlib.pyplot as plt

class MonteCarloPricer():
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

    """
    calculate option price at each time step t by averaging and discounting to present value.
    """
    def calc_option_price(self):
        np.random.seed(42)
        option_prices_array = np.zeros((self.num_steps + 1, self.num_sims))

        for step in range(self.num_steps + 1):
            for sim in range(self.num_sims):
                S_t = self.S_0
                for item in range(step):
                    z = np.random.normal(0, 1)
                    S_t *= np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * z)
                if self.option_type == 'call':
                    option_price = max(S_t - self.K, 0)
                else:
                    option_price = max(self.K - S_t, 0)
                option_prices_array[step, sim] = option_price

        average_option_prices = np.mean(option_prices_array, axis=1)
        
        # Discount all the option prices to present time
        discounted_option_prices = average_option_prices * np.exp(-self.r * np.arange(self.num_steps + 1) * self.dt)
        
        return option_prices_array, discounted_option_prices
    

    """
    Plot option price at each time step t.
    """
    def plot_option_price(self, option_prices):
        plt.plot(np.arange(self.num_steps, option_prices, label = 'Option Price'))
        plt.xlabel('Time Step')
        plt.ylabel('Option Price')
        plt.title('Option Price at Each Time Step')
        plt.legend()
        plt.grid(True)
        plt.show()

    """
    Estimate delta for the option price at time step t by recalculating option price at time step t with underlying
    asset prices S +- eplison and apply the formula (option_price_up - option_price_down) / (2 * epsilon * self.S_0).
    """
    def calc_option_delta(self, epsilon):
        deltas = np.zeros(self.num_steps + 1)
        
        for step in range(self.num_steps + 1):

            S_up = self.S_0 * (1 + epsilon)
            option_prices_array_up, _ = self.calc_option_price(S_up)
            option_price_up = option_prices_array_up[step].mean()

            S_down = self.S_0 * (1 - epsilon)
            option_prices_array_down, _ = self.calc_option_price(S_down)
            option_price_down = option_prices_array_down[step].mean()

            delta = (option_price_up - option_price_down) / (2 * epsilon * self.S_0)
            deltas[step] = delta

        return deltas



        
    
    
                




        


    


    
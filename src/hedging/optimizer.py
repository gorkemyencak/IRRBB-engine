import numpy as np

from scipy.optimize import minimize

from src.hedging.swaps import Hedge_Tenors, hedge_dv01_vector

class HedgeOptimizer:
    """ Finds swap notionals that neutralize bank DV01 """
    def __init__(
            self, 
            bank_kr_dv01_vector
    ):
        
        # total bank DV01 exposure
        self.bank_kr_dv01_vector = np.array(bank_kr_dv01_vector)

        # DV01 of swaps per 1 million notional
        self.swap_dv01 = hedge_dv01_vector()

    
    def portfolio_dv01(
            self,
            notionals
    ):
        """ DV01 of bank after adding hedge swaps """
        hedge_dv01 = notionals * self.swap_dv01
        total_dv01 = self.bank_kr_dv01_vector + hedge_dv01         # total risk = bank risk + hedge risk

        return total_dv01
    
    # objective function to minimize
    def objective(
            self,
            notionals
    ):
        """ 
        Objective target is to set DV01 ~= 0
        Optimizer minimizes squared errors        
        """
        residual_curve = self.portfolio_dv01(notionals = notionals)

        return np.sum(residual_curve ** 2)
    

    def optimize(self):
        """ 
        Runs hedge optimization 
        It returns optimal swap notionals vector for a given bound per each vector element
        """
        # start with no hedges
        x0 = np.zeros(len(Hedge_Tenors))

        # Limit trade sizes per each tenor
        hedge_bounds = [(-5000000, +5000000) for _ in Hedge_Tenors]

        result = minimize(
            fun = self.objective, 
            x0 = x0,
            bounds = hedge_bounds,
            method = 'L-BFGS-B'
        )

        return result.x
    

    def hedge_report(
            self, 
            notionals
    ):

        print("\n--- Recommended Hedge Trades ---\n")
        for tenor, notional in zip(Hedge_Tenors, notionals):

            side = "Pay Fixed" if notional > 0 else "Receive Fixed"

            print(f"{side:12} {abs(notional)/1e6:8.4f}m {tenor}Y swap")

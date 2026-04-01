import numpy as np
import pandas as pd

class NMDModel:
    """ 
    Behavioral model for Non-Maturity Deposits
    It produces behavioral cashflows and repricing profile 
    """
    def __init__(
            self,
            balance,            # total outstanding deposit
            product_type,       # 'retail_current', 'retail_savings', 'corporate'
            core_ratio,         # % considered stable
            avg_life_years,     # behavioral maturity of core portion
            beta                # deposit beta (repricing sensitivity) -> ranges between 0 and 1
    ):
        
        self.balance = balance
        self.product_type = product_type
        self.core_ratio = core_ratio
        self.avg_life = avg_life_years
        self.beta = beta                

    # split into core vs non-core deposits
    def split_balances(self):
        core = self.balance * self.core_ratio
        non_core = self.balance * (1 - self.core_ratio)

        return core, non_core
    
    # behavioral maturity via decay of core deposits
    def generate_core_cashflows(
            self,
            horizon_years = 10,
            steps_per_year = 12
    ):
        
        core_balance, _ = self.split_balances()

        n_steps = horizon_years * steps_per_year
        dt = 1 / steps_per_year
        times = np.arange(1, n_steps + 1) * dt

        remaining = core_balance * np.exp(-times / self.avg_life)       # Balance(t) = B_0 * e^(-t/avg_life)
        amortization = np.diff(np.insert(remaining, 0, core_balance))

        cashflows = -amortization       # liability runoff

        return times, cashflows
    
    # behavioral maturity via decay of non-core deposits
    def generate_noncore_cashflows(
            self
    ):
        
        _, noncore_balance = self.split_balances()

        times = np.array([1/12])        # non-core deposits are assumed to have 1 month maturity
        cashflows = np.array([noncore_balance])

        return times, cashflows
    
    # deposit repricing for NII calculation -> when the market rates move, deposit rates move only partially
    def deposit_rate_shock(
            self,
            rate_shock
    ):
        
        return self.beta * rate_shock
    
    # combine into total behavioral cashflows
    def total_cashflows(self):

        t_core, cf_core = self.generate_core_cashflows()
        t_noncore, cf_noncore = self.generate_noncore_cashflows()

        times = np.concatenate([t_core, t_noncore])
        cashflows = np.concatenate([cf_core, cf_noncore])

        # sort chronogically to avoid duplicates for core and non-core time steps
        idx = np.argsort(times)
        times = times[idx]
        cashflows = cashflows[idx]

        return times, cashflows
    
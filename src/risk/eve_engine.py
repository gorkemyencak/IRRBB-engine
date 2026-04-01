import pandas as pd
import numpy as np

class EVEEngine:
    """ Computes Economiv Value of Equity (EVE) using DiscountCurve class -> PV(Assets) - PV(Liabilities) """
    def __init__(
            self,
            discount_curve
    ):
        
        self.discount_curve = discount_curve
    

    def compute_eve(
            self,
            portfolio_cf: pd.DataFrame
    ):
        
        df = portfolio_cf.copy()
        
        df['signed_cf'] = np.where(
            df['instrument_type'] == 'asset',
            df['total_cashflow'],               # inflow
            -df['total_cashflow']               # outflow
        )

        cashflows = df['signed_cf'].values
        times = df['ttm'].values

        eve = self.discount_curve.present_value(
            cashflows = cashflows,
            times = times
        )
        
        return eve

    

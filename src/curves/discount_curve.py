import numpy as np
import pandas as pd

class DiscountCurve:
    """ Builds discount factors, forward rates, present value and BPV from zero-rates """
    def __init__(
            self,
            curve_df
    ):
        self.curve = curve_df.copy()
        self._build_discount_factors()
        self._build_forward_rates()
    
    # Discount factors
    def _build_discount_factors(self):

        self.curve['discount_factor'] = np.exp(
            -self.curve['zero_rate'] * self.curve['tenor_years']
        )
    
    # Forward rates
    def _build_forward_rates(self):

        t = self.curve['tenor_years'].values
        z = self.curve['zero_rate'].values

        # Numerical derivative
        dz_dt = np.gradient(z*t, t)

        self.curve['forward_rate'] = dz_dt
    
    # Discount factor at a given maturity
    def get_discount_factor(self, maturity):

        return np.interp(
            maturity,
            self.curve['tenor_years'],
            self.curve['discount_factor']
        )
    
    # Continuously compounded zero rate from discount factor
    def get_zero_rate(
            self,
            t: float
    ):
        
        if t == 0:
            return 0.0
        
        df = self.get_discount_factor(t)

        return -np.log(df) / t
    
    # Present value of cashflows
    def present_value(
            self,
            cashflows,
            times
    ):
        """
        Parameters
        cashflows: cashflow array
        times: year array
        """
        dfs = np.array([self.get_discount_factor(t) for t in times])

        return np.sum(np.array(cashflows) * dfs)
    
    # PV01/DV01
    def pv01(
            self,
            cashflows,
            times, 
            shock_bps = 1
    ):
        
        shocked_curve = self.curve.copy()
        shocked_curve['zero_rate'] += shock_bps / 10000

        shocked_dc = DiscountCurve(shocked_curve)

        pv_base = self.present_value(
            cashflows = cashflows,
            times = times
        )

        pv_shocked = shocked_dc.present_value(
            cashflows = cashflows,
            times = times
        )

        return pv_shocked - pv_base


import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

class CurveInterpolator:
    """ Interpolating interest rate term structures, tenors expressed in years """
    def __init__(
            self,
            tenors,
            rates
    ):
        """ 
        Parameters
        tenors: maturity array in years
        rates: zero rate array in decimal
        """
        self.tenors = np.array(tenors, dtype = float)
        self.rates = np.array(rates, dtype = float)

        # sorted arrays
        idx = np.argsort(self.tenors)
        self.tenors = self.tenors[idx]
        self.rates = self.rates[idx]

    # 1. Linear interpolation on zero rates
    def linear_rate_interp(
            self,
            target_tenors
    ):
        
        f = interp1d(
            self.tenors, 
            self.rates,
            fill_value = "extrapolate" # pyright: ignore
        )

        return f(target_tenors) 
        
    # 2. Log-linear interpolation on discount factors
    def log_discount_interp(
            self,
            target_tenors
    ):
        
        discount_factors = np.exp(-self.rates * self.tenors)

        f = interp1d(
            self.tenors,
            discount_factors,
            fill_value = 'extrapolate' # pyright: ignore
        )

        log_discountfactor_interp = f(target_tenors)
        discountfactor_interp = np.exp(log_discountfactor_interp)

        # converting back to zero rates
        zero_rates = -np.log(discountfactor_interp) / target_tenors

        return zero_rates
    
    def build_interp_curve(
            self,
            max_years = 30,
            method = 'log_df'
    ):
        """ Returns interpolated curve up to max_years maturity """
        days = np.arange(1, int(max_years * 365) + 1)
        target_tenors = days / 365
        rates = np.empty(len(target_tenors))

        if method == 'linear':
            rates = self.linear_rate_interp(
                target_tenors = target_tenors
            )
        elif method == 'log_df':
            rates = self.log_discount_interp(
                target_tenors = target_tenors
            )
        else:
            raise ValueError(f"Unknown method {method}, must be either 'linear' or 'log_df'")
        
        curve_df = pd.DataFrame({
            'tenor_years': target_tenors,
            'zero_rate': rates
        })

        return curve_df
            
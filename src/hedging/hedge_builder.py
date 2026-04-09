import numpy as np

from src.instruments.interest_rate_swap import IRSwap

from src.hedging.swaps import Hedge_Tenors

def _interpolate_zero_rate(
        discount_curve,
        target_tenor
):
    """ 
    Linear interpolation of zero rate for missing tenors
    Otherwise, it returns the exact match for zero rate
    """
    curve = discount_curve.curve.sort_values('tenor_years')

    tenors = curve['tenor_years'].values
    rates = curve['zero_rate'].values
    
    if target_tenor in tenors:
        # exact match
        return rates[tenors.tolist().index(target_tenor)]
    
    else:
        # linear interpolation
        interpolated_rate = np.interp(
            target_tenor,
            tenors,
            rates
        )

        return float(interpolated_rate)


def build_hedge_swaps(
        notionals,
        discount_curve,
        payments_per_year = 4
):
    """ 
    Converting optimized notionals into actual IRSwap trades 
    using interpolated swap rates
    """
    hedge_swaps = []
    for tenor, notional in zip(Hedge_Tenors, notionals):
        
        # interpolated swap rate if missing, o/w exact match
        swap_rate = _interpolate_zero_rate(
            discount_curve = discount_curve,
            target_tenor = tenor
        )

        swap = IRSwap(
            notional = notional,
            fixed_rate = swap_rate,
            maturity_years = tenor,
            payments_per_year = payments_per_year,
            pay_fixed = True if notional > 0 else False
        )

        hedge_swaps.append(swap)
    
    return hedge_swaps

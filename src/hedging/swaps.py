import numpy as np

# tenor we allow Treasure to trade
Hedge_Tenors = np.array([1, 2, 3, 5, 7, 10]) # in years

def swap_dv01(
        notional: float,
        maturity: float
) -> float:
    
    """ 
    Approximate DV01 of an IRS 
    It returns DV01 in currency units per 1 bp move
    """
    # swap duration ~= 0.8 * maturity
    duration = 0.8 * maturity
    dv01 = notional * duration / 10000

    return dv01


def hedge_dv01_vector():
    """ 
    DV01 of each hedge swap tenor for 1 million notional 
    It returns numpy array of DV01s aligned with Hedge_Tenors
    """
    dv01s = []

    for tenor in Hedge_Tenors:
        dv01_1m = swap_dv01(
            notional = 1000000,
            maturity = tenor
        )

        dv01s.append(dv01_1m)
    
    return np.array(dv01s)

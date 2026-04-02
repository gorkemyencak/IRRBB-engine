import numpy as np
import pandas as pd

class FloatingRateLoan:
    """ Floating rate loan using forward curve repricing """
    def __init__(
            self,
            notional,
            spread,
            maturity_years,
            payments_per_year = 4
    ):
        
        self.notional = notional
        self.spread = spread
        self.maturity = maturity_years
        self.freq = payments_per_year

    
    def generate_cashflows(
            self,
            discount_curve
    ):
        """ Uses forward rates from DiscountCurve """
        n_periods = int(self.maturity * self.freq)
        dt = 1 / self.freq

        times = np.arange(1, n_periods + 1) * dt
        remaining_notional = self.notional

        cashflows = []
        for t in times:

            forward_rate = np.interp(
                t,
                discount_curve.curve['tenor_years'],
                discount_curve.curve['forward_rate']
            )

            floating_rate = forward_rate + self.spread
            interest = remaining_notional * floating_rate * dt

            # bullet repayment at maturity
            principal = self.notional if t == times[-1] else 0

            cashflows.append(interest + principal)
        
        return times, np.array(cashflows)
    

    def present_value(
            self,
            discount_curve
    ):
        
        times, cfs = self.generate_cashflows(
            discount_curve = discount_curve
        )

        return discount_curve.present_value(
            cashflows = cfs,
            times = times
        )

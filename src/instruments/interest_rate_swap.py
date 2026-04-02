import numpy as np

class IRSwap:
    """ pay-fixed, receive-floating IR Swap -> positive PV when rates rise """
    def __init__(
            self,
            notional,
            fixed_rate,
            maturity_years,
            payments_per_year = 4
    ):
        
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.maturity = maturity_years
        self.freq = payments_per_year

    # fixed leg cashflows (paid by bank)
    def _fixed_leg_cashflows(self):

        n = int(self.maturity * self.freq)
        dt = 1 / self.freq
        times = np.arange(1, n + 1) * dt
        
        fixed_coupon = self.notional * self.fixed_rate * dt
        cashflows = np.full(n, fixed_coupon)

        return times, cashflows

    # floating leg using forward curve
    def _floating_leg_cashflows(
            self,
            discount_curve
    ):
        
        n = int(self.maturity * self.freq)
        dt = 1 / self.freq
        times = np.arange(1, n + 1) * dt

        cashflows = []

        for t in times:

            forward = np.interp(
                t,
                discount_curve.curve['tenor_years'],
                discount_curve.curve['forward_rate'] # zero_rate
            )

            float_coupon = self.notional * forward * dt
            cashflows.append(float_coupon)
        
        return times, np.array(cashflows)
    

    def present_value(
            self,
            discount_curve
    ):
        """ PV = PV(float leg) - PV(fixed leg) """
        t_fix, cf_fix = self._fixed_leg_cashflows()
        t_float, cf_float = self._floating_leg_cashflows(discount_curve = discount_curve)

        pv_fixed = discount_curve.present_value(
            cashflows = cf_fix,
            times = t_fix
        )

        pv_float = discount_curve.present_value(
            cashflows = cf_float,
            times = t_float
        )

        return pv_float - pv_fixed


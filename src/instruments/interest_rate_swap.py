import numpy as np

class IRSwap:
    """ pay-fixed, receive-floating IR Swap -> positive PV when rates rise """
    def __init__(
            self,
            notional: float,
            fixed_rate: float,
            maturity_years: float,
            payments_per_year: int = 4,
            pay_fixed: bool = True      # trade direction
    ):
        
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.maturity = maturity_years
        self.freq = payments_per_year
        self.pay_fixed = pay_fixed

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
    
    
    def compute_1y_nii(
            self,
            discount_curve,
            rate_shock
    ):
        """ Approximate 1-year NII of swap under shocked rates """
        accrual = 1 / self.freq
        n_payments = self.freq

        forward_rate = discount_curve.get_zero_rate(t = 1) + rate_shock

        fixed_leg = self.fixed_rate * self.notional * accrual * n_payments
        float_leg = forward_rate * self.notional * accrual * n_payments

        if self.pay_fixed:
            return float_leg - fixed_leg
        else:
            return fixed_leg - float_leg
    

    def present_value(
            self,
            discount_curve
    ):
        """ 
        Swap PV depends on trade direction

        Pay-fixed swap:
            receive floating - pay fixed
        
        Receive-fixed swap:
            receive fixed - pay floating
        """
        # build both fixed and floating legs
        t_fix, cf_fix = self._fixed_leg_cashflows()
        t_float, cf_float = self._floating_leg_cashflows(discount_curve = discount_curve)

        # discount both fixed and floating legs
        pv_fixed = discount_curve.present_value(
            cashflows = cf_fix,
            times = t_fix
        )
        pv_float = discount_curve.present_value(
            cashflows = cf_float,
            times = t_float
        )

        # trade direction
        if self.pay_fixed:
            # pay fixed, receive floating
            pv = pv_float - pv_fixed
        else:
            # receive fixed, pay floating
            pv = pv_fixed - pv_float

        return pv


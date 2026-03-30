import numpy as np
import pandas as pd

class IRRBBShock:
    """ Basel IRRBB standardized IR shocks """
    def __init__(
            self,
            base_curve: pd.DataFrame
    ):
        
        self.base_curve = base_curve.copy()

    # Shock shape helper
    def _short_long_scaling(self, tenors):
        """
        Bsael maturity-dependent scaling
        Short-end -> stronger shocks
        Long-end -> weaker shocks

        It produces a scaling factor ranges between 1.0 and 0.5; so that Basel applies
        weighted shocks across all maturities, but with stronger weight at specific parts 
        of the curve.
        """
        short_end = 2.0
        long_end = 20.0

        scale = np.where(
            tenors <= short_end,
            1.0,
            np.where(
                tenors >= long_end,
                0.5,
                1.0 - 0.5 * (tenors - short_end) / (long_end - short_end)
            )
        )

        return scale
    
    def _apply_shock(self, shock_bps):

        shocked_curve = self.base_curve.copy()
        scale = self._short_long_scaling(
            tenors = shocked_curve['tenor'].values
        )

        shock_decimal = shock_bps / 10000
        shocked_curve['rate'] = shocked_curve['rate'] + scale * shock_decimal

        return shocked_curve

    # Basel Scenarios
    def parallel_up(self):

        return self._apply_shock(shock_bps = 200)

    def parallel_down(self):

        return self._apply_shock(shock_bps = -200)

    def steepener(self):

        shocked_curve = self.base_curve.copy()
        scale = self._short_long_scaling(shocked_curve['tenor'].values)

        short_shock = -100 / 10000
        long_shock = +100 / 10000

        shocked_curve['rate'] += scale * short_shock + (1 - scale) * long_shock

        return shocked_curve

    def flattener(self):

        shocked_curve = self.base_curve.copy()
        scale = self._short_long_scaling(shocked_curve['tenor'].values)

        short_shock = +100 / 10000
        long_shock = -100 / 10000

        shocked_curve['rate'] += scale * short_shock + (1 - scale) * long_shock

        return shocked_curve

    def short_rate_up(self):

        shocked_curve = self.base_curve.copy()
        scale = self._short_long_scaling(shocked_curve['tenor'].values)

        short_shock = +250 / 10000

        shocked_curve['rate'] += scale * short_shock

        return shocked_curve

    def short_rate_down(self):

        shocked_curve = self.base_curve.copy()
        scale = self._short_long_scaling(shocked_curve['tenor'].values)

        short_shock = -250 / 10000

        shocked_curve['rate'] += scale * short_shock

        return shocked_curve

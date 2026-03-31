import pandas as pd
import numpy as np

from src.instruments.base_instrument import BaseInstrument

from src.cashflows.schedule import generate_payment_schedule

class FixedRateLoan(BaseInstrument):
    """ Fixed rate loan container """
    def __init__(
            self,
            notional: float,
            start_date: str,
            maturity_date: str,
            fixed_rate: float,
            payment_frequency: str = 'M'
    ):
        
        super().__init__(
            notional = notional,
            start_date = start_date,
            maturity_date = maturity_date
        )

        self.fixed_rate = fixed_rate
        self.payment_frequency = payment_frequency

        freq_to_periods = {
            'M': 12,
            'Q': 4,
            'S': 2,
            'A': 1
        }

        self.periods_per_year = freq_to_periods[self.payment_frequency]
    
    # Annuity payment calculator for a fully amortizing and equal payment loan
    def _annuity_payment(
            self,
            n_periods
    ):
        
        r = self.fixed_rate / self.periods_per_year

        payment = (
            self.notional * r /
            (1 - (1 + r) ** (-n_periods))
        )

        return payment
    
    # Cashflow generator
    def generate_cashflows(self) -> pd.DataFrame:
        
        dates = generate_payment_schedule(
            start_date = self.start_date,
            maturity_date = self.maturity_date,
            frequency = self.payment_frequency
        )

        n_periods = len(dates)
        payment = self._annuity_payment(
            n_periods = n_periods
        )

        outstanding = self.notional
        rows = []

        for date in dates:
            interest = outstanding * self.fixed_rate / self.periods_per_year
            principal = payment - interest
            outstanding -= principal

            rows.append({
                'date': date,
                'principal': principal,
                'interest': interest,
                'total_cashflow': principal + interest,
                'outstanding_balance': outstanding
            })
        
        df = pd.DataFrame(rows)

        return df





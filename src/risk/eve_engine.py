import pandas as pd
import numpy as np

class EVEEngine:
    """ Computes Economiv Value of Equity (EVE) using DiscountCurve class -> PV(Assets) - PV(Liabilities) """
    def __init__(
            self,
            discount_curve,
            valuation_date
    ):
        
        self.discount_curve = discount_curve
        self.valuation_date = valuation_date
    

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
    

    def compute_eve_from_instruments(
            self,
            fixed_loans,
            floating_loans,
            nmd_model
    ):
        """ Full Bank EVE """
        asset_pv = 0.0

        # fixed loans
        for loan in fixed_loans:
            asset_pv += loan.present_value(
                discount_curve = self.discount_curve,
                valuation_date = self.valuation_date
            )
        
        # floating loans
        for loan in floating_loans:
            asset_pv += loan.present_value(self.discount_curve)
        
        # NMD liabilities
        times, cashflows = nmd_model.total_cashflows()
        liability_pv = self.discount_curve.present_value(
            cashflows = cashflows,
            times = times
        )

        eve = asset_pv - liability_pv
        
        return eve

    

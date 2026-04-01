import pandas as pd

from datetime import timedelta

class NIIEngine:
    """ Computes 12-month Net Interest Income under a given curve """
    def __init__(
            self,
            valuation_date
    ):
        
        self.valuation_date = valuation_date
        self.horizon_date = self.valuation_date + timedelta(days = 365)

    # filter next 12M cashflows
    def _filter_1y_cashflows(
            self,
            portfolio_cf: pd.DataFrame
    ):
        
        df = portfolio_cf.copy()
        df['date'] = pd.to_datetime(df['date'])
        filtered_df = df.loc[
            (df['date'] > self.valuation_date) &
            (df['date'] <= self.horizon_date)
        ]

        return filtered_df
    
    # attach repricing rate from curve
    def _attach_rates(
            self,
            cashflows: pd.DataFrame,
            curve: pd.DataFrame
    ):
        """ Repricing rule: use 1Y rate as proxy for repricing rate """
        rate_1Y = curve.loc[
            curve['tenor'] == 1,
            'rate'
        ].values[0]

        cashflows = cashflows.copy()
        cashflows['repricing_rate'] = rate_1Y

        return cashflows
    

    def compute_nii(
            self,
            portfolio_cf: pd.DataFrame,
            curve: pd.DataFrame
    ):
        
        # step 1: keep only next 12M cashflows
        cf_1y = self._filter_1y_cashflows(
            portfolio_cf = portfolio_cf
        )

        # step 2: attach repricing rate
        cf_1y = self._attach_rates(
            cashflows = cf_1y,
            curve = curve
        )

        # step 3: recompute interest under repricing rate
        cf_1y['repriced_interest'] = (
            cf_1y['outstanding_balance'] * cf_1y['repricing_rate']
        )

        # step 4: split asset vs liability
        asset_interest = cf_1y.loc[
            cf_1y['instrument_type'] == 'asset',
            'repriced_interest'
        ].sum()

        liability_interest = cf_1y.loc[
            cf_1y['instrument_type'] == 'liability',
            'repriced_interest'
        ].sum()

        nii = asset_interest - liability_interest

        return nii


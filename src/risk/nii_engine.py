import pandas as pd

from datetime import timedelta

class NIIEngine:
    """ Computes 12-month Net Interest Income under a given curve """
    def __init__(
            self,
            valuation_date,
            payments_per_year = 4
    ):
        
        self.valuation_date = valuation_date
        self.horizon_date = self.valuation_date + timedelta(days = 365)
        self.payments_per_year = payments_per_year

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
    

    def compute_nii_from_instruments(
            self,
            fixed_loans,
            floating_loans,
            nmd_model,
            discount_curve,
            rate_shock,
            swaps = None
    ):
        """ 12M NII simulation using instruments including swap carry for hedged NII """
        nii_assets = 0.0
        nii_swaps = 0.0

        # fixed loans -> fixed income over 1Y horizon (contractual interest [no repricing])
        for loan in fixed_loans:
            df = loan.generate_cashflows()
            df['date'] = pd.to_datetime(df['date'])

            cf_1y = df.loc[
                (df['date'] > self.valuation_date) &
                (df['date'] <= self.horizon_date)
            ]

            nii_assets += cf_1y['interest'].sum()

        # floating loans -> reprice using shocked forward rate
        shocked_short_rate = discount_curve.get_zero_rate(t = 1) + rate_shock 

        for loan in floating_loans:
            times, cfs = loan.generate_cashflows(
                discount_curve = discount_curve
            )
            mask = times <= 1
            repriced_interest = shocked_short_rate * loan.notional * times[mask]

            nii_assets += repriced_interest.sum()

        # deposits -> behavioral beta repricing
        deposit_rate_change = nmd_model.deposit_rate_shock(
            rate_shock = rate_shock
        )
        deposit_cost = nmd_model.balance * deposit_rate_change

        # swaps
        if swaps is not None:
            for swap in swaps:
                nii_swaps += swap.compute_1y_nii(
                    discount_curve = discount_curve,
                    rate_shock = rate_shock
                )

        nii_total = nii_assets - deposit_cost + nii_swaps # fixed + floating - nmd + swaps

        return nii_total
    
    # Basel shock scenario based NII computation
    def compute_nii_scenario(
            self,
            fixed_loans,
            floating_loans,
            nmd_model,
            base_curve,
            discount_curve,
            swaps = None
    ):
        """ 
        Computes NII under a shocked curve by inferring the short-rate shock
        from the curve shift 
        """

        # infer short-rate shock from 1Y tenor for both base and shocked scenarios
        base_short_rate = base_curve.get_zero_rate(t=1)
        shocked_short_rate = discount_curve.get_zero_rate(t=1)

        # Basel shock rate
        rate_shock = shocked_short_rate - base_short_rate

        # scenario-based NII
        nii = self.compute_nii_from_instruments(
            fixed_loans = fixed_loans,
            floating_loans = floating_loans,
            nmd_model = nmd_model,
            discount_curve = discount_curve,
            rate_shock = rate_shock,
            swaps = swaps  
        )

        return nii

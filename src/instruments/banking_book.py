import numpy as np

class BankingBook:
    """
    Aggregates full banking book:
        * Assets
        * Liabilities
        * Non-maturity deposits (behavioral)
    """
    def __init__(self):
        
        self.assets = []
        self.liabilities = []
        self.nmd_models = []

    # adding banking book instruments
    def add_asset(self, instrument):
        self.assets.append(instrument)
    
    def add_liability(self, instrument):
        self.liabilities.append(instrument)
    
    def add_nmd(self, instrument):
        self.nmd_models.append(instrument)
    
    # aggregate instruments
    def _aggregate_instruments(
            self, 
            instruments
    ):
        
        all_times = []
        all_cashflows = []

        for inst in instruments:
            times, cashflows = inst.cashflows()
            all_times.extend(times)
            all_cashflows.extend(cashflows)
        
        return np.array(all_times), np.array(all_cashflows)
    
    
    def asset_cashflows(self):

        return self._aggregate_instruments(self.assets)

    def liability_cashflows(self):

        return self._aggregate_instruments(self.liabilities)
    
    def nmd_cashflows(self):

        all_times = []
        all_cashflows = []

        for nmd in self.nmd_models:
            times, cashflows = nmd.total_cashflows()
            all_times.extend(times)
            all_cashflows.extend(cashflows)
        
        return np.array(all_times), np.array(all_cashflows)
    
    def total_liability_cashflows(self):

        t1, cf1 = self.liability_cashflows()
        t2, cf2 = self.nmd_cashflows()

        times = np.concatenate([t1, t2])
        cashflows = np.concatenate([cf1, cf2])

        # sorted cfs
        idx = np.argsort(times)
        times = times[idx]
        cashflows = cashflows[idx]

        return times, cashflows
    
    # EVE calculation
    def compute_eve(
            self,
            discount_curve
    ):
        
        # Assets PV
        t_assets, cf_assets = self.asset_cashflows()
        pv_assets = discount_curve.present_value(
            cashflows = cf_assets,
            times = t_assets
        )

        # Liabilities PV
        t_liabilities, cf_liabilities = self.liability_cashflows()
        pv_liabilities = discount_curve.present_value(
            cashflows = cf_liabilities,
            times = t_liabilities
        )

        eve = pv_assets - pv_liabilities

        return eve
    
    # EVE computation under Basel shock scenarios
    def compute_Eve_sensitivity(
            self,
            shocked_curve_dict
    ):
        
        results = {}

        for scenario, curve in shocked_curve_dict.items():
            results[scenario] = self.compute_eve(
                discount_curve = curve
            )

        return results
    
    # NII impact from deposits
    def compute_deposit_nii_impact(
            self,
            rate_shock
    ):
        """ Computes 1Y repricing effect """
        impact = 0

        for nmd in self.nmd_models:
            repricing = nmd.deposit_rate_shock(
                rate_shock = rate_shock
            )

            impact += repricing * nmd.balance
        
        return impact



    

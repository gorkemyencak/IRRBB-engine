import numpy as np
from scipy.optimize import minimize

from src.hedging.swaps import Hedge_Tenors

from src.curves.discount_curve import DiscountCurve

from src.instruments.interest_rate_swap import IRSwap

from src.risk.eve_engine import EVEEngine
from src.risk.nii_engine import NIIEngine

class IRRBBOptimizer:
    """ Simultaneously minimizes delta EVE and delta NII aross Basel shock scenarios """
    def __init__(
            self,
            base_curve_df,
            shocked_curves,
            fixed_loans,
            floating_loans,
            nmd_model,
            valuation_date,
            w_eve = 0.6,        # weights are assumed that we are running a retail bank
            w_nii = 0.4,
            lambda_reg = 1e-8,  # regularization term to avoid over-hedging
            hedge_budget = 1e7  # total allowable hedge limit on trades across term structure
    ):
        
        self.base_curve_df = base_curve_df
        self.shocked_curves = shocked_curves
        self.fixed_loans = fixed_loans
        self.floating_loans = floating_loans
        self.nmd_model = nmd_model
        self.valuation_date = valuation_date
        self.w_eve = w_eve
        self.w_nii = w_nii
        self.lambda_reg = lambda_reg
        self.hedge_budget = hedge_budget
    
    # creating IRSwap instruments from optimal notionals
    def _build_hedge_swaps(
            self,
            optimal_notionals
    ):
        
        swaps = []
        for tenor, notional in zip(Hedge_Tenors, optimal_notionals):

            pay_fixed = True if notional > 0 else False

            swap = IRSwap(
                notional = notional,
                fixed_rate = 0.035, # dummy par rate
                maturity_years = tenor,
                payments_per_year = 4,
                pay_fixed = pay_fixed
            )

            swaps.append(swap)
        
        return swaps
    
    # computing base EVE & base NII with hedge swaps
    def _compute_base_metrics(
            self,
            hedge_swaps
    ):
        
        base_dc = DiscountCurve(
            self.base_curve_df
        )

        eve_engine = EVEEngine(
            discount_curve = base_dc,
            valuation_date = self.valuation_date
        )

        nii_engine = NIIEngine(
            valuation_date = self.valuation_date
        )

        base_eve = eve_engine.compute_eve_from_instruments(
            fixed_loans = self.fixed_loans,
            floating_loans = self.floating_loans,
            nmd_model = self.nmd_model,
            swaps = hedge_swaps
        )

        base_nii = nii_engine.compute_nii_from_instruments(
            fixed_loans = self.fixed_loans,
            floating_loans = self.floating_loans,
            nmd_model = self.nmd_model,
            discount_curve = base_dc,
            rate_shock = 0,
            swaps = hedge_swaps
        )

        return base_eve, base_nii
    
    # computing delta EVE and NII vectors across Basel scenarios
    def _scenario_delta_vectors(
            self,
            hedge_swaps
    ):
        
        base_dc = DiscountCurve(
            curve_df = self.base_curve_df
        )

        base_eve, base_nii = self._compute_base_metrics(
            hedge_swaps = hedge_swaps
        )

        delta_eve_vector, delta_nii_vector = [], []

        for shocked_curve_df in self.shocked_curves.values():

            shocked_dc = DiscountCurve(
                curve_df = shocked_curve_df
            )

            eve_engine = EVEEngine(
                discount_curve = shocked_dc,
                valuation_date = self.valuation_date
            )

            nii_engine = NIIEngine(
                valuation_date = self.valuation_date
            )

            shocked_eve = eve_engine.compute_eve_from_instruments(
                fixed_loans = self.fixed_loans,
                floating_loans = self.floating_loans,
                nmd_model = self.nmd_model,
                swaps = hedge_swaps
            )

            rate_shock = (
                shocked_dc.curve.loc[shocked_dc.curve['tenor_years'] == 1, 'zero_rate'].values[0]
                - base_dc.curve.loc[base_dc.curve['tenor_years'] == 1, 'zero_rate'].values[0]
            )

            shocked_nii = nii_engine.compute_nii_from_instruments(
                fixed_loans = self.fixed_loans,
                floating_loans = self.floating_loans,
                nmd_model = self.nmd_model,
                discount_curve = shocked_dc,
                rate_shock = rate_shock,
                swaps = hedge_swaps
            )

            delta_eve_vector.append(shocked_eve - base_eve)
            delta_nii_vector.append(shocked_nii - base_nii)

        return np.array(delta_eve_vector), np.array(delta_nii_vector)
            
    # constraint on total hedge amount 
    def total_hedge_constraint(
            self,
            notionals
    ):
        """ Total absolute hedge size must stay within budget """
        return self.hedge_budget - np.sum(np.abs(notionals))

    # least square minimizer as an objective function
    def objective(
            self,
            optimal_notinoals
    ):
        
        hedge_swaps = self._build_hedge_swaps(
            optimal_notionals = optimal_notinoals
        )

        delta_eve_vector, delta_nii_vector = self._scenario_delta_vectors(
            hedge_swaps = hedge_swaps
        )

        # IRRBB risks
        eve_risk = np.sum(delta_eve_vector ** 2)
        nii_risk = np.sum(delta_nii_vector ** 2)

        # regularization term to avoid over-hedging
        reg_term = self.lambda_reg * np.sum(optimal_notinoals ** 2)

        # total objective including regularization term
        obj_total = (
            self.w_eve * eve_risk
            + self.w_nii * nii_risk
            + reg_term
        )

        return obj_total
    
    # optimization runner
    def optimize(self):
        """
        Runs hedge optimization
        It returns optimal swap notionals vector under Basel shock scenarios for a given bound per each vector element
        """
        # start with no hedges
        x0 = np.zeros(len(Hedge_Tenors))

        # Limit trade sizes per each tenor
        hedge_bounds = [(-5e6, +5e6) for _ in Hedge_Tenors]

        # Constraint on the total hedge amount
        constraints = [
            {
                'type': 'ineq',
                'fun': self.total_hedge_constraint
            }
        ]

        result = minimize(
            fun = self.objective,
            x0 = x0,
            bounds = hedge_bounds,
            constraints = constraints,
            method = 'SLSQP'
        )

        return result.x
    

    def hedge_report(
            self,
            opt_notionals
    ):
        
        print("\n--- Recommended Hedge Trades ---\n")

        total = np.sum(np.abs(opt_notionals))
        print(f"Total hedge notionals used: {total:,.0f}")
        print(f"Budget limit: {self.hedge_budget:,.0f}\n")

        for tenor, notional in zip(Hedge_Tenors, opt_notionals):

            side = "Pay Fixed" if notional > 0 else "Receive Fixed"

            print(f"{side:12} {abs(notional)/1e6:8.4f}m {tenor}Y swap")
    

    def hedge_diagnostics(
            self,
            opt_notionals
    ):
        
        hedge_swaps = self._build_hedge_swaps(
            optimal_notionals = opt_notionals
        )

        delta_eve_vector, delta_nii_vector = self._scenario_delta_vectors(
            hedge_swaps = hedge_swaps
        )

        eve_risk = np.sum(delta_eve_vector ** 2)
        nii_risk = np.sum(delta_nii_vector ** 2)
        reg_term = self.lambda_reg * np.sum(opt_notionals ** 2)

        obj_total = (
            self.w_eve * eve_risk
            + self.w_nii * nii_risk
            + reg_term
        )

        print("\n--- Hedge Diagnostics ---\n")
        print(f"EVE risk component: {eve_risk:,.2f}")
        print(f"NII risk component: {nii_risk:,.2f}")
        print(f"Regularization: {reg_term:,.2f}")
        print(f"Objective value: {obj_total:,.2f}")

    # helper function to compare different scenarios
    def scenario_impact(
            self,
            optimal_notionals
    ):
        
        hedge_swaps = self._build_hedge_swaps(
            optimal_notionals = optimal_notionals
        )

        delta_eve_vec, delta_nii_vec = self._scenario_delta_vectors(
            hedge_swaps = hedge_swaps
        )

        return delta_eve_vec, delta_nii_vec


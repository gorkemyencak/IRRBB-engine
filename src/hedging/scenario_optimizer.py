import numpy as np
from scipy.optimize import minimize

from src.hedging.swaps import Hedge_Tenors

from src.instruments.interest_rate_swap import IRSwap

from src.risk.eve_engine import EVEEngine

from src.curves.discount_curve import DiscountCurve

class ScenarioHedgeOptimizer:
    """ Deriving swap notionals that minimize Basel IRRBB delta EVE across all shock scenarios """
    def __init__(
            self,
            base_curve_df,
            shocked_curves,
            fixed_loans,
            floating_loans,
            nmd_model,
            valuation_date
    ):
        
        self.base_curve_df = base_curve_df
        self.shocked_curves = shocked_curves
        self.fixed_loans = fixed_loans
        self.floating_loans = floating_loans
        self.nmd_model = nmd_model
        self.valuation_date = valuation_date

    # creating IRSwap instruments from optimizer notionals
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
    
    # computing base EVE with hedge swaps
    def _compute_base_eve(
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

        eve = eve_engine.compute_eve_from_instruments(
            fixed_loans = self.fixed_loans,
            floating_loans = self.floating_loans,
            nmd_model = self.nmd_model,
            swaps = hedge_swaps
        )

        return eve
    
    # computing delta EVE vector across Basel scenarios
    def _scenario_delta_eve_vector(
            self,
            hedge_swaps
    ):
        
        base_eve = self._compute_base_eve(
            hedge_swaps = hedge_swaps
        )

        delta_eve_results = []

        for shocked_curve_df in self.shocked_curves.values():

            shocked_dc = DiscountCurve(
                shocked_curve_df
            )

            eve_engine = EVEEngine(
                discount_curve = shocked_dc,
                valuation_date = self.valuation_date
            )

            shocked_eve = eve_engine.compute_eve_from_instruments(
                fixed_loans = self.fixed_loans,
                floating_loans = self.floating_loans,
                nmd_model = self.nmd_model,
                swaps = hedge_swaps
            )

            delta_eve = shocked_eve - base_eve

            delta_eve_results.append(delta_eve)
        
        return np.array(delta_eve_results)
    
    # least square minimizer as an objective function
    def objective(
            self,
            optimal_notinoals
    ):
        
        hedge_swaps = self._build_hedge_swaps(
            optimal_notionals = optimal_notinoals
        )

        delta_eve_vec = self._scenario_delta_eve_vector(
            hedge_swaps = hedge_swaps
        )

        # minimize total IRRBB risk
        obj = np.sum(delta_eve_vec ** 2)

        return obj
    
    # optimization runner
    def optimize(self):
        """
        Runs hedge optimization
        It returns optimal swap notionals vector under Base shock scenarios for a given bound per each vector element        
        """
        # start with no hedges
        x0 = np.zeros(len(Hedge_Tenors))

        # Limit trade sizes per each tenor
        hedge_bounds = [(-5000000, +5000000) for _ in Hedge_Tenors]

        result = minimize(
            fun = self.objective,
            x0 = x0,
            bounds = hedge_bounds,
            method = 'L-BFGS-B'
        )

        return result.x
    

    def hedge_report(
            self,
            opt_notionals
    ):
        
        print("\n--- Recommended Hedge Trades ---\n")
        for tenor, notional in zip(Hedge_Tenors, opt_notionals):

            side = "Pay Fixed" if notional > 0 else "Receive Fixed"

            print(f"{side:12} {abs(notional)/1e6:8.4f}m {tenor}Y swap")


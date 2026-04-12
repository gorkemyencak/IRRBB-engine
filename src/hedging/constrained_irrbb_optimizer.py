import numpy as np
from scipy.optimize import minimize

from src.hedging.swaps import Hedge_Tenors

from src.curves.discount_curve import DiscountCurve

from src.instruments.interest_rate_swap import IRSwap

from src.risk.eve_engine import EVEEngine
from src.risk.nii_engine import NIIEngine

class ConstrainedIRRBBOptimizer:
    """ 
    Finds the cheapest swap hedgee that keeps the bank within EVE and NII limits 
    across Basel shock scenarios 

    In mathematical form:
    Minimize
        Hedge Cost
    Subject to:
        |Delta EVE| <= EVE_limit
        |Delta NII| <= NII_limit
    """
    def __init__(
            self,
            base_curve_df,
            shocked_curves,
            fixed_loans,
            floating_loans,
            nmd_model,
            valuation_date,
            eve_limit,
            nii_limit,
            hedge_cost_per_notional = 0.001
    ):
        
        self.base_curve_df = base_curve_df
        self.shocked_curves = shocked_curves
        self.fixed_loans = fixed_loans
        self.floating_loans = floating_loans
        self.nmd_model = nmd_model
        self.valuation_date = valuation_date
        self.eve_limit = eve_limit
        self.nii_limit = nii_limit
        self.hedge_cost_per_notional = hedge_cost_per_notional

    # creating IRSwap instruments from optimal notionals
    def _build_hedge_swaps(
            self,
            optimal_notionals
    ):
        
        swaps = []
        for tenor, notional in zip(Hedge_Tenors, optimal_notionals):

            pay_fixed = True if notional > 0 else False

            swap = IRSwap(
                notional = abs(notional),
                fixed_rate = 0.035, # dummy par rate
                maturity_years = tenor,
                payments_per_year = 4,
                pay_fixed = pay_fixed
            )

            swaps.append(swap)
        
        return swaps
    
    # computing delta EVE vector across different Basel shock scenarios with hedge swaps
    def _compute_eve_vector(
            self,
            hedge_swaps
    ):
        """ It returns a vector consisting of Delta_EVE for all Basel shock scenarios """
        base_dc = DiscountCurve(
            curve_df = self.base_curve_df
        )

        eve_engine = EVEEngine(
            discount_curve = base_dc,
            valuation_date = self.valuation_date
        )

        base_eve_with_hedge_swap = eve_engine.compute_eve_from_instruments(
            fixed_loans = self.fixed_loans,
            floating_loans = self.floating_loans,
            nmd_model = self.nmd_model,
            swaps = hedge_swaps
        )

        delta_eve_vector = []
        for shocked_curve_df in self.shocked_curves.values():

            shocked_dc = DiscountCurve(
                curve_df = shocked_curve_df
            )

            eve_engine_shocked = EVEEngine(
                discount_curve = shocked_dc,
                valuation_date = self.valuation_date
            )

            shocked_eve_with_hedge_swap = eve_engine_shocked.compute_eve_from_instruments(
                fixed_loans = self.fixed_loans,
                floating_loans = self.floating_loans,
                nmd_model = self.nmd_model,
                swaps = hedge_swaps
            )

            delta_eve = shocked_eve_with_hedge_swap - base_eve_with_hedge_swap

            delta_eve_vector.append(delta_eve)
        
        return np.array(delta_eve_vector)
    
    # computing delta NII vector across different Basel shock scenarios with hedge swaps
    def _compute_nii_vector(
            self,
            hedge_swaps
    ):
        """ It returns a vector consisting of Delta_NII for all Basel shock scenarios """
        base_dc = DiscountCurve(
            curve_df = self.base_curve_df
        )

        nii_engine = NIIEngine(
            valuation_date = self.valuation_date,
            payments_per_year = 4
        )

        base_nii_with_hedge_swap = nii_engine.compute_nii_from_instruments(
            fixed_loans = self.fixed_loans,
            floating_loans = self.floating_loans,
            nmd_model = self.nmd_model,
            discount_curve = base_dc,
            rate_shock = 0,
            swaps = hedge_swaps
        )

        delta_nii_vector = []
        for shocked_curve_df in self.shocked_curves.values():

            shocked_dc = DiscountCurve(
                curve_df = shocked_curve_df
            )

            nii_engine_shocked = NIIEngine(
                valuation_date = self.valuation_date,
                payments_per_year = 4
            )

            shocked_nii_with_hedge_swap = nii_engine_shocked.compute_nii_scenario(
                fixed_loans = self.fixed_loans,
                floating_loans = self.floating_loans,
                nmd_model = self.nmd_model,
                base_curve = base_dc,
                discount_curve = shocked_dc,
                swaps = hedge_swaps
            )

            delta_nii = shocked_nii_with_hedge_swap - base_nii_with_hedge_swap

            delta_nii_vector.append(delta_nii)
        
        return np.array(delta_nii_vector)
    
    # EVE constraint
    def _eve_constraint(self):

        constraints = []
        for i in range(len(self.shocked_curves)):

            def constraint(notionals, idx = i):
                
                swaps = self._build_hedge_swaps(optimal_notionals = notionals)
                eve_vec = self._compute_eve_vector(hedge_swaps = swaps)

                return self.eve_limit - abs(eve_vec[idx])

            constraints.append({
                'type': 'ineq',
                'fun': constraint
            }) 
        
        return constraints
    
    # NII constraint
    def _nii_constraint(self):

        constraints = []
        for i in range(len(self.shocked_curves)):

            def constraint(notionals, idx = i):

                swaps = self._build_hedge_swaps(optimal_notionals = notionals)
                nii_vec = self._compute_nii_vector(hedge_swaps = swaps)

                return self.nii_limit - abs(nii_vec[idx])
            
            constraints.append({
                'type': 'ineq',
                'fun': constraint
            })
        
        return constraints
    
    # the cost of using hedge option as an objective
    def hedge_cost(
            self,
            optimal_notionals
    ):

        total_hedge_cost = np.sum(np.abs(optimal_notionals)) * self.hedge_cost_per_notional 

        return total_hedge_cost
    
    # optimization runner
    def optimize(self):
        """ 
        Runs constrained hedge optimization 
        It returns optimal swap notionals satisfying EVE and NII limits 
        under Basel shock scenarios for a given bound per each vector element
        """
        # start with no hedges
        x0 = np.zeros(len(Hedge_Tenors))

        # limit trade sizes per each tenor
        hedge_bounds = [(-5e6, +5e6) for _ in Hedge_Tenors]

        # EVE and NII limit contraints
        constraints = []
        constraints += self._eve_constraint()
        constraints += self._nii_constraint()

        result = minimize(
            fun = self.hedge_cost,
            x0 = x0,
            bounds = hedge_bounds,
            constraints = constraints,
            method = 'SLSQP',
            options = {'maxiter': 10}
        )

        return result.x


    def hedge_report(
            self,
            opt_notionals
    ):
        
        print("\n--- Recommended Hedge Trades ---\n")

        total = np.sum(np.abs(opt_notionals))
        print(f"Total hedge notionals used: {total:,.0f}")

        for tenor, notional in zip(Hedge_Tenors, opt_notionals):

            side = "Pay Fixed" if notional > 0 else "Receive Fixed"

            print(f"{side:12} {abs(notional)/1e6:8.4f}m {tenor}Y swap")
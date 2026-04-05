import numpy as np
import pandas as pd
from copy import deepcopy

from src.curves.discount_curve import DiscountCurve

from src.risk.eve_engine import EVEEngine

Key_Tenors = [1, 2, 5, 10, 20, 30]

class KeyRateDV01:
    """ Computes bank Key-Rate DV01 by shocking one curve node at a time """
    def __init__(
            self,
            base_curve_df: pd.DataFrame,
            fixed_loans: list,
            floating_loans: list,
            nmd_model,
            valuation_date
    ):
        
        self.base_curve_df = base_curve_df
        self.fixed_loans = fixed_loans
        self.floating_loans = floating_loans
        self.nmd_model = nmd_model
        self.valuation_date = valuation_date

    # apply local bump around shocked tenor
    def _apply_local_bump(
            self,
            tenor: int,
            shock_bps: float
    ):
        
        curve = deepcopy(self.base_curve_df)

        # apply local bump only on the shocked tenor
        idx = (np.abs(curve['tenor_years'] - tenor)).argmin()

        curve.loc[idx, 'zero_rate'] += shock_bps / 10000 # type: ignore

        #curve.loc[curve['tenor_years'] == tenor, 'zero_rate'] += shock_bps / 10000
        
        return curve 

    # shock a single tenor of the zero curve
    def _shock_single_tenor(
            self,
            tenor: int,
            shock_bps: float
    ):

        return self._apply_local_bump(tenor = tenor, shock_bps = shock_bps)
    
    # compute EVE under shocked curve
    def _compute_EVE_shocked_curve(
            self,
            curve_df
    ):
        
        discount_curve = DiscountCurve(curve_df = curve_df)

        eve_engine = EVEEngine(
            discount_curve = discount_curve,
            valuation_date = self.valuation_date
        )

        eve = eve_engine.compute_eve_from_instruments(
            fixed_loans = self.fixed_loans,
            floating_loans = self.floating_loans,
            nmd_model = self.nmd_model
        )

        return eve
    
    # compute Key-Rate DV01 vector
    def compute_key_rate_dv01(
            self,
            shock_bps = 1
    ):
        
        key_rate_dv01_results = {}
        for tenor in Key_Tenors:

            # Shock up
            curve_up = self._shock_single_tenor(
                tenor = tenor,
                shock_bps = shock_bps
            )
            eve_up = self._compute_EVE_shocked_curve(
                curve_df = curve_up
            )

            # Shock down
            curve_down = self._shock_single_tenor(
                tenor = tenor,
                shock_bps = -shock_bps
            )
            eve_down = self._compute_EVE_shocked_curve(
                curve_df = curve_down
            )

            # Arithmetic mean
            kr_dv01 = (eve_up - eve_down) / 2 / (shock_bps / 10000)
            
            key_rate_dv01_results[f"{tenor}Y"] = kr_dv01
        
        return pd.Series(key_rate_dv01_results, name = 'KeyRateDV01')

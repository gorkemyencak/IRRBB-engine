import pandas as pd

from src.risk.IRRBB_shocks import IRRBBShock

class CurveShocks:
    """ Converts base zero curve into Basel shocked zero curves """
    def __init__(
            self,
            zero_curve_df: pd.DataFrame
    ):
        """ 
        Expected cols: 
        tenor_years
        zero_rate
        """
        self.base_curve = zero_curve_df.copy()

        # Rename to match IRRBBShock format
        shock_df = pd.DataFrame({
            'tenor': self.base_curve['tenor_years'],
            'rate': self.base_curve['zero_rate']
        })

        self.shock_engine = IRRBBShock(
            base_curve = shock_df
        )

    
    def _convert_back(
            self,
            shocked_df
    ):
        """ Convert IRRBBShock output into DiscountCurve format """
        return pd.DataFrame({
            'tenor_years': shocked_df['tenor'],
            'zero_rate': shocked_df['rate']
        })
    

    def generate_all_scenarios(self):

        scenarios = {
            'parallel_up': self.shock_engine.parallel_up(),
            'parallel_down': self.shock_engine.parallel_down(),
            'short_up': self.shock_engine.short_rate_up(),
            'short_down': self.shock_engine.short_rate_down(),
            'steepener': self.shock_engine.steepener(),
            'flattener': self.shock_engine.flattener()
        }

        return {
            name: self._convert_back(df)
            for name, df in scenarios.items()
        }

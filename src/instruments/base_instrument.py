import pandas as pd

class BaseInstrument:
    """ Parent class for banking book instruments """
    def __init__(
            self,
            notional: float,
            start_date: str,
            maturity_date: str
    ):
        
        self.notional = notional
        self.start_date = start_date
        self.maturity_date = maturity_date
    

    def generate_cashflows(self) -> pd.DataFrame:
        """ 
        Child classes must implement generate_cashflows method 
        Must return DataFrame with columns: ['date', 'principal', 'interest', 'total_cashflow']
        """
        raise NotImplementedError("generate_cashflows() must be implemented in child class")


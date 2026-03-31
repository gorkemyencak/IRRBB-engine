import numpy as np
import pandas as pd

from datetime import datetime

BUCKETS = [
    ("1M", 0, 1/12),
    ("3M", 1/12, 3/12),
    ("6M", 3/12, 6/12),
    ("1Y", 6/12, 1),
    ("2Y", 1, 2),
    ("3Y", 2, 3),
    ("5Y", 3, 5),
    ("7Y", 5, 7),
    ("10Y", 7, 10),
    ("15Y", 10, 15),
    ("20Y", 15, 20),
    ("30Y", 20, np.inf)
]

class CashflowBucketer:
    """ Cashflow bucket class implementing standard supervisory IRRBB time buckets """
    def __init__(
            self,
            valuation_date: datetime
    ):
        
        self.valuation_date = valuation_date

    # convert date series into year fractions
    def _year_fraction(
            self,
            date_series: pd.Series
    ) -> pd.Series:
        
        return (date_series - self.valuation_date).dt.days / 365.25
    

    def _assign_bucket(
            self,
            time_to_maturity: float 
    ) -> str:
        
        for name, lower, upper in BUCKETS:
            if lower < time_to_maturity <= upper:
                return name
        return "30Y"
    

    def assign_buckets(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        
        df = df.copy()

        # compute the time to maturity
        df['ttm'] = self._year_fraction(
            date_series = df['date']
        )

        # remove past cashflows
        df = df.loc[df['ttm'] > 0]

        # assign basel buckets
        df['bucket'] = df['ttm'].apply(self._assign_bucket)
        
        return df
    

    def bucket_cashflows(
            self,
            portfolio_cf: pd.DataFrame
    ) -> pd.DataFrame:
        
        df = self.assign_buckets(df = portfolio_cf)

        bucketed_cf = (
            df
            .groupby('bucket')
            .agg(
                principal = ('principal', 'sum'),
                interest = ('interest', 'sum'),
                total_cashflow = ('total_cashflow', 'sum')
            )
            .reindex([b[0] for b in BUCKETS])
            .fillna(0)
            .reset_index()
        )

        return bucketed_cf
    
    # computing net cashflow per bucket
    def compute_gap(
            self,
            bucketed_cf: pd.DataFrame
    ) -> pd.DataFrame:
        
        gap = bucketed_cf.copy()
        gap['gap'] = gap['total_cashflow']
        gap['cumulative_gap'] = gap['gap'].cumsum()
        
        return gap
    




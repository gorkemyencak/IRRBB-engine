import os
import pandas as pd
from pathlib import Path
from fredapi import Fred
from src.config.datasets_config import SERIES

FRED_API_KEY = "41504b53ebf306bcd89ceb69bbd6eba8"

class FredYCDownloader:
    """ Downloading US Treasury yield curve from FRED """
    def __init__(
            self,
            data_dir = 'data/raw/yield_curve'
    ):
        
        project_root = Path(__file__).resolve().parents[2]
        self.data_dir = project_root / data_dir
        self.data_dir.mkdir(parents = True, exist_ok = True)

        self.fred = Fred(api_key = FRED_API_KEY)

    
    def download(self) -> pd.DataFrame:

        file_path = self.data_dir / 'fred_yield_curve.csv'

        if not self.data_dir.exists():
            print("Downloading Treasury yield curve from FRED..")
            df = pd.DataFrame()

            for tenor, series in SERIES.items():
                df[tenor] = self.fred.get_series(series)
            
            df.index.name = 'Date'
            df = df.sort_index()

            # save the dataframe locally
            df.to_csv(file_path)

            return df  
        
        else:
            print("Treasury yield curve dataset already downloaded..")
            return pd.read_csv(file_path)

        


    

import os
import pandas as pd
from pathlib import Path

from src.data.kaggle_downloader import KaggleDownloader
from src.config.datasets_config import DATASETS

class KaggleCSVLoader:

    def __init__(
            self,
            dataset_key: str,
            data_dir: str = 'data/raw'
    ):
        
        self.dataset_key = dataset_key
        self.dataset_config = DATASETS[dataset_key]

        project_root = Path(__file__).resolve().parents[2]
        self.base_data_dir = project_root / data_dir
        self.base_data_dir.mkdir(parents = True, exist_ok = True)

        self.dataset_dir = self.base_data_dir / self.dataset_key
        self.dataset_dir.mkdir(parents = True, exist_ok = True)

        self.parquet_dir = self.base_data_dir / 'parquet' / self.dataset_key
        self.parquet_dir.mkdir(parents = True, exist_ok = True)

        self.dataset_downloader = KaggleDownloader(
            kaggle_id = self.dataset_config['kaggle_id'],
            download_path = self.dataset_dir
        )

    
    def download_dataset(self):

        missing_files = [
            file for file in self.dataset_config['files']
            if not(self.dataset_dir / file).exists()
        ]

        if missing_files:
            print(f"Missing files {missing_files}. Downloading dataset..")
            self.dataset_downloader.download()
        else:
            print(f"{self.dataset_key} already downloaded..")


    def load_local_csv(
            self,
            file_name: str
    ) -> pd.DataFrame:
        """ Loading CSV from local raw data directory with parquet caching """
        # Fast Path
        parquet_path = self._parquet_path(file_name)

        if parquet_path.exists():
            print(f"Loading cached parquet {parquet_path.name}")
            return pd.read_parquet(parquet_path)

        # Slow Path
        file_path = self.dataset_dir / file_name    

        if not file_path.exists():
            raise FileNotFoundError(f"{file_name} not found in {self.dataset_dir}") 
        
        # Auto-detect compression
        if file_name.endswith('.gz'):
            df = pd.read_csv(file_path, compression = 'gzip', low_memory = False)
        else:
            df = pd.read_csv(file_path, low_memory = False)
        
        # Save parquet cache
        self._save_parquet(df, file_name)
        
        return df
    

    def load_all(self):

        data = {}
        for file in self.dataset_config['files']:
            data[file] = self.load_local_csv(file)
        
        return data
    

    def save_df(
            self,
            df: pd.DataFrame,
            file_name: str
    ):
        """ Saving dataframe into local raw data directory """
        file_path = self.dataset_dir / file_name

        df.to_csv(
            file_path,
            index = False
        )

        print(f"Dataframe {file_name} saved into local raw data directory..")
    

    def _parquet_path(
            self,
            file_name: str
    ) -> Path:
        
        return self.parquet_dir / file_name.replace('csv.gz', '.parquet').replace('.csv', '.parquet')

    
    def _save_parquet(
            self,
            df: pd.DataFrame,
            file_name: str
    ):
        
        parquet_path = self._parquet_path(file_name)
        df.to_parquet(
            parquet_path,
            index = False
        )
        print(f"Cached -> {parquet_path}")
    


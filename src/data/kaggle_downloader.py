import os
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

class KaggleDownloader:
    """ Kaggle dataset downloader """

    def __init__(
            self,
            kaggle_id: str,
            download_path: Path
    ):
        
        self.kaggle_id = kaggle_id
        self.download_path = download_path

    
    def download(self):
        print(f"Downloading Kaggle dataset: {self.kaggle_id}")

        self.download_path.mkdir(parents = True, exist_ok = True)

        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(
            self.kaggle_id,
            path = self.download_path,
            unzip = True
        )

        print("Download completed..")



    


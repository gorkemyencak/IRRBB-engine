import os

from src.data.kaggle_loader import KaggleCSVLoader
from src.data.fred_downloader import FredYCDownloader

FRED_API_KEY = "41504b53ebf306bcd89ceb69bbd6eba8"

def main():

    loader_yc = FredYCDownloader()
    data_yc = loader_yc.download() 

    loader_ld = KaggleCSVLoader(dataset_key = 'loan_default')
    loader_ld.download_dataset()
    data_ld = loader_ld.load_all()
    accepted_ld = data_ld['accepted_2007_to_2018Q4.csv.gz']
    rejected_ld = data_ld['rejected_2007_to_2018Q4.csv.gz']

    print(data_yc.head())
    print(accepted_ld.head())
    print(rejected_ld.head())

if __name__ == '__main__':
    main()

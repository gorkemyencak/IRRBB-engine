import os

from src.data.kaggle_loader import KaggleCSVLoader

def main():

    loader_yc = KaggleCSVLoader(dataset_key = 'yield_curve')
    loader_yc.download_dataset()
    data_yc = loader_yc.load_all()
    df_yc = data_yc['index.csv']

    loader_ld = KaggleCSVLoader(dataset_key = 'loan_default')
    loader_ld.download_dataset()
    data_ld = loader_ld.load_all()
    accepted_ld = data_ld['accepted_2007_to_2018Q4.csv.gz']
    rejected_ld = data_ld['rejected_2007_to_2018Q4.csv.gz']

    print(df_yc.head())
    print(accepted_ld.head())
    print(rejected_ld.head())

if __name__ == '__main__':
    main()

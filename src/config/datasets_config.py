DATASETS = {
    "yield_curve": {
        "kaggle_id": "federalreserve/interest-rates",
        "files": [
            "index.csv"
        ]
    },
    "loan_default": {
        "kaggle_id": "wordsforthewise/lending-club",
        "files": [
            "accepted_2007_to_2018Q4.csv.gz",
            "rejected_2007_to_2018Q4.csv.gz"
        ]
    }
}

SERIES = {
    '1M': 'DGS1MO',
    '3M': 'DGS3MO',
    '6M': 'DGS6MO',
    '1Y': 'DGS1',
    '2Y': 'DGS2',
    '5Y': 'DGS5',
    '10Y': 'DGS10',
    '30Y': 'DGS30'
}

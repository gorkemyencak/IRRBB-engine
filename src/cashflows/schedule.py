import pandas as pd

def generate_payment_schedule(
        start_date,
        maturity_date,
        frequency = 'M'
):
    """ 
    Generates payment schedule between start and maturity 
    
    Frequency:
    M: monthly
    Q: quarterly
    S: semi-annual
    A: annual
    """
    freq_map = {
        'M': 'MS',
        'Q': 'QS',
        'S': '2QS',
        'A': 'AS'
    }

    if frequency not in freq_map:
        raise ValueError('Unsupported frequency!')
    
    dates = pd.date_range(
        start = start_date,
        end = maturity_date,
        freq = freq_map[frequency] 
    )

    # ensure that maturity data is included in the data
    if dates[-1] != pd.to_datetime(maturity_date):
        dates = dates.append(pd.Index([pd.to_datetime(maturity_date)]))
    
    return dates



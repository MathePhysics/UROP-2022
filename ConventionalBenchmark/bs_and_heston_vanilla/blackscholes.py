"""Implements Black-Scholes option pricing model."""  

import numpy as np
from scipy.stats import norm

def generate_bs_vec(df):
    """
    Produces result for multiple B-S runs for call options only (analytical).  

    Args:
        - df: dataframe, containing all parameters and has the following entries  
            - underlyings_price: float, current price of underlying
            - volatility: float, current volatility of underlying
            - rate: float, risk free rate
            - strike: float, strike price
            - days_to_maturity: float, days to maturity
            - moneyness: float, moneyness of option

    Output:
        - result: ndarray, containing prices for each sample
    """
    S   = df['underlyings_price'].values 
    vol   = df['volatility'].values 
    r     = df['rate'].values   
    K     = df['strike'].values  
    T     = df['days_to_maturity'].values / 365
    m     = df['moneyness'].values

    d1 = (np.log(m) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    S_call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return S_call

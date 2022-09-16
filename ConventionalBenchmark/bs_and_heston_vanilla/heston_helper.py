"""Implements various functions for the Heston model."""  

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm


def generate_heston_paths(S, T, K, r, kappa, theta, v_0, rho, xi, 
                          steps, num_sims):  
    '''
    Produces result for a single heston run.
    
    '''
    dt = T/steps
    # size = (num_sims, steps)
    # prices = np.zeros(size)
    # vols = np.zeros(size)
    S_t = S + np.zeros(num_sims)
    v_t = v_0 + np.zeros(num_sims)
    for t in tqdm(range(steps), colour="green"):
    # for t in range(steps):
        # [hex (#00ff00), BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE]
        WT1 = np.random.normal(0,1,size=num_sims)
        WT2 = np.random.normal(0,1,size=num_sims)
        WT3 = rho * WT1 + np.sqrt(1-rho**2)*WT2
        # WT = np.vstack((WT1, WT3)).T

        v_t = np.maximum(v_t, 0)
        S_t = S_t*(np.exp( (r- 0.5*v_t)*dt+ np.sqrt(v_t * dt) * WT1 )) #WT[:,0]
        v_t = v_t + kappa*(theta-v_t)*dt + xi*np.sqrt(v_t * dt)*WT3    #WT[:,1]
        # prices[:, t] = S_t # can be returned when plotting is required
        # vols[:, t] = v_t   # omitted to save memory 
    
    S_call = np.exp(-1 * r * T) * np.sum(np.maximum(S_t - K, 0)) / num_sims
    S_put = np.exp(-1 * r * T) * np.sum(np.maximum(K-S_t, 0)) / num_sims
    nu_avg = np.mean(v_t)
    
    return [S_call, S_put, nu_avg]    


def generate_heston_paths_test(S, T, K, r, kappa, theta, v_0, rho, xi, 
                          steps, num_sims):  
    '''
    Produces result for a single heston run for testing the vector version.

    Args:
    - S: np.array, contains spot prices of the underlying assets in the basket
    - T: float, days to maturity divided by 365
    - K: float, strike price
    - r: float, the risk-free rate
    - kappa: float, mean reversion rate of variance
    - theta: float, long-term average price variance
    - v_0: float, initial volatility
    - rho: np.array, contains correlations between each random underlying return process and its associated random volatility process  
    - xi: float, volatility of variance
    - steps: int, num time steps
    - num_sims: int, no. of simulations for each sample 
    '''
    dt = T/steps
    dim = len(S)
    S_t = np.repeat([S], num_sims,axis=0)
    v_t = v_0 + np.zeros(num_sims)[:,np.newaxis]

    for t in range(steps):
        WT1 = np.random.normal(0,1,size=(num_sims,dim))
        WT2 = np.random.normal(0,1,size=(num_sims,dim))
        WT3 = rho * WT1 + np.sqrt(1-rho**2)*WT2

        v_t = np.maximum(v_t, 0)
        S_t = S_t*(np.exp( (r - 0.5*v_t)*dt+ np.sqrt(v_t * dt) * WT1)) 
        v_t = v_t + kappa*(theta-v_t)*dt + xi*np.sqrt(v_t * dt)*WT3    

    S_t = np.mean(S_t,axis = 1)
    S_call = np.exp(-1 * r * T) * np.sum(np.maximum(S_t - K, 0)) / num_sims
    
    return S_call


def generate_heston_vec(df, steps=1000, num_sims=100000):  

    '''
    Produces result for multiple heston runs for call options only.

    Args:  
        - df: dataframe, containing all parameters
        - steps: int, num time steps
        - num_sims: int, no. of simulations for each sample  

    Output:  
        - result: ndarray, containing average prices over num_sims
    '''  

    N     = len(df)
    # out   = np.zeros((N, ))
    dt    = df['days_to_maturity'].values /steps 
    S_t   = df['underlyings_price'].values 
    v_t   = df['volatility'].values 
    r     = df['rate'].values 
    theta = df['mean_volatility'].values 
    kappa = df['reversion'].values 
    xi    = df['var_of_vol'].values 
    K     = df['strike'].values 
    rho   = df['rho'].values 
    T     = df['days_to_maturity'].values * 365
    
    for t in tqdm(range(steps), colour="green"):
        
        # the random normal samples are of shape (num_sims, N)
        WT1 = np.random.normal(0,1,size=(num_sims, N))
        WT2 = np.random.normal(0,1,size=(num_sims, N))
        WT3 = rho * WT1 + np.sqrt(1-rho**2)*WT2

        v_t = np.maximum(v_t, 0)
        S_t = S_t*(np.exp( (r- 0.5*v_t)*dt+ np.sqrt(v_t * dt) * WT1 )) 
        v_t = v_t + kappa*(theta-v_t)*dt + xi*np.sqrt(v_t * dt)*WT3
    
    S_call = np.exp(-1 * r * T) * np.sum(np.maximum(S_t - K, 0), axis = 0) / num_sims
    
    # S_put = np.exp(-1 * r * T) * np.sum(np.maximum(K-S_t, 0)) / num_sims
    nu_avg = np.mean(v_t)
    
    return S_call 
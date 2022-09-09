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


def generate_inputs_nn(real_data, dim, num_samples):
    """
    Returns num_samples numbers of 7 + 2*dim input features 
    which is later fed into the neural networks.

    Args:  
      - real_data: pd dataframe containing the values obtained from wrds
      - dim: int, the number of underlyings in the basket
      - num_samples: number of samples to take  

    Output:
      - inputs: ndarray, 
    
    """
    def rand(num = num_samples):
      return np.random.random(num)

    def corr(dim):
      return -0.05 - 0.7 * rand(dim)

    # Initial spot price of each sampled from historical data
    def spots(dim): 
      return np.array(real_data['underlyings_price'].sample(n=dim, replace=True))

    # randomly sample from historical data
    maturity  = np.array(real_data['days_to_maturity'].sample(n = num_samples, replace = True))
    rate    = np.array(real_data['rate'].sample(n = num_samples, replace = True))
    strikes  = np.array(real_data['strike'].sample(n = num_samples, replace = True))

    # randomly generate parameters
    rate_reversion  = 0.01 + 5 * rand()          # kappa
    vol_of_vol    = 0.01 + 0.7 * rand()         # lambda
    long_term_var = 0.001 + 0.05 * rand()         # theta
    initial_var  = 0.001 + 0.05 * rand()         # sqrt(v0)

    inputs      = np.array([maturity, strikes, initial_var, long_term_var, rate_reversion,
                               vol_of_vol, rate]).T.tolist()
    for i in range(len(inputs)):
        spot = spots(dim)
        inputs[i].insert(0, spot)
        inputs[i].insert(1, corr(dim))        

    inputs = np.array(inputs, dtype = object)
    return inputs  


def generate_heston_paths_vec(df, num_samples, steps=1000, num_sims=100000):  
    '''
    Produces result for multiple heston runs for call options only.

    Args:  
        - df: dataframe, containing all parameters
        - num_samples:
        - steps: int, num time steps
        - num_sims: int, no. of simulations for each sample  

    Output:  
        - result: ndarray, containing average prices over num_sims
    '''  
    N = len(df)
    df[df.columns.values[2:]] = df[df.columns.values[2:]].astype('float')
    dt    = df['days_to_maturity'].values /steps
    S_0   = df['underlyings_price'].values.tolist() # add tolist 
    v_0   = df['volatility'].values  
    r     = df['rate'].values  
    theta = df['mean_volatility'].values  
    kappa = df['reversion'].values  
    xi    = df['vol_of_var'].values  
    K     = df['strike'].values[:,np.newaxis]  
    rho   = np.array([x.astype('float64') for x in df['rho'].values])
    T     = df['days_to_maturity'].values 

    dim = len(S_0[0])
    S_t = np.tile(S_0, (1,num_sims)).reshape(num_samples,num_sims,dim).transpose(1,2,0)

    v_t = (v_0 + np.zeros((num_sims,1))[:,np.newaxis])


    for t in range(steps):
        WT1 = np.random.normal(0,1,size=(num_sims, dim, N))
        WT2 = np.random.normal(0,1,size=(num_sims, dim, N))
        WT3 = rho.T * WT1 + np.sqrt(1-rho.T**2)*WT2

        v_t = np.maximum(v_t, 0)
        S_t = S_t*(np.exp( ((r- 0.5*v_t)*dt + np.sqrt((v_t * dt)) * WT1 )))
        v_t = v_t + kappa*(theta-v_t)*dt + xi*np.sqrt((v_t * dt))*WT3  
        
    S_t = S_t.transpose(2,0,1)
    S_t = np.mean(S_t,axis = 2)
    S_call = np.exp(-1 * r * T) * np.sum(np.maximum(S_t-K,0), axis = 1 )/ num_sims
    
    return S_call
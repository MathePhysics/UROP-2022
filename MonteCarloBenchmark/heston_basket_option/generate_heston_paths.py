def generate_heston_paths(S, T, r, kappa, theta, v_0, rho, xi, 
                          steps, num_sims):  
    '''
    Produces result for a single heston run.
    
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

def generate_heston_paths_vec(df, steps=1000, num_sims=100000):  
    '''
    Produces result for nultiple heston runs.
    
    '''
    N = len(df)
    dt    = df['days_to_maturity'].values /steps 
    S_0   = df['underlyings_price'].values 
    v_0   = df['volatility'].values.astype('float')  
    r     = df['rate'].values.astype('float')  
    theta = df['mean_volatility'].values.astype('float')  
    kappa = df['reversion'].values.astype('float')  
    xi    = df['vol_of_var'].values.astype('float')  
    K     = df['strike'].values.astype('float')[:,np.newaxis]  
    rho   = np.array([x.astype('float64') for x in df['rho'].values])
    T     = df['days_to_maturity'].values.astype('float') 

    dim = len(S_0[0])
    S_t = np.array([np.repeat([S_0[i]], num_sims, axis=0) for i in range(len(S_0))]).transpose(1,2,0)
    v_t = (v_0 + np.zeros((num_sims,1))[:,np.newaxis])


    for t in range(steps):
        WT1 = np.random.normal(0,1,size=(num_sims, dim, N))
        WT2 = np.random.normal(0,1,size=(num_sims,dim, N))
        WT3 = rho.T * WT1 + np.sqrt(1-rho.T**2)*WT2

        v_t = np.maximum(v_t, 0)
        S_t = S_t*(np.exp( ((r- 0.5*v_t)*dt + np.sqrt((v_t * dt).astype('float')) * WT1 ).astype('float')))
        v_t = v_t + kappa*(theta-v_t)*dt + xi*np.sqrt((v_t * dt).astype('float'))*WT3  
        
    S_t = S_t.transpose(2,0,1)
    S_t = np.mean(S_t,axis = 2)
    S_call = np.exp(-1 * r * T) * np.sum(np.maximum(S_t-K,0), axis = 1 )/ num_sims
    
    return S_call
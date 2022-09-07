def generate_inputs_nn(dim, num_samples):
    """
    Returns num_samples numbers of 7 + 2*dim input features 
    which is later fed into the neural networks.

    Args:
    - dim: int, the number of underlyings in the basket
    - num_samples: int, number of samples to be generated
    
    """
    def rand(num = num_samples):
      return np.random.random(num)

    def corr(dim):
      return -0.05 - 0.7 * rand(dim) # 

    def spots(dim): # Initial spot price of each sampled from historical data
      return np.array(real_data['underlyings_price'].sample(n=dim, replace=True))

    maturity  = np.array(real_data['days_to_maturity'].sample(n = num_samples, replace = True))
    rate    = np.array(real_data['rate'].sample(n = num_samples, replace = True))
    strikes  = np.array(real_data['strike'].sample(n = num_samples, replace = True))
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
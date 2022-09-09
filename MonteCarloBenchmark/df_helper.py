"""Implements some helper functions for the Heston model."""

import numpy as np
import pandas as pd  



def flattenDim(dim, inputs):
    """
    
    Returns a pandas dataframe with assest price and rho expanded.  

    Args:  
        - dim: dimension of assest price basket
        - inputs: pandas dataframe of basket options, containing columns
                  'underlyings_price' and 'rho'  

    Output:  
        - outputs: pandas dataframe
    """
    for i in range(dim):
        inputs[f'Underlying_{i}'] = np.vstack(inputs.underlyings_price.values)[:,i]

    for i in range(dim):
        inputs[f'Rho_{i}'] = np.vstack(inputs.rho.values)[:,i]
    inputs = inputs.drop(['underlyings_price', 'rho'], axis=1)
    return inputs
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import normalize, MinMaxScaler 

# define a global variable minmax_scaler for later to reverse the transformation
minmax_scaler = MinMaxScaler()


def getDatasets(dataframe, scaling = True):
    """
    Returns tuple of scaled train, valid, and test datasets.  

    Args:
        - dataframe: ndarray, dataframe of the data  

    Output:  
        - (train_ds, valid_ds, test_ds): tuple of datasets which are between 0 and 1
    
    """
    N = len(dataframe)
    indices = np.random.permutation(N)
    train, val, test = np.split(dataframe[indices], [int(.8*N), int(.9*N)])
    # add normalizing layer here
    if scaling:
        train = minmax_scaler.fit_transform(train)
        val = minmax_scaler.transform(val)
        test = minmax_scaler.transform(test)
    print(train.shape, val.shape, test.shape)
    train_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train, tf.float32))
    valid_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val, tf.float32))
    test_ds  = tf.data.Dataset.from_tensor_slices(tf.cast(test, tf.float32))
    return (train_ds, valid_ds, test_ds)


def shuffle_and_batch_dataset(dataset, batch_size, shuffle_buffer=None):
    """
    Returns the shuffled and batched Dataset.  

    Args:  
        - dataset: tf.data.Dataset
        - batch_size: int, batch size
        - shuffle_buffer: int, shuffle buffer size
    """
    if shuffle_buffer is not None:
      out = dataset.shuffle(shuffle_buffer).batch(batch_size)
    else:
      out = dataset.batch(batch_size)
    
    return out  

# def flattenDim(dim, inputs):
#     """
    
#     Returns a pandas dataframe with assest price and rho expanded.  

#     Args:  
#         - dim: dimension of assest price basket
#         - inputs: pandas dataframe of basket options, containing columns
#                   'underlyings_price' and 'rho'  

#     Output:  
#         - outputs: pandas dataframe
#     """
#     for i in range(dim):
#         inputs[f'Underlying_{i}'] = np.vstack(inputs.underlyings_price.values)[:,i]

#     for i in range(dim):
#         inputs[f'Rho_{i}'] = np.vstack(inputs.rho.values)[:,i]
#     inputs = inputs.drop(['underlyings_price', 'rho'], axis=1)
#     return inputs


def map_dataset(dataset, map_func):
    """
    Return a mapped dataset.  

    Args:
        - dataset: tf.data.Dataset
        - map_func: function, mapping function
    """  
    out = dataset.map(map_func)
    return out  

# map_function
def xy_split(data):
    """
    Returns the x and y tensors from the dataset.  

    Args:
        - data: content of a dataset  

    Output:
        - (x,y): tuple of x the features and y the label  
    """  

    return (data[:,:-1], data[:,-1])  



def pipeline1(dataframe_BS, prefetch = True, scaling = True):
    """
    Returns a tuple of processed prefetched data for training. 

    Args:
        - dataframe_BS: ndarray, dataframe of the data
        - prefetch: boolean, whether to prefetch the data
        - scaling: boolean, whether to scale the data  
        
    Output:
        - (train_ds, valid_ds, test_ds): tuple of datasets
    """  
    train_ds, valid_ds, test_ds = getDatasets(dataframe_BS, scaling)
    train_ds = shuffle_and_batch_dataset(train_ds, batch_size=32)
    valid_ds = shuffle_and_batch_dataset(valid_ds, batch_size=32)
    test_ds = shuffle_and_batch_dataset(test_ds, batch_size=32)
    train_ds = map_dataset(train_ds, xy_split)
    valid_ds = map_dataset(valid_ds, xy_split)
    test_ds = map_dataset(test_ds, xy_split)
    if prefetch:
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
        valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return (train_ds, valid_ds, test_ds)
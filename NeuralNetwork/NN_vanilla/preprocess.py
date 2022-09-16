import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler 

# define a global variable minmax_scaler for later to reverse the transformation
minmax_scaler = MinMaxScaler()


def getDatasets(dataframe, scaling = True):
    """
    Returns tuple of scaled train, valid, and test datasets.  

    Args:
        - dataframe: ndarray, dataframe of the data  
        - scaling: boolean, whether to scale the data, default is True
    Output:  
        - (train_ds, valid_ds, test_ds): tuple of datasets which are between 0 and 1 if scaling is True
    
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



def makeArr_BS(df):
    """Returns a numpy array given the pandas dataframe"""
    dataframe_BS = np.vstack((df['strike'].values,
                      df['underlyings_price'].values,
                      df['days_to_maturity'].values,
                      df['volatility'].values,
                      df['rate'].values,
                      df['contract_price'].values)).T  
    return dataframe_BS



def getNormalizedData_BS(df):
    """
    Returns the normalized data using StandardScaler from sklearn.  

    Args:  
        - dataframe: pandas array, dataframe of the data

    Output:  
        - (train_ds, valid_ds, test_ds): tuple of datasets
    """  
    N = len(df)
    # drop the non-numeric columns
    try:
        df = df.drop(['callput'], axis = 1)
        df = df.drop(['date_traded'], axis=1)
    except:
        pass

    df = df.sample(frac=1)
    df_train = df[:int(.8*N)]
    df_val = df[int(.8*N):int(.9*N)]
    df_test = df[int(.9*N):]

    # normalize the data
    normalizer = StandardScaler()
    df_train[df.columns] = normalizer.fit_transform(df_train[df.columns])
    df_val[df.columns] = normalizer.transform(df_val[df.columns]) # transform using the values from the training set
    df_test[df.columns] = normalizer.transform(df_test[df.columns])

    train = makeArr_BS(df_train)
    val = makeArr_BS(df_val)
    test = makeArr_BS(df_test)
    print(train.shape, val.shape, test.shape)

    train_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train, tf.float32))
    valid_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val, tf.float32))
    test_ds  = tf.data.Dataset.from_tensor_slices(tf.cast(test, tf.float32))

    return (train_ds, valid_ds, test_ds)


def getScaledData_BS(df):
    """
    Returns the scaled data using MinMaxScaler from sklearn.  

    Args:  
        - dataframe: pandas array, dataframe of the data

    Output:  
        - (train_ds, valid_ds, test_ds): tuple of datasets
    """   
    N = len(df)
    # drop the non-numeric columns
    try:
        df = df.drop(['callput'], axis = 1)
        df = df.drop(['date_traded'], axis=1)
    except:
        pass

    df = df.sample(frac=1)
    df_train = df[:int(.8*N)]
    df_val = df[int(.8*N):int(.9*N)]
    df_test = df[int(.9*N):]

    # normalize the data
    normalizer = MinMaxScaler()
    df_train[df.columns] = normalizer.fit_transform(df_train[df.columns])
    df_val[df.columns] = normalizer.transform(df_val[df.columns]) # transform using the values from the training set
    df_test[df.columns] = normalizer.transform(df_test[df.columns])

    train = makeArr_BS(df_train)
    val = makeArr_BS(df_val)
    test = makeArr_BS(df_test)
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



def pipeline1(dataframe_BS, prefetch = True, scaling = True, batch_size = 32, shuffle_buffer = 1000):
    """
    Returns a tuple of processed prefetched data for training. 

    Args:
        - dataframe_BS: ndarray, dataframe of the data
        - prefetch: boolean, whether to prefetch the data
        - scaling: boolean, whether to scale the data  
        - batch_size: int, batch size, default 32
        - shuffle_buffer: int, shuffle buffer size, default 1000
        
    Output:
        - (train_ds, valid_ds, test_ds): tuple of datasets
    """  
    train_ds, valid_ds, test_ds = getDatasets(dataframe_BS, scaling)
    train_ds = shuffle_and_batch_dataset(train_ds, batch_size=batch_size, shuffle_buffer=shuffle_buffer)
    valid_ds = shuffle_and_batch_dataset(valid_ds, batch_size=batch_size, shuffle_buffer=shuffle_buffer)
    test_ds = shuffle_and_batch_dataset(test_ds, batch_size=batch_size, shuffle_buffer=shuffle_buffer)
    train_ds = map_dataset(train_ds, xy_split)
    valid_ds = map_dataset(valid_ds, xy_split)
    test_ds = map_dataset(test_ds, xy_split)
    if prefetch:
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
        valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return (train_ds, valid_ds, test_ds)  


def pipeline2(df, prefetch = True, scaling = 'minmax', batch_size = 32, shuffle_buffer = 1000):
    """
    Returns a tuple of processed prefetched data for training. 

    Args:
        - df: pandas dataframe, dataframe of the data
        - prefetch: boolean, whether to prefetch the data
        - scaling: string, either minmax or normalize, default minmax
        - batch_size: int, batch size, default 32
        - shuffle_buffer: int, shuffle buffer size, default 1000
        
    Output:
        - (train_ds, valid_ds, test_ds): tuple of datasets
    """  
    if scaling == 'minmax':
        train_ds, valid_ds, test_ds = getScaledData_BS(df)
    elif scaling == 'normalize':
        train_ds, valid_ds, test_ds = getNormalizedData_BS(df)
    else:
        raise ValueError('scaling must be either minmax or normalize')

    train_ds = shuffle_and_batch_dataset(train_ds, batch_size=batch_size, shuffle_buffer=shuffle_buffer)
    valid_ds = shuffle_and_batch_dataset(valid_ds, batch_size=batch_size, shuffle_buffer=shuffle_buffer)
    test_ds = shuffle_and_batch_dataset(test_ds, batch_size=batch_size, shuffle_buffer=shuffle_buffer)
    train_ds = map_dataset(train_ds, xy_split)
    valid_ds = map_dataset(valid_ds, xy_split)
    test_ds = map_dataset(test_ds, xy_split)
    if prefetch:
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
        valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return (train_ds, valid_ds, test_ds) 
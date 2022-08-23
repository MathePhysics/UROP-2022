import numpy as np
import pandas as pd
import tensorflow as tf

def getDatasets(dataframe):
    """
    Returns tuple of train, valid, and test datasets.
    """
    N = len(dataframe)
    indices = np.random.permutation(N)
    train, val, test = np.split(dataframe[indices], [int(.8*N), int(.9*N)])
    print(train.shape, val.shape, test.shape)
    train_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train, tf.float32))
    valid_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val, tf.float32))
    test_ds  = tf.data.Dataset.from_tensor_slices(tf.cast(test, tf.float32))
    # train_ds = 0
    # valid_ds = 0
    # test_ds  = 0

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

#TODO: add pipeline for preprocessing
def pipeline1():
    pass
    
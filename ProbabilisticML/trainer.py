"""For training and other functions in the notebook."""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from preprocess import *
from models import *
from callbacks import *

def pipeline(dir):
    """Pipeline for training and testing."""
    df = pd.read_csv(os.path.join(dir, 'data/data.csv'))
    df = df.drop(['Unnamed: 0'], axis=1)
    dataframe_BS = makeArr_BS(df)
    train_ds, valid_ds, test_ds = pipeline1(dataframe_BS, scaling=False)
    df.sample(5)
    return train_ds, valid_ds, test_ds

def viewData(dir):
    """View the data."""
    df = pd.read_csv(os.path.join(dir, 'data/data.csv'))
    df = df.drop(['Unnamed: 0'], axis=1)
    return df
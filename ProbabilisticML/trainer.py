"""For training and other functions in the notebook."""

from gc import callbacks
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


def trainBNN(train_ds, valid_ds):
    """Trains the BNN and returns a model."""
    model = getBNN(input_shape=(5,),
                output_shape=(1,),
                num_layers = 3, 
                hidden_units = [300,100,100],
                activation = 'tanh',
                prior = fixedPrior,
                posterior = posterior_mean_field,
                train_size = 128,
                batchnorm = True,
                # dropout = [0.5 for _ in range(num_layers)]
                )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0001, decay_steps = 4000, decay_rate = 0.85, staircase = True)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                    loss=tf.keras.losses.MeanAbsolutePercentageError(),
                    metrics=[tf.keras.metrics.MeanSquaredError()])

    printing = PrintProgress(num_epochs=10)
    history = model.fit(train_ds, epochs=100, validation_data=valid_ds, callbacks=[printing])

    return model


def trainMDN(train_ds, valid_ds):
    """Trains the MDN and returns a model."""  
    model = getMDN(input_shape=(5,),
                num_components=30,     
                output_shape=(1,),
                num_layers = 5, 
                hidden_units = [50,300,300,180,50],
                activation = 'sigmoid',
                regularizer = tf.keras.regularizers.l1_l2(3.16277e-07, 1e-08),
                batchnorm=True
                )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0001, decay_steps = 4000, decay_rate = 0.85, staircase = True)
    
    # perhaps a little change here with loss and metrics
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule), loss = lambda y, model: -model.log_prob(y), 
                metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsolutePercentageError(name="accuracy")])
    printing = PrintProgress(num_epochs=10)
    model.fit(train_ds, epochs=100, validation_data=valid_ds, callbacks=[printing])

    return model

def takeSample(test_ds):
    """Take a sample from the data."""
    for x, y in test_ds.take(1):
        sample = x
        sample_y = y
    
    return sample, sample_y
"""Implements the custom tuner class."""  

import keras_tuner
import numpy as np
import tensorflow as tf 
import tensorflow_probability as tfp

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsolutePercentageError
from tensorflow.keras.metrics import MeanSquaredError


from models import getMDN, getBNN


tfpl = tfp.layers
tfd = tfp.distributions


def tuned_MDN(hp):
    """
    Returns a compiled hyperModel for keras tuner.  

    - Number of layers: 1-5
    - Number of hidden units: 5-7, step 1
    - Learning rate: 1e-4 - 1e-2, log sampling
    - Rate of lr decay: 0.85-0.9995
    - l1_coeff: 1e-8 - 1e-6.5, log sampling
    - l2_coeff: 1e-8 - 1e-6.5, log sampling
    - Loss: 
    - Metrics:
    """  

    # defining a set of hyperparameters for tuning and a range of values for each
    num_layers = hp.Int('num_layers', min_value=1, max_value=5) 

    num_components = hp.Int('num_components', min_value=30, max_value=200, step=10)

    activation = hp.Choice('activation', ['sigmoid', 'tanh'])

    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=0.01, sampling = 'log')
    rate_decay = hp.Float('rate_decay', min_value=0.85, max_value=0.9995)
    l1_reg = hp.Float('l1_coeff', min_value=10**(-8), max_value=10**(-6.5))
    l2_reg = hp.Float('l2_coeff', min_value=10**(-8), max_value=10**(-6.5))
    
    
    hidden_units = []
    for i in range(num_layers):
        hidden_unit = hp.Int(f'units_{i+1}', min_value=5, max_value=300, step=50)
        hidden_units.append(hidden_unit)

    model = getMDN(input_shape=(5,),
                    num_components=num_components,     
                    output_shape=(1,),
                    num_layers = num_layers, 
                    hidden_units = hidden_units,
                    activation = activation,
                    regularizer = tf.keras.regularizers.l1_l2(l1_reg,l2_reg)
                    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate, decay_steps = 4000, decay_rate = rate_decay, staircase = True)
    
    # perhaps a little change here with loss and metrics
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule), 
                loss = lambda y, model: -model.log_prob(y), 
                metrics = [tf.keras.metrics.MeanSquaredError(),
                            tf.keras.metrics.MeanAbsolutePercentageError(name='accuracy')])    

    return model
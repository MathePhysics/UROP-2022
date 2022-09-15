import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1, l2, l1_l2

def model_builder_basket(dim,
                num_layers   = 2,
                hidden_units = [14,7],
                output_shape = (1,),
                activation = 'elu',
                regularizer = None,
                initializer = tf.keras.initializers.he_uniform(),
                final_activation = 'linear',
                dropout = None,
                batchnorm = False
                ): 
    """
    Returns a model for training and testing, used for basket options.  

    Args:
        - dim: int, basket size
        - num_layers: int, number of hidden layers
        - hidden_units: list of number of hidden units in each layer
        - output_shape: shape of the output data
        - activation: string, activation function
        - initializer: initializer for the weights
        - final_activation: string, activation function of final layer
        - dropout: list, dropout rate for each layer, default None
        - batchnorm: bool, specifies if batch normalization is used, default False 
    
    Output:  
        - model: tf.keras.Model, compiled if compile is True
    """  
    assert num_layers == len(hidden_units), "Number of hidden units must match number of layers"
    if dropout is not None:  
        assert num_layers == len(dropout), "Number of dropout rates must match number of layers"

    input_shape = (7 + 2*dim,)

    # define global variables for later use
    # so that the input&output shapes are consistent with build_models used by tuner
    global input_shape_glob
    global dim_glob
    global output_shape_glob

    input_shape_glob = input_shape
    dim_glob = dim
    output_shape_glob = output_shape

    inputs = Input(shape=input_shape)
    h = Flatten()(inputs)

    for i, layer in enumerate(hidden_units):
        h = tf.keras.layers.Dense(layer, activation=activation, kernel_regularizer= regularizer,
                                    kernel_initializer = initializer)(h)
        if dropout:
                h = tf.keras.layers.Dropout(dropout[i])(h)
        if batchnorm:
                h = tf.keras.layers.BatchNormalization()(h)

    if final_activation is not None:
        outputs = tf.keras.layers.Dense(output_shape[0], activation=final_activation,
                                            kernel_initializer = initializer)(h)
    else:
        outputs = tf.keras.layers.Dense(output_shape[0], 
                                            kernel_initializer = initializer)(h)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)  

    return model   


def tuned_model_basket(hp):
    """
    Returns a compiled hyperModel for keras tuner, used for basket options. 

    """  
    # defining a set of hyperparameters for tuning and a range of values for each
    num_layers = hp.Int('num_layers', min_value=1, max_value=5) 
    activation = hp.Choice('activation', ['elu','tanh', 'relu', 'sigmoid'])
    learning_rate = hp.Float('learning_rate', min_value=10**(-3), max_value=0.01)
    rate_decay = hp.Float('rate_decay', min_value=0.85, max_value=0.9995)
    l1_reg = hp.Float('l1_regularizer', min_value=10**(-8), max_value=10**(-6.5))
    l2_reg = hp.Float('l1_regularizer', min_value=10**(-8), max_value=10**(-6.5))
    batchnorm = hp.Boolean(name = 'batchnorm')
        
    hidden_units, dropouts = [],[]
    for i in range(num_layers):
        hidden_unit = hp.Int(f'units_{i+1}', min_value=5, max_value=7)
        hidden_units.append(hidden_unit)
        dropout = hp.Float(f'dropout_{i+1}', min_value=0.0, max_value=0.5, step=0.1)
        dropouts.append(dropout)

    model = model_builder_basket(dim,
                        num_layers = num_layers, 
                        hidden_units = hidden_units,
                        dropout = dropouts,
                        activation = activation,
                        batchnorm = batchnorm,
                        regularizer = tf.keras.regularizers.l1_l2(l1_reg,l2_reg)
                        )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate, decay_steps = 4000, decay_rate = rate_decay, staircase = True)
        
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule), loss = tf.keras.losses.MeanAbsolutePercentageError(), 
                    metrics = [tf.keras.metrics.MeanSquaredError()])

    return model  
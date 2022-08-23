import argparse
import numpy as np
import tensorflow as tf

def getModel(input_shape = (7,),
            hidden_units = [14,7],
            output_shape = (1,),
            activation = 'elu',
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1),
            final_activation = None,
            dropout = None,
            batchnorm = False):
    """
    Returns a model for training and testing.  

    Args:
        - input_shape: shape of the input data
        - hidden_units: list of number of hidden units in each layer
        - output_shape: shape of the output data
        - activation: string, activation function
        - initializer: initializer for the weights
        - final_activation: string, activation function of final layer
        - dropout: list, dropout rate for each layer
        - batchnorm: bool, specifies if batch normalization is used
    
    Output:
        - model: tf.keras.Model
    """  

    inputs = tf.keras.layers.Input(shape=input_shape)
    h = tf.keras.layers.Flatten()(inputs)

    for i, layer in enumerate(hidden_units):
        h = tf.keras.layers.Dense(layer, activation=activation, 
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


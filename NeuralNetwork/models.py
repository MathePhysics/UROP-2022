import argparse
from pyexpat import model
import numpy as np
import tensorflow as tf

def getModel(input_shape = (7,),
            num_layers   = 2,
            hidden_units = [14,7],
            output_shape = (1,),
            activation = 'elu',
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1),
            final_activation = None,
            dropout = None,
            batchnorm = False,
            compile = False,
            optimizer = tf.keras.optimizers.Adam,
            learning_rate = 0.001,
            loss = tf.keras.losses.MeanAbsoluteError(name='loss'),
            metrics = tf.keras.metrics.MeanAbsolutePercentageError(name='accuracy')
            ): 
    """
    Returns a model for training and testing.  

    Args:
        - input_shape: shape of the input data
        - num_layers: int, number of hidden layers
        - hidden_units: list of number of hidden units in each layer
        - output_shape: shape of the output data
        - activation: string, activation function
        - initializer: initializer for the weights
        - final_activation: string, activation function of final layer
        - dropout: list, dropout rate for each layer, default None
        - batchnorm: bool, specifies if batch normalization is used, default False
        - compile: bool, specifies if the model is compiled, default False
        - optimizer: tf.keras.optimizers.Optimizer, optimizer to be used for training
        - learning_rate: float, learning rate for the optimizer, default 0.001
        - loss: tf.keras.losses, loss function to be used for training
        - metrics: tf.keras.metrics, metrics to be used for evaluation  
    
    Output:  
        - model: tf.keras.Model, compiled if compile is True
    """  
    assert num_layers == len(hidden_units), "Number of hidden units must match number of layers"
    if dropout is not None:  
        assert num_layers == len(dropout), "Number of dropout rates must match number of layers"

    # define global variables for later use
    # so that the input&output shapes are consistent with build_models used by tuner
    global input_shape_glob
    global output_shape_glob

    input_shape_glob = input_shape
    output_shape_glob = output_shape

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

    if compile:
        model.compile(optimizer(learning_rate), loss, metrics)

    return model   


def tuneLayer(hp):
    """
    Returns a compiled hyperModel for keras tuner. (this is private)   
    """  

    num_layers = hp.Int('num_layers', min_value=1, max_value=5)  
    hidden_units = []
    for i in range(num_layers):
        hidden_unit = hp.Int(f'units_{i+1}', min_value=5, max_value=50, step=5)
        hidden_units.append(hidden_unit)

    # learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling = 'log')

    model = getModel(input_shape=input_shape_glob,
                    output_shape=output_shape_glob,
                    num_layers = num_layers, 
                    hidden_units = hidden_units,
                    compile = True
                    )

    return model  

def tuneLR(hp):
    """
    Returns a compiled hyperModel for keras tuner. (this is private)   
    """  

    num_layers = hp.Int('num_layers', min_value=1, max_value=5)  
    hidden_units = []
    for i in range(num_layers):
        hidden_unit = hp.Int(f'units_{i+1}', min_value=5, max_value=50, step=5)
        hidden_units.append(hidden_unit)

    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling = 'log')

    model = getModel(input_shape=input_shape_glob,
                    output_shape=output_shape_glob,
                    num_layers = num_layers, 
                    hidden_units = hidden_units,
                    compile = True,
                    learning_rate = learning_rate
                    )

    return model


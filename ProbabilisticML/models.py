import argparse
import numpy as np
import tensorflow as tf  
import tensorflow_probability as tfp  

from distributions import *

tfpl = tfp.layers
tfd = tfp.distributions

def getMDN(input_shape = (7,),
            num_components = 200,
            num_layers   = 2,
            hidden_units = [14,7],
            output_shape = (1,),
            activation = 'elu',
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1),
            regularizer = tf.keras.regularizers.l1_l2(0.000001,0.000001),
            dropout = None,
            batchnorm = False
            ): 
    """
    Returns a mixture density network model for training and testing. 
    Last layer will be a tfpl.MixtureSameFamily layer with a tfd.MultivariateNormalTriL distribution
    or tfpl.IndependentNormal layer with a tfd.Normal distribution.   

    Args:  
        - input_shape: shape of the input data  
        - num_components: number of components in the mixture density network
        - num_layers: int, number of hidden layers
        - hidden_units: list of number of hidden units in each layer
        - output_shape: shape of the output data
        - activation: string, activation function
        - initializer: initializer for the weights
        - regularizer: regularizer for the weights
        - dropout: list, dropout rate for each layer, default None
        - batchnorm: bool, specifies if batch normalization is used, default False 
    
    Output:  
        - model: tf.keras.Model, not compiled
    """  
    assert num_layers == len(hidden_units), "Number of hidden units must match number of layers"
    if dropout is not None:  
        assert num_layers == len(dropout), "Number of dropout rates must match number of layers"

    components = []

    components.append(tf.keras.layers.Input(shape=input_shape))
    components.append(tf.keras.layers.Flatten())

    for i, layer in enumerate(hidden_units):
        components.append(tf.keras.layers.Dense(layer, activation=activation, 
                                  kernel_initializer = initializer,
                                  kernel_regularizer = regularizer))
        if dropout:
            components.append(tf.keras.layers.Dropout(dropout[i]))
        if batchnorm:
            components.append(tf.keras.layers.BatchNormalization())

    params_size = tfpl.MixtureSameFamily.params_size(
                        num_components=num_components,
                        component_params_size=tfpl.IndependentNormal.params_size([output_shape[0]])
                        )

    components.append(tf.keras.layers.Dense(params_size, activation=None,
                                        kernel_initializer = initializer)
                                        )

    components.append(tfpl.MixtureSameFamily(num_components, 
                            tfpl.IndependentNormal([output_shape[0]])
                            )
                        )

    model = tf.keras.Sequential(components)

    return model   


def getBNN(input_shape = (7,),
            num_layers   = 2,
            train_size = 1000,
            hidden_units = [14,7],
            prior = fixedPrior,
            posterior = posterior_mean_field,
            output_shape = (1,),
            activation = 'tanh',
            dropout = None,
            batchnorm = False):

    """
    Returns a Bayesian neural network model for training and testing. 

    Args:  
        - input_shape: shape of the input data  
        - num_layers: int, number of hidden layers
        - train_size: int, number of training samples, needed for weighting KL divergence  
        - hidden_units: list of number of hidden units in each layer
        - prior: prior distribution, default fixedPrior for DenseVariational layer
        - posterior: posterior distribution, default posterior_mean_field for DenseVariational layer
        - output_shape: shape of the output data
        - activation: string, activation function
        - dropout: list, dropout rate for each layer, default None
        - batchnorm: bool, specifies if batch normalization is used, default False 
    
    Output:  
        - model: tf.keras.Model, not compiled
    """    

    assert num_layers == len(hidden_units), "Number of hidden units must match number of layers"
    if dropout is not None:  
        assert num_layers == len(dropout), "Number of dropout rates must match number of layers"


    components = []

    components.append(tf.keras.layers.Input(shape=input_shape))
    components.append(tf.keras.layers.Flatten())

    for i, layer in enumerate(hidden_units):

        components.append(tfpl.DenseVariational(layer,
                            make_prior_fn=prior, 
                            make_posterior_fn=posterior, 
                            kl_weight=1/train_size, 
                            activation=activation
                        ))

        if dropout:
            components.append(tf.keras.layers.Dropout(dropout[i]))
        if batchnorm:
            components.append(tf.keras.layers.BatchNormalization())

    components.append(tfpl.DenseVariational(output_shape[0],
                            make_prior_fn=prior,
                            make_posterior_fn=posterior,
                            kl_weight=1/train_size,
                            activation=None)
                            )

    model = tf.keras.Sequential(components)
    
    return model  


def getMDNEnsemble():
    pass
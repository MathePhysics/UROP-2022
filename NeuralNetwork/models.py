import argparse
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1, l2, l1_l2

def getModel(input_shape = (7,),
            num_layers   = 2,
            hidden_units = [14,7],
            output_shape = (1,),
            activation = 'elu',
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1),
            regularizer = l1_l2(0.000001,0.000001),
            final_activation = None,
            dropout = None,
            batchnorm = False
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
        - regularizer: regularizer for the weights
        - final_activation: string, activation function of final layer
        - dropout: list, dropout rate for each layer, default None
        - batchnorm: bool, specifies if batch normalization is used, default False 
    
    Output:  
        - model: tf.keras.Model, compiled if compile is True
    """  
    assert num_layers == len(hidden_units), "Number of hidden units must match number of layers"
    if dropout is not None:  
        assert num_layers == len(dropout), "Number of dropout rates must match number of layers"

    inputs = Input(shape=input_shape)
    h = Flatten()(inputs)

    for i, layer in enumerate(hidden_units):
        h = Dense(layer, activation=activation, 
                                  kernel_initializer = initializer,
                                  kernel_regularizer = regularizer)(h)
        if dropout:
            h = Dropout(dropout[i])(h)
        if batchnorm:
            h = BatchNormalization()(h)
    if final_activation is not None:
        outputs = Dense(output_shape[0], activation=final_activation,
                                        kernel_initializer = initializer)(h)
    else:
        outputs = Dense(output_shape[0], 
                                        kernel_initializer = initializer)(h)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)  

    return model   



def model_builder_basket(dim,
            num_layers   = 2,
            hidden_units = [14,7],
            output_shape = (1,),
            activation = 'elu',
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1),
            final_activation = None,
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
        h = Dense(layer, activation=activation, 
                                  kernel_initializer = initializer)(h)
        if dropout:
            h = Dropout(dropout[i])(h)
        if batchnorm:
            h = BatchNormalization()(h)
    if final_activation is not None:
        outputs = Dense(output_shape[0], activation=final_activation,
                                        kernel_initializer = initializer)(h)
    else:
        outputs = Dense(output_shape[0], 
                                        kernel_initializer = initializer)(h)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)  

    return model   


def tuned_model_basket(hp):
    """
    Returns a compiled hyperModel for keras tuner, used for basket options. 

    """  
    # defining a set of hyperparameters for tuning and a range of values for each
    num_layers = hp.Int('num_layers', min_value=1, max_value=5) 

    activation = hp.Choice('activation', 
        ['elu','tanh', 'ReLu', 'sigmoid', 'gelu','LeakyReLU']) # do you really need all of these?

    learning_rate = hp.Float('learning_rate', min_value=10**(-3), max_value=0.01)

    rate_decay = hp.Float('rate_decay', min_value=0.85, max_value=0.9995)

    l1_reg = hp.Float('l1_regularizer', min_value=10**(-8), max_value=10**(-6.5))
    l2_reg = hp.Float('l1_regularizer', min_value=10**(-8), max_value=10**(-6.5))

    initializer = hp.Choice('initializer', 
        ['uniform', 'glorot_uniform', 'he_uniform', 'normal']) # do you really need all of these?

    batchnorm = hp.Boolean(name = 'batchnorm')
    
    hidden_units, dropouts = [],[]

    for i in range(num_layers):
        hidden_unit = hp.Int(f'units_{i+1}', min_value=5, max_value=7)
        hidden_units.append(hidden_unit)
        dropout = hp.Float(f'dropout_{i+1}', min_value=0.0, max_value=0.5, step=0.1) # do you really need all of these?
        dropouts.append(dropout)

    model = model_builder_basket(dim=dim_glob,
                    output_shape=output_shape_glob,
                    num_layers = num_layers, 
                    hidden_units = hidden_units,
                    dropout = dropouts,
                    activation = activation,
                    initializer = initializer,
                    batchnorm = batchnorm,
                    regularizer = tf.keras.regularizers.l1_l2(l1_reg,l2_reg)
                    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate, decay_steps = 4000, decay_rate = rate_decay, staircase = True)
    
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule), 
                loss = tf.keras.losses.MeanAbsolutePercentageError(), 
                metrics = [tf.keras.metrics.MeanSquaredError()])

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



def tuned_model(hp):
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

    # https://stats.stackexchange.com/questions/402618/can-sinx-be-used-as-activation-in-deep-learning
    activation = hp.Choice('activation', ['elu', 'tanh'])

    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=0.01, sampling = 'log')
    rate_decay = hp.Float('rate_decay', min_value=0.85, max_value=0.9995)
    l1_reg = hp.Float('l1_coeff', min_value=10**(-8), max_value=10**(-6.5))
    l2_reg = hp.Float('l2_coeff', min_value=10**(-8), max_value=10**(-6.5))
    
    
    hidden_units = []
    for i in range(num_layers):
        hidden_unit = hp.Int(f'units_{i+1}', min_value=5, max_value=7)
        hidden_units.append(hidden_unit)

    model = getModel(input_shape=input_shape_glob,
                    output_shape=output_shape_glob,
                    num_layers = num_layers, 
                    hidden_units = hidden_units,
                    activation = activation,
                    regularizer = tf.keras.regularizers.l1_l2(l1_reg,l2_reg)
                    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate, decay_steps = 4000, decay_rate = rate_decay, staircase = True)
    
    # perhaps a little change here with loss and metrics
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule), loss = tf.keras.losses.MeanAbsolutePercentageError(), 
                metrics = [tf.keras.metrics.MeanSquaredError()])

    return model

class DenseResidualBlock(tf.keras.layers.Layer):
    
    def __init__(self, layer_per_block=[10,10,10], activation='elu', 
                initializer='random_uniform',
                l2reg_coeff=0.01, **kwargs):
        """
        Class initializer for a custom dense residual block with batch normalization.  

        Args:   
            - layer_per_block: number of layers in the block
            - activation: string, activation function to use in the dense layers
            - initializer: tf.keras.initializers, initializer to use in the dense layer
            - l2reg_coeff: coefficient for L2 regularization
            - **kwargs: keyword arguments for the parent class  
        """  

        super(DenseResidualBlock, self).__init__(**kwargs)

        self.l2reg_coeff = l2reg_coeff
        self.activation = activation
        self.layer_per_block = layer_per_block
        self.initializer = initializer


    def build(self, input_shape):
        for i in range(len(self.layer_per_block)):
            setattr(self, 'dense_{}'.format(i+1), Dense(self.layer_per_block[i],
                                                    activation=self.activation,
                                                    kernel_initializer=self.initializer,
                                                    kernel_regularizer=l2(self.l2reg_coeff),
                                                    input_shape = input_shape,
                                                    name='dense_{}'.format(i+1)))
                                                    
            setattr(self, 'bn_{}'.format(i+1), BatchNormalization(name='bn_{}'.format(i+1)))


    def call(self, inputs, training=False):
        h = inputs  
        for i in range(len(self.layer_per_block)):
            h = getattr(self, 'dense_{}'.format(i+1))(h)
            h = getattr(self, 'bn_{}'.format(i+1))(h, training=training)
        return h + inputs


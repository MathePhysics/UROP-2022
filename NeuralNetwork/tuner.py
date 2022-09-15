"""Implements the custom tuner class."""  

import keras_tuner
import numpy as np
import tensorflow as tf 

from models import getModel


##################################################################################################
# def tuned_model_basket(hp):
#     """
#     Returns a compiled hyperModel for keras tuner, used for basket options. 

#     """  
#     # defining a set of hyperparameters for tuning and a range of values for each
#     num_layers = hp.Int('num_layers', min_value=1, max_value=5) 

#     activation = hp.Choice('activation', 
#         ['elu','tanh', 'ReLu', 'sigmoid', 'gelu','LeakyReLU']) # do you really need all of these?

#     learning_rate = hp.Float('learning_rate', min_value=10**(-3), max_value=0.01)

#     rate_decay = hp.Float('rate_decay', min_value=0.85, max_value=0.9995)

#     l1_reg = hp.Float('l1_regularizer', min_value=10**(-8), max_value=10**(-6.5))
#     l2_reg = hp.Float('l1_regularizer', min_value=10**(-8), max_value=10**(-6.5))

#     initializer = hp.Choice('initializer', 
#         ['uniform', 'glorot_uniform', 'he_uniform', 'normal']) # do you really need all of these?

#     batchnorm = hp.Boolean(name = 'batchnorm')
    
#     hidden_units, dropouts = [],[]

#     for i in range(num_layers):
#         hidden_unit = hp.Int(f'units_{i+1}', min_value=5, max_value=7)
#         hidden_units.append(hidden_unit)
#         dropout = hp.Float(f'dropout_{i+1}', min_value=0.0, max_value=0.5, step=0.1) # do you really need all of these?
#         dropouts.append(dropout)

#     model = model_builder_basket(dim=dim_glob,
#                     output_shape=output_shape_glob,
#                     num_layers = num_layers, 
#                     hidden_units = hidden_units,
#                     dropout = dropouts,
#                     activation = activation,
#                     initializer = initializer,
#                     batchnorm = batchnorm,
#                     regularizer = tf.keras.regularizers.l1_l2(l1_reg,l2_reg)
#                     )

#     lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#         learning_rate, decay_steps = 4000, decay_rate = rate_decay, staircase = True)
    
#     model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule), 
#                 loss = tf.keras.losses.MeanAbsolutePercentageError(), 
#                 metrics = [tf.keras.metrics.MeanSquaredError()])

#     return model   


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

    # activation = hp.Choice('activation', ['elu', 'tanh'])

    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=0.01, sampling = 'log')
    rate_decay = hp.Float('rate_decay', min_value=0.85, max_value=0.9995)
    l1_reg = hp.Float('l1_coeff', min_value=10**(-8), max_value=10**(-6.5))
    l2_reg = hp.Float('l2_coeff', min_value=10**(-8), max_value=10**(-6.5))
    
    
    hidden_units = []
    for i in range(num_layers):
        hidden_unit = hp.Int(f'units_{i+1}', min_value=50, max_value=500, step=50)
        hidden_units.append(hidden_unit)

    model = getModel(input_shape=input_shape_glob,
                    output_shape=output_shape_glob,
                    num_layers = num_layers, 
                    hidden_units = hidden_units,
                    activation = 'elu',
                    regularizer = tf.keras.regularizers.l1_l2(l1_reg,l2_reg)
                    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate, decay_steps = 4000, decay_rate = rate_decay, staircase = True)
    
    # perhaps a little change here with loss and metrics
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule), loss = tf.keras.losses.MeanAbsolutePercentageError(name='loss'), 
                metrics = [tf.keras.metrics.MeanSquaredError(name='accuracy')])

    return model

##################################################################################################
############################################################################################################
def tuneSine(hp):
    """
    Returns a compiled hyperModel for keras tuner.  The input shape is limited to (5,) and out shape to (1,)

    - Number of layers: 1-5
    - Number of hidden units: 5-7, step 1
    - Learning rate: 1e-4 - 1e-2, log sampling
    - Rate of lr decay: 0.85-0.9995
    - l1_coeff: 1e-8 - 1e-6.5, log sampling
    - l2_coeff: 1e-8 - 1e-6.5, log sampling
    """  

    # defining a set of hyperparameters for tuning and a range of values for each
    num_layers = hp.Int('num_layers', min_value=1, max_value=5) 

    # https://stats.stackexchange.com/questions/402618/can-sinx-be-used-as-activation-in-deep-learning

    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=0.01, sampling = 'log')
    rate_decay = hp.Float('rate_decay', min_value=0.85, max_value=0.9995)
    l1_reg = hp.Float('l1_coeff', min_value=10**(-8), max_value=10**(-6.5))
    l2_reg = hp.Float('l2_coeff', min_value=10**(-8), max_value=10**(-6.5))
    
    list_of_layers = []

    for i in range(num_layers):
        hidden_unit = hp.Int(f'units_{i+1}', min_value=50, max_value=300, step=50)
        list_of_layers.append(tf.keras.layers.Dense(hidden_unit, kernel_regularizer = tf.keras.regularizers.l1_l2(l1_reg,l2_reg)))
        list_of_layers.append(tf.keras.layers.Lambda(lambda x: tf.math.sin(x)))

    list_of_layers.append(tf.keras.layers.Dense(1))

    model = tf.keras.Sequential(list_of_layers)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate, decay_steps = 4000, decay_rate = rate_decay, staircase = True)
    
    # perhaps a little change here with loss and metrics
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule), loss = tf.keras.losses.MeanAbsolutePercentageError(name='loss'), 
                metrics = [tf.keras.metrics.MeanSquaredError(name='accuracy')])

    return model


##################################################################################################


############################################################################################################
# for the moment being, we only use the RandomSearch tuner, 
# as other tuners need more thorough explanation
# Namely, we need to describe how BayesianOptimization and Hyperband work under the hood

class customTuner(keras_tuner.RandomSearch):

    def __init__(self, input_shape, output_shape, dim=None, basket=False, **kwargs):
        """
        Initializes the custom tuner class.    
        
        Args:  
            - input_shape: the shape of the input data
            - output_shape: the shape of the output data
            - dim: int, the dimension of basket, default None
            - basket: bool, whether to use basket model or not, default False
        """  
        global input_shape_glob
        global dim_glob
        global output_shape_glob
        global basket_glob

        super(customTuner, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dim = dim  
        self.basket = basket

        input_shape_glob = input_shape
        dim_glob = dim
        output_shape_glob = output_shape
        basket_glob = basket

    def run_trial(self, trial, train_ds, valid_ds, epochs, **kwargs):
        # overrides the run_trial method of the RandomSearch class
        # should return the result of model.fit()
        hp = trial.hyperparameters  

        # if self.basket:
        #     assert self.dim is not None, "Basket dimension must be specified"
        #     compiled_model = tuned_model_basket(hp)
        #     history = compiled_model.fit(train_ds, validation_data=valid_ds, epochs=epochs, **kwargs)
        #     return  history
        # else:
        compiled_model = tuned_model(hp)
        history = compiled_model.fit(train_ds, validation_data=valid_ds, epochs=epochs, **kwargs)
        return  history



# define a hypermodel subclass

class customHyperModel(keras_tuner.HyperModel):

  def build(self, hp):
    # this should return a compiled model
    # if basket_glob:
    #     return tuned_model_basket(hp)
    # else:
    return tuned_model(hp)
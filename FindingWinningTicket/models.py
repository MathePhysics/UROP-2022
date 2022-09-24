'''Initialize custom model for training and pruning.'''

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds  



'''Initialize custom model for training and pruning.'''

class maskInit(tf.keras.initializers.Initializer):

  def __init__(self, mask = None, 
               preinit_weights = None,
               initializer= tf.keras.initializers.HeNormal(),
               **kwargs):
    '''
    Returns custom weight initializer with mask.  

    Parameters:  
      - mask: ndarray of a single mask for one layer
      - preinit_weights: weights
      - initializer: tf.keras.initializers object
    '''

    self.mask = mask
    self.initializer = initializer
    self.preinit_weights = preinit_weights

  def __call__(self, shape, dtype = None):

    out = None

    if self.preinit_weights is not None and self.mask is not None:
        out = tf.math.multiply(self.mask, self.preinit_weights)

    elif self.preinit_weights is None and self.mask is not None: 

        out = tf.math.multiply(self.mask, self.initializer(shape)) 

    elif self.preinit_weights is None and self.mask is None:
        # first initialization of model without any weights/masks
        out = self.initializer(shape)

    assert out is not None

    return out  


def makeFC(preinit_weights = None, masks = None,
              input_shape = 784,
              output_shape = 10,
              layers = [300, 100],
              activation = 'relu',
              initializer = tf.keras.initializers.random_uniform(),
              final_activation = None,
              BatchNorm = False,
              Dropout = None,
              optimizer = tf.keras.optimizers.Adam(0.001),
              loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.SparseCategoricalAccuracy()
             ):
    '''
    Returns a model for pruning.   

    Args:    
        - preinit_weights: dictionary of ndarray, the initialized weights, default to None,
                None when first initializing  
        - masks: dictionary of ndarray, each element is a mask for all the weights  
        - input_shape: tuple, shape of input data
        - output_shape: tuple, shape of output data
        - layers: list of integers, specifying nodes in hidden layers, 
                [layer1, layer2, ..., layerN]
        - activation: string, activation function for hidden layers  
        - initializer: string, initializer for weights, default to random_uniform
        - final_activation: string, activation function for output layer
        - BatchNorm: boolean, whether to use batch normalization
        - Dropout: list of floats, dropout rate for hidden layers only
        - optimizer: optimizer
        - loss_fn: loss function for the nn
        - metrics: metrics for evaluation  

    Returns:
        - model: tf.keras.Sequential model that can be pruned later  
    '''  

    model = tf.keras.Sequential(name = "ModeltoPrune")
    model.add(tf.keras.layers.Flatten(input_shape = input_shape))
    # model.add(tf.keras.layers.InputLayer(input_shape = input_shape))
        
    num_layer = len(layers)  
    mask_list = []
    mask_keys = []

    if masks is not None:
        for key in masks.keys():
            if 'kernel' in key:
                mask_keys.append(key)
        mask_list = [masks[key] for key in mask_keys]
    
    preinit_weight_list = []

    if preinit_weights is not None:
        for key in preinit_weights.keys():
            if 'kernel' in key:
                preinit_weight_list.append(preinit_weights[key])


    for i in range(num_layer):
        if masks is None:
            mask = None
        else:
            mask = mask_list[i]

        if preinit_weights is None:
            preinit_weight = None
        else:
            preinit_weight = preinit_weight_list[i]

        model.add(tf.keras.layers.Dense(layers[i], activation = activation,
                    kernel_initializer = maskInit(mask = mask, preinit_weights = preinit_weight, initializer=initializer)))
        if BatchNorm:
            model.add(tf.keras.layers.BatchNormalization())
        if Dropout is not None:
            model.add(tf.keras.layers.Dropout(Dropout[i]))

    if masks is None:
        last_mask = None
    else:
        last_mask = mask_list[-1]

    if preinit_weights is None:
        last_preinit_weight = None
    else:
        last_preinit_weight = preinit_weight_list[-1]
        
    if final_activation is not None:
        model.add(tf.keras.layers.Dense(output_shape, activation = final_activation,
                    kernel_initializer = maskInit(mask = last_mask, preinit_weights = last_preinit_weight)))

    else:
        model.add(tf.keras.layers.Dense(output_shape,
                    kernel_initializer = maskInit(mask = last_mask, preinit_weights = last_preinit_weight)))

    # if BatchNorm and Dropout is None:
    #     for i in range(num_layer):
    #         if masks is None:
    #             mask = None
    #         else:
    #             mask = mask_list[i]

    #         if preinit_weights is None:
    #             preinit_weight = None
    #         else:
    #             preinit_weight =preinit_weight_list[i]

    #         if layers[0]==300 and i==num_layer-1:
    #           model.add(tf.keras.layers.Dense(layers[i], 
    #           kernel_initializer=maskInit(mask=mask, preinit_weights = preinit_weight)))
    #           model.add(tf.keras.layers.BatchNormalization())
    #         else:
    #           model.add(tf.keras.layers.Dense(layers[i], 
    #           activation=activation,
    #           kernel_initializer=maskInit(mask=mask, preinit_weights = preinit_weight)))
    #           model.add(tf.keras.layers.BatchNormalization())

    # if BatchNorm and Dropout is not None:
    #     for i in range(num_layer):
    #         if masks is None:
    #             mask = None
    #         else:
    #             # the masks in pruning function includes the biases
    #             # which are np.ones(shape of bias)
    #             mask = mask_list[i]
    #         if preinit_weights is None:
    #             preinit_weight = None
    #         else:
    #             preinit_weight =preinit_weight_list[i]
            
    #         dropout = Dropout[i]
            
    #         if layers[0]==300 and i==num_layer-1:
    #           model.add(tf.keras.layers.Dense(layers[i], 
    #           kernel_initializer=maskInit(mask=mask, preinit_weights = preinit_weight)))
    #           model.add(tf.keras.layers.BatchNormalization())
    #           model.add(tf.keras.layers.Dropout(dropout))
    #         else:
    #           model.add(tf.keras.layers.Dense(layers[i], 
    #           activation=activation,
    #           kernel_initializer=maskInit(mask=mask, preinit_weights = preinit_weight)))
    #           model.add(tf.keras.layers.BatchNormalization())
    #           model.add(tf.keras.layers.Dropout(dropout))

    # if not BatchNorm and Dropout is not None:
    #     for i in range(num_layer):
    #         if masks is None:
    #             mask = None
    #         else:
    #             # the masks in pruning function includes the biases
    #             # which are np.ones(shape of bias)
    #             mask = mask_list[i]
    #         if preinit_weights is None:
    #             preinit_weight = None
    #         else:
    #             preinit_weight =preinit_weight_list[i]
                
    #         dropout = Dropout[i]

    #         if layers[0]==300 and i==num_layer-1:
    #           model.add(tf.keras.layers.Dense(layers[i], 
    #           kernel_initializer=maskInit(mask=mask, preinit_weights = preinit_weight)))
    #           model.add(tf.keras.layers.Dropout(dropout))
    #         else:
    #           model.add(tf.keras.layers.Dense(layers[i], 
    #           activation=activation,
    #           kernel_initializer=maskInit(mask=mask, preinit_weights = preinit_weight)))
    #           model.add(tf.keras.layers.Dropout(dropout))

    # if not BatchNorm and Dropout is None:
    #     for i in range(num_layer):
    #         if masks is None:
    #             mask = None
    #         else:
    #             # the masks in pruning function includes the biases
    #             # which are np.ones(shape of bias)
    #             mask = mask_list[i]
    #         if preinit_weights is None:
    #             preinit_weight = None
    #         else:
    #             preinit_weight =preinit_weight_list[i]
                
    #         if layers[0]==300 and i==num_layer-1:
    #           model.add(tf.keras.layers.Dense(layers[i], 
    #           kernel_initializer=maskInit(mask=mask, preinit_weights = preinit_weight)))

    #         else:
    #           model.add(tf.keras.layers.Dense(layers[i], 
    #           activation=activation,
    #           kernel_initializer=maskInit(mask=mask, preinit_weights = preinit_weight)))


    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=metrics)
    # model.summary()

    return model

class customLinear(tf.keras.layers.Layer):  

    def __init__(self, num_out,
                activation = 'relu',
                initializer = tf.keras.initializers.HeNormal(),
                BatchNorm=None, 
                Dropout=None,
                mask=None,
                preinit_weights=None,
                **kwargs):
        '''
        Returns a custom linear layer.

        Parameters:
            - num_out: number of output nodes
            - activation: activation function
            - initializer: initializer of layer, default to kaiming
            - BatchNorm: bool, whether to use batch normalization
            - Dropout: float, dropout rate
            - mask: mask covering weights for pruning
            - preinit_weights: weights before training
        '''
        super(customLinear, self).__init__()
        self.num_out = num_out
        self.activation = activation
        self.BatchNorm = BatchNorm
        self.Dropout = Dropout
        self.initializer = initializer
        self.mask = mask
        self.preinit_weights = preinit_weights

    def call(self, inputs):
        out = tf.keras.layers.Dense(self.num_out,
                                    activation=self.activation,
                                    kernel_initializer=maskInit(self.mask,self.preinit_weights,
                                    self.initializer))(inputs)
        if self.BatchNorm is not None:
            out = tf.keras.layers.BatchNormalization()(out)
        if self.Dropout is not None:
            out = tf.keras.layers.Dropout(self.Dropout)(out)

        return out
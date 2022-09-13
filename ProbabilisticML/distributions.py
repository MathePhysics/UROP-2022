import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfpl = tfp.layers
tfd = tfp.distributions


def trainablePrior(kernel_size, bias_size, dtype=None):
    """Create a trainable prior distribution."""
    n = kernel_size + bias_size
    prior_model = tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(2*n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfp.distributions.Normal(loc=tf.math.sigmoid(t[...,0]), scale=tf.math.exp(t[...,1])),
                )
            )
        ]
    )
    return prior_model

def fixedPrior(kernel_size, bias_size, dtype = None):
    """Create a fixed prior distribution."""
    n = kernel_size + bias_size
    prior_model = tf.keras.Sequential([
        tfp.layers.DistributionLambda(  
            lambda t: tfd.MultivariateNormalDiag(loc = tf.zeros(n)  ,  scale_diag = tf.ones(n)
                                                
            ))
        
    ])
    
    return prior_model  


def posterior(kernel_size, bias_size, dtype=None):
    """Create a trainable posterior distribution."""
    n = kernel_size + bias_size
    posterior_model = tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype 
            ),
            tfp.layers.MultivariateNormalTriL(n)
        ]
    )
    return posterior_model

def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    """Create a mean field normal posterior distribution."""
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                        scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1))
    ])
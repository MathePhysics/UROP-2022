'''Auxiliary functions.'''

import numpy as np
import tensorflow as tf  

def customPruneFC(model, prune_perc = 0.5):
  '''Returns a mask covering weights of lower magnitudes.''' 
  masks = [] 
  for layer in model.trainable_weights:
    # ignore the bias
    weight = layer.numpy()
    if len(weight.shape)==2:
      perc = np.percentile(weight, prune_perc * 100)
      mask = np.array(weight>perc, dtype = 'float32')
      # print(tf.math.count_nonzero(mask)/len(mask.flatten()))
      masks.append(mask)
    else:
      masks.append(np.ones_like(weight))
  return masks   


def getInitWeight(prune_model):
  '''Returns a list of initialized weights from model.'''
  prune_weight_list = prune_model.get_weights()
  init_weight_list = []
  for weight in prune_weight_list:
    if len(weight.shape)!=0:
      init_weight_list.append(weight)
  return init_weight_list  


def numParam(model):
  '''Returns number of nonzero parameters in a model.'''
  num_params_after = 0
  for layer in model.trainable_weights:
    if len(layer.numpy().shape)>1:
      num_params_after += tf.math.count_nonzero(layer).numpy()
  print(f"\n \n After pruning, the total no. of nonzero weights is: {num_params_after} \n \n")  


def test_masks(model):
  """Returns mask as all ones for testing."""
  fake_mask = []
  for layer in model.trainable_weights:
    fake_mask.append(np.ones_like(layer.numpy()))
  return fake_mask



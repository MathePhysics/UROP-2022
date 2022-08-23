'''Auxiliary functions.'''

import numpy as np
import tensorflow as tf  

def customPruneFC(model, prune_perc = 0.5):
  '''
  Returns a dict of masks covering weights of lower magnitudes.  

  Args:  
    - model: tf.keras.model for pruning  
    - prune_perc: float, percentage of weights to remove,
                  namely the weights that are of percentile below this
                  will be removed
  
  Output:
    -  masks: a dictionary of ndarrays, mask for all trainable variables,
              for the kernel (weight matrices) this is the normal mask,
              for all other trainable variables, this is all ones
  ''' 
  masks = {} 
  for layer in model.trainable_weights:
    # ignore other trainable variables
    layer_name = layer.name
    if 'kernel' in layer_name:
      weight = layer.numpy()
      perc = np.percentile(weight, prune_perc * 100)
      mask = np.array(weight>perc, dtype = 'float32')
      # print(tf.math.count_nonzero(mask)/len(mask.flatten()))
      masks[f'{layer_name}'] = mask
    else:
      masks[f'{layer_name}'] = np.ones_like(layer.numpy().shape)
  return masks   


def getInitWeight(prune_model):
  '''Returns a dictionary of initialized weights from model.'''
  init_weight_dict = {}
  for layer in prune_model.trainable_weights:
    init_weight_dict[layer.name] = layer.numpy()
  return init_weight_dict 


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


def earlyStop(hist_metric, patience = 5, use_loss = True):

  """
  Returns a boolean value for stopping training.  
  
  Args:
    - hist_metric: list, historical metrics for determining early stop
    - patience: int, no. of epochs to wait before stopping, default 5   
    - use_loss: bool, determines if the metric should be loss  

  Output:  
    - flag: bool, determines if training should stop  
  """    
  N = len(hist_metric)

  if N <= 2*patience:
    return False  

  #TODO: implement a better way to capture increase in loss  

  counter = 0
  if use_loss:
    for i in range(N-1, 0, -1):
      if hist_metric[i]>=hist_metric[i-1]:
        counter+=1
      if counter>patience:
        return True
  else:
    for i in range(N-1, 0, -1):
      if hist_metric[i]<=hist_metric[i-1]:
        counter+=1
      if counter>patience:
        return True
  
  return False

  






import time
import numpy as np
import pandas as pd


# define global variables
dims = [1, 4, 7, 10, 13, 16]


def get_dfs(dims, num_samples):
    """
    Return a list of pd.DataFrame objects containing option information for all dimensions.

    Args:
    - dims (list of int): a list of basket sizes to be considered
    - num_samples (int): the number of samples to be considered

    """
    dataframes_list = []
    for dim in dims:
      temp_df = pd.read_csv(f"/content/drive/MyDrive/data/{num_samples}_basket_data_{dim}.csv")
      temp_df = temp_df.drop(['Unnamed: 0'], axis=1)

      # move column with contract prices to the end
      cols = list(temp_df.columns.values) 
      cols.pop(cols.index('contract_price')) 
      temp_df = temp_df[cols+['contract_price']] 

      dataframes_list.append(temp_df)
    return dataframes_list


def using_tuner(i, train_datasets, valid_datasets, tuners):
    """
    Return the best combination of hyperparamters and the best models for given dimension.

    Args:
    - i (int): the dimension to be consider
    - train_datasets (list of tf.data.Dataset): a list of train datasets for all dimensions
    - valid_datasets (list of tf.data.Dataset): a list of validation datasets for all dimensions
    - tuners (list): a list of tuners to be used
    
    """

    tuners[i].search(train_datasets[i], epochs = 5, validation_data = valid_datasets[i])    

    best_hp_list = tuners[i].get_best_hyperparameters(1)[0]
    best_model = tuners[i].get_best_models(num_models=1)[0]

    return best_hp_list, best_model

def train_models(dims, train_datasets, valid_datasets, best_models):
    """
    Fits the list of best models for all dimensions. 
    Returns list of training histories and list of trainning time for all dimensions.

    Args:
    - dims (list of int): a list of basket sizes to be considered
    - train_datasets (list of tf.data.Dataset): a list of train datasets for all dimensions
    - valid_datasets (list of tf.data.Dataset): a list of validation datasets for all dimensions
    - best_models (list of tf.keras.Model): a list of best models for all dimensions
    """
    begin_train, end_train = [0]*len(dims),[0]*len(dims)
    best_models_history = []

    for i, _ in enumerate(dims):
        begin_train[i] = time.time()
        history = best_models[i].fit(train_datasets[i], epochs = 10, validation_data = valid_datasets[i])
        end_train[i] = time.time()
        best_models_history.append(history)
    
    train_time = np.array(end_train) - np.array(begin_train)
    
    return best_models_history, train_time


def evaluate_models(dims, test_datasets, best_models):
    """
    Returns results of evaluation and a list of test time for all dimension.

    Args:
    - dims (list of int): a list of basket sizes to be considered
    - test_datasets (list of tf.data.Dataset): a list of test datasets for all dimensions
    - best_models (list of tf.keras.Model): a list of best models for all dimensions

    """
    
    begin_test, end_test = [0]*len(dims),[0]*len(dims)
    results = []
    for i, _ in enumerate(dims):
        begin_test[i] = time.time()
        result = best_models[i].evaluate(test_datasets[i])
        end_test[i] = time.time()
        results.append(result) 
        
    test_time = np.array(end_test) - np.array(begin_test)

    return results, test_time


def get_best_mse_df(dims,best_models_history):
    """
    Returns a pd.DataFrame containing best validation MSE for all dimensions.

    Args:
    - dims (list of int): a list of basket sizes to be considered
    - best_models_history (list of tf.keras.callbacks.History): list of training histories for all dimensions

    """
    return pd.DataFrame([min(best_models_history[i].history['val_mean_squared_error']) for i in range(len(dims))], index = dims, columns = ['Best Validation MSE'])


def evaluate_models(best_models_rn, best_models_hb, best_models_bo, test_datasets):
    """
    Returns the test results for all dimensions with chosen tuner, and test time.

    Args:
        best_models_rn (list): a list of models with best hyperparameters obtained from random tuner
        best_models_hb (list): a list of models with best hyperparameters obtained from hyperband tuner
        best_models_bo (list): a list of models with best hyperparameters obtained from Bayesian optimization
        test_datasets (list): a list of test datasets for all dimensions

    """
    begin_test, end_test = [0]*len(dims),[0]*len(dims)
    results = []
    
    for i, _ in enumerate(dims):
        if i in [1,7,13]:
            begin_test[i] = time.time()
            result = best_models_rn[i].evaluate(test_datasets[i])
            end_test[i] = time.time()
            results.append(result) 
        elif i in [16]:
            begin_test[i] = time.time()
            result = best_models_hb[i].evaluate(test_datasets[i])
            end_test[i] = time.time()
            results.append(result) 
        else:
            begin_test[i] = time.time()
            result = best_models_bo[i].evaluate(test_datasets[i])
            end_test[i] = time.time()
            results.append(result) 
    
    test_time = np.array(end_test)-np.array(begin_test)
    
    return results, test_time


def get_train_time(begin_train_rn,begin_train_hb,begin_train_bo, end_train_rn, end_train_hb, end_train_bo):
    """
    Return a list of train time for all dimensions with chosen tuner.

    Args:
        begin_train_rn (list): the list of train time for random tuner
        begin_train_hb (list): the list of train time for hyperband tuner
        begin_train_bo (list): the list of train time for Bayesian optimization
        end_train_rn (list): the list of test time for random tuner
        end_train_hb (list): the list of test time for hyperband tuner
        end_train_bo (list): the list of test time for Bayesian optimization
        
    """
    train_time = []
    for i in range(len(dims)):
        if i in [4,13]:
            train_time.append(np.array(end_train_rn[i])-np.array(begin_train_rn[i]))
        elif i in [1, 7]:
            train_time.append(np.array(end_train_hb[i])-np.array(begin_train_hb[i]))
        else:
            train_time.append(np.array(end_train_bo[i])-np.array(begin_train_bo[i]))
    
    return train_time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def propocessed(data, shuffle = True):
    """
    Returns tuples of normalized train and test data.  
    
    Args:
        - dataframe: ndarray, dataframe of the data  
        - shuffle: boolean, whether to shuffle the dataframe
    
    Output:  
        - (x_train, y_train) , (x_test, y_test): tuple of arrays which are between 0 and 1
    
    """

    # separate features and targets
    x_data, y_data = data[:,:-1], data[:,-1]

    # perform train and test set split with optional shuffle
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data ,test_size = 0.2, shuffle= shuffle)

    # standardize the dataset
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    return (x_train, y_train) , (x_test, y_test)  


def makeBS(df):
    """
    Extract inputs to BS model from the dataframe passed in
    
    Arg:
        df: pd.DataFrame, contains all data read from data.csv

    Output:
        np.array: contains inputs to BS model 
    """
    
    dataframe_BS = np.vstack((df['strike'].values,
                      df['underlyings_price'].values,
                      df['days_to_maturity'].values,
                      df['volatility'].values,
                      df['rate'].values,
                      df['contract_price'].values)).T  
    return dataframe_BS
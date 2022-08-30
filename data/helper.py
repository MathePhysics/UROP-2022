import sys
import pandas as pd
import numpy as np

def get_data(db, statement, date_col):
    """
    Use SQL to get data and store in pandas DataFrames.
    Standardise the format of dates to pd.Timestamp.  

    Args:
    - db: database connection
    - statement: SQL statement to get data
    - date_col: name of column containing dates  

    Output:
    - res: pandas DataFrame with standardised dates
    """
    res = db.raw_sql(statement)
    res[f'{date_col}'] = res[f'{date_col}'].apply(lambda x: pd.Timestamp(x))
    return res    


def contract_ids(capEforEuro, newYearDay, newYearEve):
    """
    Get all call options info whose start date is between newYearDay and newYearEve from option_history file.  

    Args:  
        - capEforEuro: cap E for Euro
        - newYearDay: new year day
        - newYearEve: new year eve
    
    Output:
        - df: pandas DataFrame with all call options info whose start date is between newYearDay and newYearEve

    """
    contract_statement = f"""SELECT optionid, securityid, strike/100000 as strike, expiration, callput
                            FROM optionm.option_history
                            WHERE exercisestyle = '{capEforEuro}' 
                                AND startdate BETWEEN '{newYearDay}' AND '{newYearEve}'
                                AND callput = 'C'
                            ORDER BY optionid ASC"""
    return get_data(contract_statement, 'expiration')  

def option_prices(year, contracts):
    """
    Get option prices (and other info) in contracts (contracts contain all options info which starts in one certain year). 
    We need contracts because `option_price_{year}` contains info of options that may not start at that year.  

    Args:
        - year: year
        - contracts: pandas DataFrame with all call options info whose start date is between newYearDay and newYearEve
    
    Output:
        - df: pandas DataFrame with option prices (and other info) in contracts

    """
    option_statement = f"""SELECT optionid, date as date_traded, last/100 AS contract_price, 
                            underlyinglast/100 AS underlyings_price, volume AS contract_volume
                            FROM optionm.option_price_{year}
                            WHERE optionid in {tuple(contracts['optionid'])}
                            AND volume > 2
                            AND last > 5
                            AND last < 40
                            AND underlyinglast > 0.5
                            AND specialsettlement = 0
                            ORDER BY optionid, date"""
    return get_data(option_statement, 'date_traded')  

def combine_option_info(contracts, all_prices, combined):
    """
    Create a pd.DataFrame that combines pd.DataFrame returned from contract_ids with that returned from option_prices (info in one year).
    Order by optionid and label axis with indices starting from 0.
    Add a new column to record days_to_maturity.
    Only rows with 1 < days_to_maturity < 366 are retained.
    Remove the column recording expiration.
    Return this one-year info after combining it with combined (previous years info).

    """
    combined_1yr = pd.merge(contracts, all_prices, how = "right", on = "optionid")
    combined_1yr = combined_1yr.sort_values(by = ['optionid'], ignore_index = True)
    combined_1yr['days_to_maturity'] = (combined_1yr['expiration']- combined_1yr['date_traded'])/np.timedelta64(1,'D')  # by dividing this we can get float (num of days)
    combined_1yr = combined_1yr[(1< combined_1yr['days_to_maturity']) & (combined_1yr['days_to_maturity'] < 366)]
    combined_1yr.drop(columns = ['expiration'], inplace = True)
    return pd.concat([combined, combined_1yr], axis = 0, ignore_index = True)

def toosmall(vec):
    """
    Quotes traded on days on which there were insuffcient interest rate values (4 are required) available to perform cubic spline were discarded.
    This function exits the script if all dates are not available.

    """
    if int(np.sum(vec)) == 0:
        print('Number too small - not enough quotes for splicing rates')
        print('Exiting script.')
        sys.exit()


# Imports
import config
import pymysql

import pandas as pd
import numpy as np
from scipy.stats import expon
from scipy.stats import randint as sp_randint
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor


def getDbConnection():
    """
    Open connection to the database
    """

    return pymysql.connect(host=config.mysql['host'],
                           port=config.mysql['port'],
                           user=config.mysql['user'],
                           passwd=config.mysql['passwd'],
                           db=config.mysql['db'],
                           charset='utf8',
                           autocommit=True,
                           cursorclass=pymysql.cursors.DictCursor)


def getPrices(hour, min_market_cap=1E9):
    """
    Generate a dataframe of the top coins

    Input:
        hour (int): number of hours for calculation
        min_market_cap (float): minimum market cap

    Output:
        Dataframe of prices of top coins over duration, hr
    """

    # Obtain observations from database
    conn = getDbConnection()
    cur = conn.cursor()

    # Get list of coins that were over the minimum market cap at any point in the past hr hours
    earliest_date = (pd.Timestamp.utcnow() - pd.Timedelta(f'+{hour}:00:00')).strftime('%Y-%m-%d %H:%M:%S')
    sql = f"SELECT DISTINCT symbol FROM coinmarketcap WHERE timestamp >= '{earliest_date}' " + \
          f"AND market_cap_USD >= '{min_market_cap}'"  # + " AND symbol IN ('ETH', 'BTC', 'LTC')"
    cur.execute(sql)
    symbols = cur.fetchall()
    symbol_str = "(" + ','.join(["'" + s['symbol'] + "'" for s in symbols]) + ")"
    # E.g. "('BTC','ETH','XRP','BCH','ADA','LTC','XEM','NEO')"

    # Get price history for those coins
    sql = f"SELECT timestamp, symbol, name, price_usd FROM coinmarketcap " + \
          f"WHERE symbol IN {symbol_str} AND timestamp >= '{earliest_date}' " + \
          f"ORDER BY timestamp"

    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()

    # Remove seconds and microseconds
    for row in rows:
        row['timestamp'] = row['timestamp'].replace(second=0, microsecond=0)

    df_price = pd.DataFrame(rows)
    df_price = df_price.pivot(values='price_usd', columns='symbol', index='timestamp')

    # Fill missing observations
    df_price.fillna(method='backfill', axis='rows', inplace=True)
    df_price.fillna(method='ffill', axis='rows', inplace=True)

    return df_price


def compileData(prices, hours_ahead):
    """
    Generate a dataframe of the top coins

    Input:
        prices (dataframe): dataframe of prices of top coins over duration, hour
        hours_ahead (int): hours ahead to perform prediction

    Output:
        Dataframe of logarithmic returns at coin-x hour and coin+hours_ahead
    """

    # Log difference
    log_prices = prices.apply(np.log10, axis=1)
    r = log_prices.diff(periods=1, axis=0)[1:]

    # Sample only at hourly increments. The sum of the series of returns is thereturn over the whole interval
    # since all but the first and last log prices cancel out
    rh = r.resample('60T').sum()

    # Create list of returns at coin-x hour up to 24 hours
    r_shifts = []
    for shift in range(1, 24):
        rs = rh.shift(shift)
        rs.columns = [name + f"-{shift}" for name in rh.columns]
        r_shifts.append(rs)

    # Merge all returns to generate dataframe of features
    r_merged = rh
    for rs in r_shifts:
        r_merged = pd.merge(r_merged, rs, how='outer', left_index=True, right_index=True)

    # Generate labels and merge dataframes
    # hours_ahead = 1 #24 # How far ahead do we want to predict? The smaller this number, the easier
    r_future = sum([rh.shift(-i) for i in range(1, hours_ahead + 1)])
    r_future.columns = [name + f"+{hours_ahead}" for name in rh.columns]
    r_merged = pd.merge(r_merged, r_future, how='outer', left_index=True, right_index=True)

    # Drop NaN
    r_merged = r_merged.dropna()

    return r_merged


def getFeatures(data, prices, hours_ahead):
    """
    Generate a dataframe of training features

    Input:
        data (dataframe): ataframe of logarithmic returns at coin-x hour and coin+hours_ahead
        prices (dataframe): dataframe of prices of top coins over duration, hour
        hours_ahead (int): hours ahead to perform prediction

    Output:
        Dataframe of logarithmic returns at coin-x hour
    """

    r_merged = data
    symbols = prices.columns.tolist()

    # Define labels
    labels = [f"{symbol}+{hours_ahead}" for symbol in symbols]

    # Drop labels from dataframe
    X = r_merged.copy()
    X.drop(labels, axis=1, inplace=True)

    return X


def getLabels(data, prices, hours_ahead):
    """
    Generate a dataframe of labels

    Input:
        data (dataframe): ataframe of logarithmic returns at coin-x hour and coin+hours_ahead
        prices (dataframe): dataframe of prices of top coins over duration, hour
        hours_ahead (int): hours ahead to perform prediction

    Output:
        Dataframe of logarithmic returns at coin+hours_ahead
    """

    r_merged = data
    symbols = prices.columns.tolist()

    # Define labels
    labels = [f"{symbol}+{hours_ahead}" for symbol in symbols]

    # Get labels only from dataframe
    y = r_merged[labels]

    return y


def getTrainDev(X, y, percent = 0.2):
    """
    Perform training-development dataset split via manual selection from the end of the dataset to prevent data leakage

    Input:
        X (dataframe): features
        y (dataframe): labels
        percent (float): percentage of data to be segmented into development set

    Output:
        X_train (dataframe): training features
        y_train (dataframe): training labels
        X_dev (dataframe): development features
        y_dev (dataframe): development labels
    """

    # Percentage of data to be segmented into development set
    n = int(len(X) * percent)

    # Splitting data in to training and development sets, and dropping 'timestamp'
    X_train = X.iloc[:-n].reset_index()
    X_train.drop('timestamp', axis=1, inplace=True)

    y_train = y.iloc[:-n].reset_index()
    y_train.drop('timestamp', axis=1, inplace=True)

    X_dev = X.iloc[-n:].reset_index()
    X_dev.drop('timestamp', axis=1, inplace=True)

    y_dev = y.iloc[-n:].reset_index()
    y_dev.drop('timestamp', axis=1, inplace=True)

    return X_train, y_train, X_dev, y_dev


def findRegressor(X_train, y_train, X_dev, y_dev):
    """
    Determine regressor that returns the highest r2 score
    """

    # Set up a dictionary to store regressors
    reg_dict = dict()

    reg = LinearRegression()
    reg_dict['LinearRegression()'] = reg

    reg = GradientBoostingRegressor()
    reg_dict['GradientBoostingRegressor()'] = reg

    reg = AdaBoostRegressor()
    reg_dict['AdaBoostRegressor()'] = reg

    reg = MLPRegressor()
    reg_dict['MLPRegressor()'] = reg

    # Set up dictionaries to store the mean squared error, and r2 score of each regressor
    mse_dict = dict()
    r2_dict = dict()

    # Fit each regressor in the dictionary to coins
    for reg_name, reg in reg_dict.items():

        # Lists to store the mean squared error, and r2 score of all coins each time a regressor is applied
        mse_all = []
        r2_all = []

        # Fit the regressor in question on each coin
        for column in y_train.columns:
            reg.fit(X_train, y_train[f'{column}'].values.ravel())
            y_predict = reg.predict(X_dev)

            # Determine the mean squared error and r2 score, and store in lists
            reg_mse = mean_squared_error(y_dev[f'{column}'].as_matrix(), y_predict)
            mse_all.append(reg_mse)

            reg_r2 = r2_score(y_dev[f'{column}'].as_matrix(), y_predict)
            r2_all.append(reg_r2)

        # Determine the average mean squared error of each regressor and store in the dictionary
        mse_average = sum(mse_all) / float(len(mse_all))
        mse_dict[f'{reg_name}'] = mse_average

        # Determine the average r2 score of each regressor and store in the dictionary
        r2_average = sum(r2_all) / float(len(r2_all))
        r2_dict[f'{reg_name}'] = r2_average

        # Find the regressor that gives the minimum mean squared error and maximum r2 score
        min_mse = min(mse_dict, key=mse_dict.get)
        max_r2 = max(r2_dict, key=r2_dict.get)

    return max_r2


def predictReturns(max_r2, X_train, y_train, X_predict, y_dev):
    """
    Predict the values of expected returns

    Input:
        X_predict (dataframe): either X_dev for development, or X_test

    Output:
        y_predict: predicted values of expected returns of coins
        r2 score: if input is X_dev, return r2 score for y_dev and y_predict
    """

    # Optimization of GradientBoostingRegressor hyperparameters
    if max_r2 == 'GradientBoostingRegressor()':
        reg = GradientBoostingRegressor()
        parameters = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                      'learning_rate': expon(scale=10),
                      'n_estimators': sp_randint(1, 100),
                      'max_depth': sp_randint(3, 5)}

    # Optimization of AdaBoostRegressor hyperparameters
    if max_r2 == 'AdaBoostRegressor()':
        reg = AdaBoostRegressor()
        parameters = {'learning_rate': expon(scale=100),
                      'loss': ['linear', 'square', 'exponential'],
                      'n_estimators': sp_randint(1, 100)}

    # Optimization of MLPRegressor hyperparameters in order of importance: 1) learning_rate; 2) momentum,
    # number of hidden, mini-batch size; 3) number of layers, learning rate decay
    if max_r2 == 'MLPRegressor()':
        reg = MLPRegressor()
        parameters = {'learning_rate_init': expon(scale=1),
                      'activation': ['tanh', 'relu'],
                      'momentum': 1 - (10 ** (-4 * np.random.rand()))}

    # Set up a dictionary to store predicted return for each coin
    prediction_dict = dict()

    # List to store the r2 score of each coin
    r2_all = []

    regressor = RandomizedSearchCV(reg, param_distributions=parameters, n_iter=10, scoring='r2')
    for column in y_train.columns:
        regressor.fit(X_train, y_train[f'{column}'].values.ravel())
        y_predict = regressor.predict(X_predict)
        prediction_dict[column] = y_predict

        try:
            reg_r2 = r2_score(y_dev[f'{column}'].as_matrix(), y_predict)
            r2_all.append(reg_r2)
        except (NameError, ValueError):
            pass

    # Determine the average r2 score of all coins
    try:
        r2_average = sum(r2_all) / float(len(r2_all))
    except (ZeroDivisionError, ValueError):
        pass

    return prediction_dict, r2_average


def main():
    # Settings
    hour = 168
    hours_ahead = 1
    min_market_cap = 1E9
    percent = 0.2

    # Get and compile data, and predict Returns
    prices = getPrices(hour, min_market_cap)
    print("Getting prices")
    data = compileData(prices, hours_ahead)
    print("Compiling data")
    X = getFeatures(data, prices, hours_ahead)
    print("Getting features")
    y = getLabels(data, prices, hours_ahead)
    print("Getting features")
    X_train, y_train, X_dev, y_dev = getTrainDev(X, y, percent)
    print("Splitting train-dev data")
    max_r2 = findRegressor(X_train, y_train, X_dev, y_dev)
    print("Finding the most optimal regressor")
    prediction_dict, r2_average = predictReturns(max_r2, X_train, y_train, X_predict, y_dev)
    print("Predicting returns")
    print(prediction_dict)
    print(r2_average)


if __name__ == "__main__":
    main()
    

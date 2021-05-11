# Importing libraries
import pandas as pd
import numpy as np

import datetime
import math

def PCC(dataframe: pd, features: list, target: str):
    p_values = []
    
    y_values = dataframe[target].to_numpy()
    y_mean = y_values.mean()
    
    for column in features:
        x_values = dataframe[column].to_numpy()
        x_mean = x_values.mean()
        numerator = sum((x_values - x_mean) * (y_values- y_mean))
        denominator = np.sqrt(sum((x_values - x_mean)**2) * sum((y_values - y_mean)**2))
        
        r = numerator / denominator
        p_values.append(r)
    
    return p_values

def kernelize(df: pd, features: list, value):
    dataframe = df.copy()
    if type(value) is list:
        for i in range(len(features)):
            dataframe[features[i]] = dataframe[features[i]] ** value[i]
    
    else:
        for column in features:
            dataframe[column] = dataframe[column] ** value
    
    return dataframe

def MSE(y_pred_values, y_values, data_length):
    
    sum_error = sum((y_pred_values - y_values) ** 2)
    
    cost = (1 / data_length) * sum_error
    
    return cost

def LinearRegression(dataframe: pd, 
                     features: list,
                     target: str,
                     rate = 0.00001,
                     epochs = 1000
                    ):
    
    # x_values contains the values of the entire column of "x_column"
    # y_values contains the values of the entire column of "y_column"
    x_values = dataframe[features]
    y_values = dataframe[target]
    
    
    # Just getting the full length of the dataframe
    total_rows = len(dataframe)
    total_columns = len(features)
    
    # Setting up weight and bias
    weights = np.zeros(total_columns)
    bias = 0
    
    # MSE array that will keep the last cost error (just one value)
    mse_array = []
    
    
    # Gradient descent portion
    for i in range(epochs):
        # pred_y_values is a new column, where the m'x+b' formula has been applied
        # Every row in pred_y_values has the formula applied
        pred_y_values = np.dot(x_values, weights) + bias
        
        # Obtaining the partial derivatives of the weight and bias
        der_weight = (1 / total_rows) * (2 * np.dot(x_values.T, (pred_y_values - y_values)))
        der_bias = (1 / total_rows) * (2 * np.sum(pred_y_values - y_values))
        
        # Calculating the new_weight and new bias
        new_weights = weights - rate * der_weight
        new_bias = bias - rate * der_bias
        
        # Helps prevent infinite values by stopping if MSE increased
        if mse_array == []:
            mse_array.append(MSE(pred_y_values, y_values, total_rows))
        else:
            mse = MSE(pred_y_values, y_values, total_rows)
            if mse > mse_array[0]:
                return (weights, bias)
            else:
                mse_array[0] = mse
                weights = new_weights
                bias = new_bias
        
    return (weights, bias)

def predict(df: pd, features: list, weights: list, bias: int, new_column_name = "Prediction"):
    dataframe = df.copy()
    x_values = dataframe[features].to_numpy()
    dataframe[new_column_name] = np.dot(x_values, weights) + bias
    return dataframe

def expandShiftRight(df, column, length, fillna_values=[]):
    new_df = df.copy()
    column_names = list(df.columns.values)
    
    new_df[column + "0"] = new_df[column]
    
    for i in range(length):
        new_df[column + str(i + 1)] = new_df[column + str(i)].shift(1)
        if fillna_values != []:
            new_df.fillna(fillna_values[i])
        
    new_df = new_df.drop(columns=[column + "0"])
    new_df = new_df.dropna()
    return new_df

def meanRightShift(df, start_col, end_col):
    
    
    pos = df.columns.get_loc(start_col)
    end_pos = df.columns.get_loc(end_col) + 1

    mean_list = []
    
    while (pos + 1 <= end_pos):
        mean = df.iloc[0:1, pos:end_pos].mean(axis=1)[-1:].values[0]
        mean_list.append(mean)
        pos += 1
    
    return mean_list

def rightShift(df, start_col, shift_length, fillna_val):
    new_df = df.copy()
    start_pos = df.columns.get_loc(start_col)
    pos = 0
    while pos <= shift_length:
        mean = fillna_val[pos]
        new_df.iloc[pos + 1: pos + 2, start_pos:] = new_df.iloc[pos:pos + 1, start_pos:].shift(1, axis = 1)
        new_df.iloc[pos + 1: pos + 2, start_pos:] = new_df.iloc[pos + 1: pos + 2, start_pos:].fillna(mean)
        pos += 1
    return new_df

if __name__ == "__main__":
    stocks = ["AMC", "F", "BB", "WEN"]
    for stock in stocks:
        # Creating a dataframe with stock data
        stock_df = pd.read_csv("stock data/" + stock + ".csv")
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])

        # Feature engineering data
        stock_df = stock_df[["Date", "Close"]]
        period = 60
        stock_df = expandShiftRight(stock_df, "Close", period)

        # Listing features to be used for prediction
        features = list(stock_df.columns.values)
        features.remove("Date")
        features.remove("Close")

        # Listing target
        target = "Close"

        # Obtaining p-values (Pearson's Formula)
        p_values = PCC(stock_df, features, target)

        # Kernelizing features to the power of their p-value
        stock_df = kernelize(stock_df, features, p_values)

        # Getting the weights and the bias values by using linear regression
        weights, bias = LinearRegression(stock_df, features, target)

        # Setting up new dataframe to store predictions (pred_df)
        new_dates = pd.bdate_range(stock_df.Date.iloc[-1] + datetime.timedelta(days=1), periods=period)
        pred_df = pd.DataFrame(new_dates, columns=["Date"])
        pred_df = pred_df.reset_index(drop=True)
        pred_df = pd.concat([stock_df[-1:], pred_df], ignore_index=True)

        # Retrieving past values to replace missing values created by shifting data
        past_values = list(stock_df.iloc[-1: , 1:period + 1].values[0])
        past_values.reverse()

        # Filling in pred_df with features needed for prediction
        pred_df = rightShift(pred_df, "Close", period - 1, past_values)
        pred_df = pred_df.iloc[1:]
        pred_df["Close"] = stock_df[-period:]["Close"]
        pred_df = pred_df.drop(columns=["Close"])

        # Kernelizing
        pred_df = kernelize(pred_df, features, p_values)

        # Making predictions
        pred_df = predict(pred_df, features, weights, bias)

        # Removing unnecessary columns
        pred_df = pred_df[["Date", "Prediction"]]
        pred_df.columns = ["Date", "Close"]

        # Saving file to csv
        save_name = stock + "_Pred.csv"
        pred_df.to_csv(save_name, index=False)
        print("Predictions computed, saved as " + save_name + ".")



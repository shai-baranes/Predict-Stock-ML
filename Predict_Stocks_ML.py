
# def header_creator(text):
#   print(f" {'#'*100}")    
#   print(" #", " "*96, "#")
#   print(" #", " "*int((95-len(text))/2), text, " "*int((94-len(text))/2), "#")
#   print(" #", " "*96, "#")
#   print("", "#"*100)

# header_creator("Introducing the Machine-Learning Model:")




# YT link: https://www.youtube.com/watch?v=1O_BenficgE&t=554s

# import inspect
# inspect.signature(tbd_func_name)

import pandas as pd
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt




pd.set_option('display.max_columns', 10)


import yfinance as yf

sp500 = yf.Ticker("^GSPC")

sp500 = sp500.history(period="max")

sp500.shape[0]
# 24,460 lines


sp500.info()
# #   Column        Non-Null Count  Dtype  
# ---  ------        --------------  -----  
#  0   Open          24460 non-null  float64
#  1   High          24460 non-null  float64
#  2   Low           24460 non-null  float64
#  3   Close         24460 non-null  float64
#  4   Volume        24460 non-null  int64  
#  5   Dividends     24460 non-null  float64
#  6   Stock Splits  24460 non-null  float64



sp500.index
# DatetimeIndex(['1927-12-30 00:00:00-05:00', '1928-01-03 00:00:00-05:00',
#                '1928-01-12 00:00:00-05:00', '1928-01-13 00:00:00-05:00',
#                ...
#                '2025-05-05 00:00:00-04:00', '2025-05-06 00:00:00-04:00',
#                '2025-05-13 00:00:00-04:00', '2025-05-14 00:00:00-04:00',
#                '2025-05-15 00:00:00-04:00', '2025-05-16 00:00:00-04:00'],




sp500["Close"].plot()
# # plt.show() # Figure_1.png


# another way to plot according to the tutor
sp500.plot.line(y="Close", use_index=True)
# plt.show() # Figure_2.png



# remove extra columns
sp500.drop(columns = ["Dividends", "Stock Splits"], inplace=True)


# by the tutor:
# del sp500["Dividends"]
# del sp500["Stock Splits"]



# prediction on whether the stock goes up or down...


sp500["Tomorrow"] = sp500["Close"].shift(-1)


sp500[["Close", "Tomorrow"]].tail().sort_index(ascending=False) # for debug
#                   Close     Tomorrow
# Date
# 2025-05-16  5958.379883          NaN
# 2025-05-15  5916.930176  5958.379883
# 2025-05-14  5892.580078  5916.930176
# 2025-05-13  5886.549805  5892.580078
# 2025-05-12  5844.189941  5886.549805


sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)


sp500[["Close", "Tomorrow", "Target"]].tail().sort_index(ascending=False)  # for debug
#                   Close     Tomorrow  Target
# Date
# 2025-05-16  5958.379883          NaN       0  # no data for comparing here
# 2025-05-15  5916.930176  5958.379883       1
# 2025-05-14  5892.580078  5916.930176       1
# 2025-05-13  5886.549805  5892.580078       1
# 2025-05-12  5844.189941  5886.549805       1


# remove historical data that will not help for prediction
sp500 = sp500.loc["1990-01-01":] # taking only the relative new data


 ####################################################################################################
 #                                                                                                  #
 #                              Introducing the Machine-Learning Model:                             #
 #                                                                                                  #
 ####################################################################################################


from sklearn.ensemble import RandomForestClassifier # our default model for Machine Learning!

# initializing the model


# n_estimators -> number of individual decision trees that we want to train (higher = more accurate - upto a limit); 100 is low for our performance!
# min_samples_split -> helps us protect against over-feating (higher meaning less accurate model and less over-feat; to play and find our optimized value)
# random_state -> if we run the same model twice, we get the same results for "1", like a temperature value? (maybe better before optimizing other params?)
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

# starting withe a simple baseline model (TBD)
train = sp500.iloc[:-100] # put all rows except for the last 100 rows into the training set
test = sp500.iloc[-100:] # put the last 100 rows into the test set

# should we look for correlations?
predictors = ["Close", "Volume", "Open", "High", "Low"] # don't use the tomorrow or target since the model doesn't get the future in real world!

model.fit(train[predictors], train["Target"])



# precision_score is saying: when we sayed that the market go up, Target=1,  did it actually go up (in percentage)
from sklearn.metrics import precision_score

# for predictions
preds = model.predict(test[predictors])

preds
# array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0,
#        1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
#        1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1])


# converting the data into a column that we can work with for our DataFrame
preds = pd.Series(preds, index=test.index) # predicted target


# let's calculate the precision score
precision_score(test["Target"], preds) # we get 0.5949367088607594 (and we want higher value) almost 50% it goes up or down from 2 choices :)



pd.concat([test["Target"], preds], axis=1)
#                            Target 
# Date
# 2024-12-20 00:00:00-05:00       1  1
# 2024-12-27 00:00:00-05:00       0  1
# ...                           ... ..
# 2025-05-12 00:00:00-04:00       1  0
# 2025-05-16 00:00:00-04:00       0  1

pd.concat([test["Target"], preds.to_frame('Predict')], axis=1)
#                            Target  Predict
# Date
# 2024-12-20 00:00:00-05:00       1        1
# 2024-12-27 00:00:00-05:00       0        1
# ...                           ...      ...
# 2025-05-12 00:00:00-04:00       1        0
# 2025-05-15 00:00:00-04:00       1        1
# 2025-05-16 00:00:00-04:00       0        1

# let's improve the model!


combined = pd.concat([test["Target"], preds.to_frame('Predict')], axis=1) # column-wise

combined.plot() # most of the time we predict stocks incline
# plt.show() # Figure_3.png








# let's build a more robust way to test our algorithm
# back-testing (for trying multiple values on our model)
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


 ####################################################################################################
 #                                                                                                  #
 #                                 Robust Mechanism for Back-Testing                                #
 #                                                                                                  #
 ####################################################################################################




# every trading year has ~ 250 trading days.
# taking 10 years.
# train our model for about a year (250) and then go to next year and so on...
# the steps are as following:
# - take the first 10 years of data and evaluate for the 11th year
# - take the first 11 years of data and evaluate for the 12th year
# - take the first 12 years of data and evaluate for the 13th year
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions) # combining the list of all DFs into a single DF (by default on row axis = axis=0)


predictions = backtest(sp500, model, predictors)

# the prediction is more prone to market drops
predictions["Predictions"].value_counts()
# Predictions
# 0    3745
# 1    2665

# while in most cases in history, we see market climb instead
predictions["Target"].value_counts()/predictions.shape[0] # ~54% that the market goes up
# Target
# 1    0.536505
# 0    0.463495

precision_score(predictions["Target"], predictions["Predictions"])
# 0.5305816135084428


# score for the last year:
precision_score(predictions[-250:]["Target"], predictions[-250:]["Predictions"])
# 0.5942028985507246




 ####################################################################################################
 #                                                                                                  #
 #                                  Adding Predictors to our Model:                                 #
 #                                                                                                  #
 ####################################################################################################

horizons = [2,5,60,250,1000] # rolling average for the: [last 2 days, last week, last ~2 months (60 trading days), last year and the last 10 years]

# let's find the ratios between today's closing price and the mean closing price for these periods
# if the market went up significantly, maybe we should anticipate a turn down and vise-versa...

# loop through the horizons and calculate their means()
new_predictors = []

# 'rolling' is for rolling window calculations, rolling through const time-frames to calculate our needs
for horizon in horizons:
    rolling_average = sp500.rolling(horizon).mean() # it rolls per column

    ratio_column = f"Close_Ratio_{horizon}" # for naming the newly introduced column
    sp500[ratio_column] = sp500["Close"] / rolling_average["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"] # trend since it sees the sum of the Target in the past fer days (few as defined by 'horizon')

    new_predictors += [ratio_column, trend_column] # instead of new_predictors.append(ratio_column) followed by ...  (also note that appending a list to list is [1,2,3,[4,5,]])...




# Appendix
# same shape remains after rolling


# using these 2 queries, we see that once we have data for the full 3 days (indices), we average it and same for all other rows per each column
sp500[["Open", "High"]].rolling(3).mean().head()
#                                  Open        High
# Date
# 1990-01-02 00:00:00-05:00         NaN         NaN
# 1990-01-03 00:00:00-05:00         NaN         NaN
# 1990-01-04 00:00:00-05:00  357.283335  359.680003
# 1990-01-05 00:00:00-05:00  358.040009  358.340007
# 1990-01-08 00:00:00-05:00  355.543345  356.223338


sp500[["Open", "High"]].head()                  
#                                  Open        High
# Date
# 1990-01-02 00:00:00-05:00  353.399994  359.690002
# 1990-01-03 00:00:00-05:00  359.690002  360.589996
# 1990-01-04 00:00:00-05:00  358.760010  358.760010
# 1990-01-05 00:00:00-05:00  355.670013  355.670013
# 1990-01-08 00:00:00-05:00  352.200012  354.239990


# my_list = [358.760010, 355.670013, 352.200012]
# print(sum(my_list)/float(len(my_list)))




sp500.dropna(inplace=True)


model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1) 
# model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1) # old


# let's re-write our predict function slightly
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1] # 'predict_proba' to return the probability that the row will be "0" or "1"
    preds[preds >= .6] = 1 # for the model to be more confident for the price to go up (we want more assurance as w don't want to trace every single day)
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


predictions = backtest(sp500, model, new_predictors) 
predictions["Predictions"].value_counts()
# 0.0    4540
# 1.0     869 # less predictions on investing (less on positive Target)



precision_score(predictions["Target"], predictions["Predictions"])
# 0.570771001150748

precision_score(predictions[-250:]["Target"], predictions[-250:]["Predictions"])
# 0.5555555555555556 even less on the last year


# you can also utilize the 'prediction.plot()' here



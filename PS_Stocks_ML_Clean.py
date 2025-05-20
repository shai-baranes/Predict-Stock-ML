import inspect
# inspect.signature(sns.scatterplot)
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier # our default model for Machine Learning!
from sklearn.metrics import precision_score
from tabulate import tabulate

sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")

VIX = yf.Ticker("^VIX")
VIX = VIX.history(period="max") # TBD get the list comprehension that removed the hrs-min-sec 00:00:00 from the format

SPY = yf.Ticker("SPY")
SPY = SPY.history(period="max") # TBD get the list comprehension that removed the hrs-min-sec 00:00:00 from the format

# VOO = yf.Ticker("VOO") # not enough data (by dropping Null we remain we too few...)
# VOO = VOO.history(period="max") # TBD get the list comprehension that removed the hrs-min-sec 00:00:00 from the format

IVV = yf.Ticker("IVV")
IVV = IVV.history(period="max") # TBD get the list comprehension that removed the hrs-min-sec 00:00:00 from the format



futures = yf.Ticker("ES=F")
futures = futures.history(period="max") # TBD get the list comprehension that removed the hrs-min-sec 00:00:00 from the format



inflation = pd.read_csv("./CSV_Inputs/Inflation.csv", index_col=0)
# inflation = pd.read_csv("./CSV_Inputs/Inflation.csv", index_col="observation_date")
interest = pd.read_csv("./CSV_Inputs/Interest_Rate.csv", index_col=0)
# market_yield = pd.read_csv("./CSV_Inputs/Market_Yield.csv", index_col=0)




inflation.index = pd.to_datetime(inflation.index)
interest.index = pd.to_datetime(interest.index)
# market_yield.index = pd.to_datetime(market_yield.index)



inflation.index.name = None
interest.index.name = None
# market_yield.index.name = None


# remove extra columns
sp500.drop(columns = ["Dividends", "Stock Splits"], inplace=True)

sp500["Tomorrow"] = sp500["Close"].shift(-1)

sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)


# utilized to align VIX and S&P dates, having different hh:mm:ss values
def format_date_index(index):
    return re.sub(r" \d{2}:\d{2}:\d{2}-\d{2}:\d{2}", "", index)



shortened_vix_date = [format_date_index(str(item)) for item in VIX.index]
VIX.index = pd.to_datetime(shortened_vix_date)

shortened_spy_date = [format_date_index(str(item)) for item in SPY.index]
SPY.index = pd.to_datetime(shortened_spy_date)

# shortened_voo_date = [format_date_index(str(item)) for item in VOO.index]
# VOO.index = pd.to_datetime(shortened_voo_date)

shortened_ivv_date = [format_date_index(str(item)) for item in IVV.index]
IVV.index = pd.to_datetime(shortened_ivv_date)

shortened_sp500_date = [format_date_index(str(item)) for item in sp500.index]
sp500.index = pd.to_datetime(shortened_sp500_date)

shortened_futures_date = [format_date_index(str(item)) for item in futures.index]
futures.index = pd.to_datetime(shortened_futures_date)

sp500['_Index_'] = shortened_sp500_date

# Reordering columns to have the new index at the left side (later we can also drop it)
sp500 = sp500.iloc[:, [7, 0, 1, 2, 3, 4, 5, 6]]  # 'Sr.no', 'Maths Score', 'Name'



# test it
sp500 = pd.concat([sp500, VIX["Close"].to_frame('VIX_Close')], axis=1)
sp500 = pd.concat([sp500, SPY["Close"].to_frame('SPY_Close')], axis=1)
# sp500 = pd.concat([sp500, VOO["Close"].to_frame('VOO_Close')], axis=1)
sp500 = pd.concat([sp500, IVV["Close"].to_frame('IVV_Close')], axis=1)

sp500 = pd.concat([sp500, futures["Close"].to_frame('Futures_Close')], axis=1) # TBD need to check that we have enough rows



sp500 = pd.concat([sp500, inflation["EXPINF10YR"].to_frame('Inflation')], axis=1)
sp500 = pd.concat([sp500, interest["REAINTRATREARAT10Y"].to_frame('Interest_Rate')], axis=1)
# sp500 = pd.concat([sp500, market_yield["DFII10"].to_frame('Market_Yield')], axis=1)




sp500['Inflation'] = sp500['Inflation'].fillna(sp500['Inflation'].ffill())
# sp500['Inflation'] = sp500['Inflation'].interpolate(method='linear')
sp500['Interest_Rate'] = sp500['Interest_Rate'].fillna(sp500['Interest_Rate'].ffill())
# sp500['Interest_Rate'] = sp500['Interest_Rate'].interpolate(method='linear')

# sp500['Market_Yield'] = sp500['Market_Yield'].interpolate(method='linear') # TBD to check - to many nulls



# remove historical data that will not help for prediction
sp500 = sp500.loc["1990-01-01":]


sp500.dropna(subset=['_Index_'], inplace=True) # drop all empty rows where we only had the Inflation rate by concat
# sp500.dropna(subset=['Market_Yield'], inplace=True) # TBD temp...


with open(r"temp_sp500_tabular_view.txt", "w") as f:
    f.write(tabulate(sp500.sort_index(ascending=False), headers='keys', tablefmt='psql', showindex=False))



sp500.drop(['_Index_'], axis=1, inplace=True) # the rolling function cannot work on these dates (was needed only for a nices tabular view)


# def backtest(data, model, predictors, start=2500, step=250):
def backtest(data, model, predictors, start=1000, step=250): # TBD for VOO
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)


horizons = [2,5,60,250,1000]
# horizons = [2,5,60,250]

new_predictors = ["Volume"]
# new_predictors = ["Volume", "Inflation", "Interest_Rate"]
# new_predictors = ["Volume", "VIX_Close"]
# new_predictors = ["Volume", "VIX_Close", "Futures_Close"]
# new_predictors = ["Volume", "VIX_Close", "Inflation"]
# new_predictors = ["VIX_Close"]

for horizon in horizons:
    rolling_average = sp500.rolling(horizon).mean() 

    ratio_close_column = f"Close_Ratio_{horizon}"
    sp500[ratio_close_column] = sp500["Close"] / rolling_average["Close"]

    ratio_future_close_column = f"Future_Close_Ratio_{horizon}"
    sp500[ratio_future_close_column] = sp500["Futures_Close"] / rolling_average["Futures_Close"]

    ratio_interest_column = f"interest_Ratio_{horizon}"
    sp500[ratio_interest_column] = sp500["Interest_Rate"] / rolling_average["Interest_Rate"]

    ratio_inflation_column = f"inflation_Ratio_{horizon}"
    sp500[ratio_inflation_column] = sp500["Inflation"] / rolling_average["Inflation"]

    ratio_spy_column = f"SPY_Close_Ratio_{horizon}"
    sp500[ratio_spy_column] = sp500["SPY_Close"] / rolling_average["SPY_Close"]

    # ratio_voo_column = f"VOO_Close_Ratio_{horizon}"
    # sp500[ratio_voo_column] = sp500["VOO_Close"] / rolling_average["VOO_Close"]

    ratio_ivv_column = f"IVV_Close_Ratio_{horizon}"
    sp500[ratio_ivv_column] = sp500["IVV_Close"] / rolling_average["IVV_Close"]

    ratio_vix_column = f"VIX_Close_Ratio_{horizon}"
    sp500[ratio_vix_column] = sp500["VIX_Close"] / rolling_average["VIX_Close"]

    # ratio_yield_column = f"yield_Ratio_{horizon}"
    # sp500[ratio_yield_column] = sp500["Market_Yield"] / rolling_average["Market_Yield"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_close_column, trend_column, ratio_spy_column, ratio_future_close_column, ratio_ivv_column, ratio_vix_column, ratio_interest_column, ratio_inflation_column]
    # new_predictors += [ratio_close_column, trend_column, ratio_interest_column, ratio_inflation_column, ratio_voo_column, ratio_spy_column, ratio_future_close_column, ratio_ivv_column, ratio_vix_column]
    # new_predictors += [ratio_close_column, trend_column, ratio_interest_column, ratio_inflation_column, ratio_yield_column]
    # new_predictors += [ratio_close_column, trend_column]


# TBD check what happens if I don't remove it?
sp500.dropna(inplace=True) # TBD to understand who provide these NANs ???



model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
# model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1) #0.65
# model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1) #0.64


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .55] = 1
    preds[preds < .55] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined




predictions = backtest(sp500, model, new_predictors) 

with open(r"temp_predictions.txt", "w") as f:
    f.write(tabulate(predictions.sort_index(ascending=False), headers='keys', tablefmt='psql', showindex=True))


score_all = precision_score(predictions["Target"], predictions["Predictions"])
# 0.570771001150748

score_250 = precision_score(predictions[-250:]["Target"], predictions[-250:]["Predictions"]) # somehow only the last period is improved after adding VIX & Volume
# 0.7368421052631579



print(f"{score_all=:.2f}")# 0.5975609756097561 -> after removing VOO -> 0.5451612903225806
print(f"{score_250=:.2f}")# 0.5483870967741935 -> after removing VOO -> 0.6785714285714286

# number of investment during the 250 days period:
investments_per_250_period = predictions[-250:].query('Predictions == 1').shape[0]

# miss oppurtunities are not mentiones here
failed_investment_250 = predictions[-250:].query('Target == 0 and Predictions == 1').shape[0]/investments_per_250_period          

success_investments_250 = predictions[-250:].query('Target == 1 and Predictions == 1').shape[0]/investments_per_250_period          
# 0.08571428571428572



print(f"{investments_per_250_period=}") 
print(f"{failed_investment_250=:.2f}") 
print(f"{success_investments_250=:.2f}") 


# score_all=0.53
# score_250=0.68
# investments_per_250_period=28
# failed_investment_250=0.32
# success_investments_250=0.68


## after changing from >= .55 into >= .6
# score_all=0.54
# score_250=0.66
# investments_per_250_period=77
# failed_investment_250=0.34
# success_investments_250=0.66


## after changing from start=1000 into start=2500
# score_all=0.54
# score_250=0.65
# investments_per_250_period=79
# failed_investment_250=0.35
# success_investments_250=0.65

# last - current
# score_all=0.54
# score_250=0.65
# investments_per_250_period=81
# failed_investment_250=0.35
# success_investments_250=0.65
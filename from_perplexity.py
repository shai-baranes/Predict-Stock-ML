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


IVV = yf.Ticker("IVV")
IVV = IVV.history(period="max") # TBD get the list comprehension that removed the hrs-min-sec 00:00:00 from the format



futures = yf.Ticker("ES=F")
futures = futures.history(period="max") # TBD get the list comprehension that removed the hrs-min-sec 00:00:00 from the format






# remove extra columns
sp500.drop(columns = ["Dividends", "Stock Splits"], inplace=True)

sp500["Tomorrow"] = sp500["Close"].shift(-1) # 'Lag1'

sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)



# sp500['garman-klass_vol'] = ((np.log(sp500['High']) - np.log(sp500['Low']))**2)/2-(2*np.log(2)-1)*((np.log(sp500['Close'])-np.log(sp500['Open']))**2)


# utilized to align VIX and S&P dates, having different hh:mm:ss values (removing the time section)
def format_date_index(index):
    return re.sub(r" \d{2}:\d{2}:\d{2}-\d{2}:\d{2}", "", index)



shortened_vix_date = [format_date_index(str(item)) for item in VIX.index]
VIX.index = pd.to_datetime(shortened_vix_date)

shortened_spy_date = [format_date_index(str(item)) for item in SPY.index]
SPY.index = pd.to_datetime(shortened_spy_date)


shortened_ivv_date = [format_date_index(str(item)) for item in IVV.index]
IVV.index = pd.to_datetime(shortened_ivv_date)

shortened_sp500_date = [format_date_index(str(item)) for item in sp500.index]
sp500.index = pd.to_datetime(shortened_sp500_date)

shortened_futures_date = [format_date_index(str(item)) for item in futures.index]
futures.index = pd.to_datetime(shortened_futures_date)

sp500['_Index_'] = shortened_sp500_date



index_arr = [i for i in range(len(sp500.columns))]
index_arr = index_arr[-1:] + index_arr[1:-1] # shifting the array in one place (for having the _index_ placed first)
# [0, 1, 2, 3, 4, 5, 6, 7, 8]


# Reordering columns to have the new index at the left side (later we can also drop it)
sp500 = sp500.iloc[:, index_arr]  # 'Sr.no', 'Maths Score', 'Name' (TBD need to update upon event of adding columns)



# test it
sp500 = pd.concat([sp500, VIX["Close"].to_frame('VIX_Close')], axis=1)
sp500 = pd.concat([sp500, SPY["Close"].to_frame('SPY_Close')], axis=1)
sp500 = pd.concat([sp500, IVV["Close"].to_frame('IVV_Close')], axis=1)

sp500 = pd.concat([sp500, futures["Close"].to_frame('Futures_Close')], axis=1) # TBD need to check that we have enough rows






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
# new_predictors = ["Volume", "garman-klass_vol"]

for horizon in horizons:
    rolling_average = sp500.rolling(horizon).mean() 

    ratio_close_column = f"Close_Ratio_{horizon}"
    sp500[ratio_close_column] = sp500["Close"] / rolling_average["Close"]

    ratio_future_close_column = f"Future_Close_Ratio_{horizon}"
    sp500[ratio_future_close_column] = sp500["Futures_Close"] / rolling_average["Futures_Close"]


    ratio_spy_column = f"SPY_Close_Ratio_{horizon}"
    sp500[ratio_spy_column] = sp500["SPY_Close"] / rolling_average["SPY_Close"]


    ratio_ivv_column = f"IVV_Close_Ratio_{horizon}"
    sp500[ratio_ivv_column] = sp500["IVV_Close"] / rolling_average["IVV_Close"]

    ratio_vix_column = f"VIX_Close_Ratio_{horizon}"
    sp500[ratio_vix_column] = sp500["VIX_Close"] / rolling_average["VIX_Close"]



    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_close_column, trend_column, ratio_spy_column, ratio_future_close_column, ratio_ivv_column, ratio_vix_column]




latest_data = sp500.iloc[[-1]].copy() # saving the last data before we lose it to the "sp500.dropna(inplace=True)" line
latest_data.drop(['Tomorrow', 'Target'], axis=1, inplace=True)



# TBD check what happens if I don't remove it?
sp500.dropna(inplace=True) # TBD to understand who provide these NANs ???



model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


def predict(train, test, predictors, model):
    # model.fit(train[predictors], train["Target"])
    # preds = model.predict_proba(test[predictors])[:,1]
    # preds[preds >= .54] = 1
    # preds[preds < .54] = 0
    # preds = pd.Series(preds, index=test.index, name="Predictions")
    # combined = pd.concat([test["Target"], preds], axis=1)
    # return combined

    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1] # list of probabilities
    probs = preds.copy()

    preds[preds >= .54] = 1
    preds[preds < .54] = 0  #use the below to add the probability column

    preds = pd.Series(preds, index=test.index, name="Predictions")
    probs = pd.Series(probs, index=test.index, name="Probabilities")
    combined = pd.concat([test["Target"], preds, probs], axis=1)
    # combined = pd.concat([test["Target"], preds], axis=1)
    return combined




predictions = backtest(sp500, model, new_predictors) 



# ======== NEW CODE ADDITION ========
# Get latest available market data
# latest_data = sp500.iloc[[-1]].copy()
# sp500.tail(1) # same as this (also note that by: sp500.iloc[-1] we get the last item as Serier and not as a 2D DataFrame)


# Prepare features for prediction (using existing pre-processed data)
X_latest = latest_data[new_predictors]

# Make prediction for tomorrow
future_pred = model.predict_proba(X_latest)[:,1]
future_class = 1 if future_pred >= 0.54 else 0 # future_class -> 'future prediction'

# Create future prediction entry
future_date = latest_data.index + pd.Timedelta(days=1) # not accounting for the weekends?
future_pred_df = pd.DataFrame({
    "Target": [np.nan],  # No actual value yet
    "Predictions": future_class,
    "Probabilities": future_pred[0]
}, index=future_date)

# Append to existing predictions
combined_predictions = pd.concat([predictions, future_pred_df])

# # Save updated predictions with future forecast
# combined_predictions.to_csv("AI_predictions.txt", sep="\t", float_format="%.2f")


# ======== END ========



with open(r"AI_predictions.txt", "w") as f:
    f.write(tabulate(combined_predictions.sort_index(ascending=False), headers='keys', tablefmt='psql', showindex=True))

# get scoring on historical predictions where we can test
score_all = precision_score(predictions["Target"], predictions["Predictions"])
# 0.570771001150748

score_250 = precision_score(predictions[-250:]["Target"], predictions[-250:]["Predictions"]) # somehow only the last period is improved after adding VIX & Volume
# 0.7368421052631579



print(f"{score_all=:.2f}")
print(f"{score_250=:.2f}")

# number of investment during the 250 days period:
investments_per_250_period = predictions[-250:].query('Predictions == 1').shape[0]

# miss oppurtunities are not mentiones here
failed_investment_250 = predictions[-250:].query('Target == 0 and Predictions == 1').shape[0]/investments_per_250_period          

success_investments_250 = predictions[-250:].query('Target == 1 and Predictions == 1').shape[0]/investments_per_250_period          
# 0.08571428571428572



print(f"{investments_per_250_period=}") 
print(f"{failed_investment_250=:.2f}") 
print(f"{success_investments_250=:.2f}") 



# score_all=0.55
# score_250=0.64
# investments_per_250_period=130
# failed_investment_250=0.36
# success_investments_250=0.64


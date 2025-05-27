# TBD add threshold param to prediction!
import inspect  # noqa: F401
# inspect.signature(sns.scatterplot)
import requests
from datetime import datetime
from fake_useragent import UserAgent
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # noqa: F401
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier # our default model for Machine Learning!
from sklearn.metrics import precision_score
from tabulate import tabulate
import fear_and_greed # not really used here, can fetch the current fear value  # noqa: F401

# ===============================================

BASE_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/" # for fear/greed API data pull
START_DATE = '2020-09-19' # to compensate over the missing data from the historical CSV file
END_DATE = datetime.today().strftime('%Y-%m-%d') # today
PROB_THRES = 0.54
# PROB_THRES = 0.54

ua = UserAgent()

headers = {
   'User-Agent': ua.random,
   }

r = requests.get(BASE_URL + START_DATE, headers = headers)
data = r.json()


fear_greed_history = r'./CSV_Inputs/fear-greed.csv'

# fng_data = pd.read_csv(filepath, usecols=['Date', 'Fear Greed'])
# fng_data['Date'] = pd.to_datetime(fng_data['Date'], format='%Y-%m-%d')  # note that the exact formatting is not doable directly from the read_csv() call
# fng_data.set_index('Date', inplace=True)


#fng stands for Fear-n-Greed
fng_data = pd.read_csv(fear_greed_history,
                 usecols=['Date', 'Fear Greed'],
                 parse_dates=[0],
                 index_col=[0],
                 )





# TBD try this using the fillna(0) instead
missing_dates = []
all_dates = (pd.date_range(fng_data.index[0], END_DATE, freq='D'))
for date in all_dates:
    if date not in fng_data.index:
        missing_dates.append(date)
        #print(date)
        fng_data.loc[date] = [0]
fng_data.sort_index(inplace=True)



# filling in the missing data from the CNN API - json file

for data in data['fear_and_greed_historical']['data']:
# for data in ((data['fear_and_greed_historical']['data'])):
   x = int(data['x'])
   x = datetime.fromtimestamp(x / 1000).strftime('%Y-%m-%d') # converting time stamps like this: 1743033600000.0
   y = int(data['y'])
   fng_data.loc[x, 'Fear Greed'] = y
   # fng_data.at[x, 'Fear Greed'] = y
# ===============================================


sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")



VIX = yf.Ticker("^VIX")
VIX = VIX.history(period="max") # TBD get the list comprehension that removed the hrs-min-sec 00:00:00 from the format

# SPY is an ETF - Exchange Trade Funds (Teudot Sal) of SP500 (we also had VOO, but had restricted data since too new)
# similarly, QQQ is ETF for NASDAQ and DJI for Daw-Johns
SPY = yf.Ticker("SPY")
SPY = SPY.history(period="max") # TBD get the list comprehension that removed the hrs-min-sec 00:00:00 from the format


IVV = yf.Ticker("IVV")
IVV = IVV.history(period="max") # TBD get the list comprehension that removed the hrs-min-sec 00:00:00 from the format



futures = yf.Ticker("ES=F")
futures = futures.history(period="max") # TBD get the list comprehension that removed the hrs-min-sec 00:00:00 from the format


# fear_and_greed.get()[0] = 69.6285714285714




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
index_arr = index_arr[-1:] + index_arr[:-1] # shifting the array in one place (for having the _index_ placed first)
# [0, 1, 2, 3, 4, 5, 6, 7, 8]


# Reordering columns to have the new index at the left side (later we can also drop it)
sp500 = sp500.iloc[:, index_arr]  # 'Sr.no', 'Maths Score', 'Name' (TBD need to update upon event of adding columns)



# test it
sp500 = pd.concat([sp500, VIX["Close"].to_frame('VIX_Close')], axis=1)
sp500 = pd.concat([sp500, SPY["Close"].to_frame('SPY_Close')], axis=1)
sp500 = pd.concat([sp500, IVV["Close"].to_frame('IVV_Close')], axis=1)

sp500 = pd.concat([sp500, futures["Close"].to_frame('Futures_Close')], axis=1) # TBD need to check that we have enough rows



sp500 = pd.concat([sp500, fng_data["Fear Greed"]], axis=1, join='inner')





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

new_predictors = ["Volume", "Fear Greed"]
# new_predictors = ["Volume", "Fear Greed"]
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


    # if horizon in [60,250,1000]:
    #     ratio_fng_column = f"FNG_Ratio_{horizon}"
    #     sp500[ratio_fng_column] = sp500["Fear Greed"] / rolling_average["Fear Greed"]
    #     new_predictors += [ratio_fng_column]



    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_close_column, trend_column, ratio_spy_column, ratio_future_close_column, ratio_ivv_column, ratio_vix_column]




latest_data = sp500.iloc[[-1]].copy() # saving the last data before we lose it to the "sp500.dropna(inplace=True)" line
latest_data.drop(['Tomorrow', 'Target'], axis=1, inplace=True)



# TBD check what happens if I don't remove it?
# sp500.dropna(inplace=True) # TBD to understand who provide these NANs ???




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

    preds[preds >= PROB_THRES] = 1
    preds[preds < PROB_THRES] = 0  #use the below to add the probability column
    # preds[preds >= .54] = 1
    # preds[preds < .54] = 0  #use the below to add the probability column

    preds = pd.Series(preds, index=test.index, name="Predictions")
    probs = pd.Series(probs, index=test.index, name="Probabilities")
    combined = pd.concat([test["Target"], preds, probs], axis=1)
    # combined = pd.concat([test["Target"], preds], axis=1)
    return combined




predictions = backtest(sp500, model, new_predictors) 



# ======== NEW CODE ADDITION PERPLEXITY========
# Get latest available market data
# latest_data = sp500.iloc[[-1]].copy() # moved up
# sp500.tail(1) # same as this (also note that by: sp500.iloc[-1] we get the last item as Serier and not as a 2D DataFrame)


# Prepare features for prediction (using existing pre-processed data)
X_latest = latest_data[new_predictors]

# Make prediction for tomorrow
pred = model.predict_proba(X_latest)[:,1]
future_prediction = 1 if pred >= PROB_THRES else 0 # future_class -> 'future prediction'
# future_class = 1 if pred >= 0.54 else 0 # future_class -> 'future prediction'

# Create future prediction entry
future_date = latest_data.index + pd.Timedelta(days=1) # not accounting for the weekends?
future_pred_df = pd.DataFrame({
    "Target": [np.nan],  # No actual value yet
    "Predictions": future_prediction,
    "Probabilities": pred[0]
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





DF_full_view = pd.concat([sp500, predictions['Predictions']], axis=1, join='inner')
DF_full_view['Daily_Change'] = (DF_full_view['Close'] - DF_full_view['Open'])/DF_full_view['Open']
columns_in_focus = ['Open', 'High', 'Low', 'Close', 'Daily_Change', 'Target', 'Predictions']

with open(r"Full_Tabular_View.txt", "w") as f:
    f.write(tabulate(DF_full_view[columns_in_focus].sort_index(ascending=False), headers='keys', tablefmt='psql', showindex=True))


DF_full_view[columns_in_focus].sort_index(ascending=False).to_csv('Full_Tabular_View.csv')





print(f"{score_all=:.2f}")
print(f"{score_250=:.2f}")

# number of investment during the 250 days period:
investments_per_250_period = predictions[-250:].query('Predictions == 1').shape[0]
investments_per_250_period_2 = DF_full_view[-250:].query('Predictions == 1').shape[0]

assert investments_per_250_period == investments_per_250_period_2




# miss opportunities are not mentions here
failed_investment_250 = predictions[-250:].query('Target == 0 and Predictions == 1').shape[0]/investments_per_250_period

success_investments_250 = predictions[-250:].query('Target == 1 and Predictions == 1').shape[0]/investments_per_250_period
# 0.08571428571428572

actual_failed_investments =  DF_full_view[-250:].query('Predictions == 1 and Daily_Change < 0').shape[0]/investments_per_250_period





print(f"{investments_per_250_period=}") 
print(f"{failed_investment_250=:.2f}") 
print(f"{success_investments_250=:.2f}") 



# score_all=0.55
# score_250=0.64
# investments_per_250_period=130
# failed_investment_250=0.36
# success_investments_250=0.64

# after adding the Fear and Greed index (w/o moving average)
score_all=0.62
score_250=0.62
investments_per_250_period=177
failed_investment_250=0.38
success_investments_250=0.62




# TBD - probably need daily percentage for the training!
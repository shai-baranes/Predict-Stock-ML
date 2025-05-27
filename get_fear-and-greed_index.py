import inspect
# inspect.signature(sns.scatterplot)

import requests, csv, json, urllib
import pandas as pd
import time
from fake_useragent import UserAgent
from datetime import datetime

BASE_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
START_DATE = '2020-09-19' # to compensate over the missing data from the historical CSV file
END_DATE = datetime.today().strftime('%Y-%m-%d') # today
# END_DATE = '2022-06-02'
ua = UserAgent()

headers = {
   'User-Agent': ua.random,
   }

r = requests.get(BASE_URL + START_DATE, headers = headers)
data = r.json()


filepath = r'./CSV_Inputs/fear-greed.csv'

# fng_data = pd.read_csv(filepath, usecols=['Date', 'Fear Greed'])
# fng_data['Date'] = pd.to_datetime(fng_data['Date'], format='%Y-%m-%d')  # note that the exact formatting is not doable directly from the read_csv() call
# fng_data.set_index('Date', inplace=True)



fng_data = pd.read_csv(filepath,
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
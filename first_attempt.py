import numpy as np
import pandas as pd
import requests
import xlsxwriter
import math
import starter_filessub
import os
import re
# os.getcwd()






# stocks = pd.read_csv('./starter_files/sp_500_stocks.csv')
stocks = pd.read_csv('./starter_files/sp_500_stocks.csv')

#     Ticker
# 0        A
# 1      AAL
# 2      AAP
# 3     AAPL
# 4     ABBV
# ..     ...
# 500    YUM
# 501    ZBH
# 502   ZBRA
# 503   ZION
# 504    ZTS



# Acquire API Token
from starter_files.secrets import POLYGON_CLOUD_API_TOKEN  # noqa: E402
# a2C96LuhRFZnI3fzToR_GYzELkG9EHrB




# for our firsy API Call, we need:
#     - 'Market capitalization' per stock
#     - 'Price' pwe stock


# search for 'iex cloud docs'
# this is probably for the real deal - not sandbox mode: https://publicapi.dev/iex-cloud-api


# https://www.iexcloud.io/






# from polygon import RESTClient

# client = RESTClient("a2C96LuhRFZnI3fzToR_GYzELkG9EHrB")

# tickers = []
# for t in client.list_tickers(
#     market="stocks",
#     active="true",
#     order="asc",
#     limit="100",
#     sort="ticker",
#     ):
#     tickers.append(t)

# print(tickers)







from polygon import RESTClient  # noqa: E402
# https://docs.astral.sh/ruff/rules/module-import-not-at-top-of-file/


client = RESTClient(POLYGON_CLOUD_API_TOKEN)  # POLYGON_API_KEY environment variable is used


with open("my_file.txt", "w", encoding='utf-8') as f:
    for ticker in client.list_tickers(market="fx", limit=1000):
        f.write(f"{str(ticker)}\n")


# Ticker(active=True, cik=None, composite_figi=None, currency_name='Australian dollar', currency_symbol='AUD', base_currency_symbol='AED', base_currency_name='United Arab Emirates dirham', delisted_utc=None, last_updated_utc='2017-01-01T00:00:00Z', locale='global', market='fx', name='United Arab Emirates dirham - Australian dollar', primary_exchange=None, share_class_figi=None, ticker='C:AEDAUD', type=None, source_feed=None)
# Ticker(active=True, cik=None, composite_figi=None, currency_name='Bahraini dinar', currency_symbol='BHD', base_currency_symbol='AED', base_currency_name='United Arab Emirates dirham', delisted_utc=None, last_updated_utc='2017-01-01T00:00:00Z', locale='global', market='fx', name='United Arab Emirates dirham - Bahraini dinar', primary_exchange=None, share_class_figi=None, ticker='C:AEDBHD', type=None, source_feed=None)
# Ticker(active=True, cik=None, composite_figi=None, currency_name='Canadian dollar', currency_symbol='CAD', base_currency_symbol='AED', base_currency_name='United Arab Emirates dirham', delisted_utc=None, last_updated_utc='2017-01-01T00:00:00Z', locale='global', market='fx', name='United Arab Emirates dirham - Canadian dollar', primary_exchange=None, share_class_figi=None, ticker='C:AEDCAD', type=None, source_feed=None)
# Ticker(active=True, cik=None, composite_figi=None, currency_name='Swiss franc', currency_symbol='CHF', base_currency_symbol='AED', base_currency_name='United Arab Emirates dirham', delisted_utc=None, last_updated_utc='2017-01-01T00:00:00Z', locale='global', market='fx', name='United Arab Emirates dirham - Swiss franc', primary_exchange=None, share_class_figi=None, ticker='C:AEDCHF', type=None, source_feed=None)
# Ticker(active=True, cik=None, composite_figi=None, currency_name='Danish krone', currency_symbol='DKK', base_currency_symbol='AED', base_currency_name='United Arab Emirates dirham', delisted_utc=None, last_updated_utc='2017-01-01T00:00:00Z', locale='global', market='fx', name='United Arab Emirates dirham - Danish krone', primary_exchange=None, share_class_figi=None, ticker='C:AEDDKK', type=None, source_feed=None)
# Ticker(active=True, cik=None, composite_figi=None, currency_name='Euro', currency_symbol='EUR', base_currency_symbol='AED', base_currency_name='United Arab Emirates dirham', delisted_utc=None, last_updated_utc='2017-01-01T00:00:00Z', locale='global', market='fx', name='United Arab Emirates dirham - Euro', primary_exchange=None, share_class_figi=None, ticker='C:AEDEUR', type=None, source_feed=None)
# ...
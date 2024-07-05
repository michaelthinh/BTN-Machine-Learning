import requests
import pandas as pd
from constant import coins, timeframes

def getDataFromCoin(coin, timeframe, day_number):
  url = f"https://www.bitstamp.net/api/v2/ohlc/{coin}/"
  params = {
          "step":timeframe,
          "limit":int(day_number),
          }
  df = requests.get(url, params=params).json()["data"]["ohlc"]

  df = pd.DataFrame(df)
  df.timestamp = pd.to_datetime(df.timestamp, unit = "s")
  df.open = df.open.astype(float)
  df.close = df.close.astype(float)
  df.high = df.high.astype(float)
  df.low = df.low.astype(float)
  return df

def getDataFromCoinToCSV(coin):
  df = getDataFromCoin(coin, timeframes['day']['value'], 1000)
  new_df = df[['timestamp', 'open', 'close', 'high', 'low']]
  new_df.to_csv(f'./data/{coin}.csv')

def getAllDataToCSV():
  for coin in coins:
    getDataFromCoinToCSV(coin)

getAllDataToCSV()
import yfinance as yf
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier

# Initialise a Ticker class to download price history for a symbol
sp500 = yf.Ticker('^GSPC')
# Query the historical prices
sp500 = sp500.history(period='max')

sp500.plot.line(y='Close', use_index=True)

# Deleting unused columns
del sp500['Dividends']
del sp500['Stock Splits']

# Setting up a target for ML

# Shift all the prices back one day
sp500['Tomorrow'] = sp500['Close'].shift(-1)
# Is tomorrow's price greater than today
sp500['Target'] = (sp500['Tomorrow'] > sp500['Close']).astype(int)
# Remove all data before 1990
sp500 = sp500.loc['1990-01-01':].copy()

# Training an initial machine learning model

# Setting up initial model


print(sp500)

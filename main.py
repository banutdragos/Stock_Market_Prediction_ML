import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas   as pd


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
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
# Split data into train and test set
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

# Use predictors to try and predict the target
predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
model.fit(train[predictors], train['Target'])

# Generate predictions and format it into a pandas series
preds = model.predict(test[predictors])
preds = pd.Series(preds, index = test.index)

# Calculate precision score
precscore = precision_score(test['Target'], preds)

# Plotting predictions and target
combined = pd.concat([test['Target'], preds], axis=1)
combined.plot()

# Creating the backtesting system to be able to use a bigger data set

def predict(train, test, predictors, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name='Predictions')
    combined = pd.concat([test['Target'], preds], axis=1)
    return combined

# min 18:38 gotta study this a lil bit more.
# start=2500 is 10 years of data used to train the first model.
# step=250 train model for a year then go to the next one.
# take 10 years of data and predict values for the 11th,
# then we'll take the first 11 years of data and predict values for the 12th and so on.
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict_v2(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

# predictions = backtest(sp500, model, predictors)
# precscore = precision_score(predictions['Target'], predictions['Predictions'])

# Adding additional predictors

# min 24:00
# Will be used to calculate the mean closing price in the last 2 days, last trading week,
# last 3 months, last year and last four years.
# Then will find a ratio between today's closing price and the closing price in those periods
# e.g. The stock went up a ton? Then it's due to fall
horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()

    ratio_column = f'Close_Ratio_{horizon}'
    sp500[ratio_column] = sp500['Close'] / rolling_averages['Close']

    trend_column = f'Trend_{horizon}'
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()['Target']

    new_predictors += [ratio_column, trend_column]

sp500 = sp500.dropna()


# Improving the model

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# min 29:00
def predict_v2(train, test, predictors, model):
    model.fit(train[predictors], train['Target'])
    # returns a probability
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name='Predictions')
    combined = pd.concat([test['Target'], preds], axis=1)
    return combined

predictions = backtest(sp500, model, new_predictors)
precscore = precision_score(predictions['Target'], predictions['Predictions'])


#print(sp500)
print('0.0 = Predicted the market will go DOWN\n1.0 = Predicted the market will go UP\n',predictions['Predictions'].value_counts())
#print(preds)
print(f'The algorithm was {precscore * 100:.3f}% correct!')
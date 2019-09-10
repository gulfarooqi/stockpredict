import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import math
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge

# set start and end date of data we want to collect
start = datetime.datetime(2010, 1 ,1)
end = datetime.datetime(2018, 1, 1)

df = web.DataReader("IBM", 'yahoo', start, end)

dfreg = df.loc[:, ['Adj Close', 'Volume']] # retreive all rows for columns Adj Close and Volume
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

# Pre-processing & Cross Validation of Data
# 1. Drop missing value
dfreg.fillna(value=-99999, inplace=True) # Drops all missing values from the dataset

# We want to seperate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out) # inserting label coulumn by shifting values ?#??? What is this doing
X = np.array(dfreg.drop(['label'], 1))

# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)

# Finally we want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:] # setting x to the sliced list of X for forcast out values - this is test set
X_train = X[:-forecast_out] # this is training set

# Separate label and identify it as y
y = np.array(dfreg['label'])
# y_test = y[-forecast_out:]
y_train = y[:-forecast_out]

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

# Ridge regression
clfridge = Ridge(alpha=2)
clfridge.fit(X_train, y_train)

# Kernel Ridge regression
clfkrr = KernelRidge(alpha=0.5)
clfkrr.fit(X_train, y_train)

X_test = X[1000:1800]
y_test = y[1000:1800]

confidenceknn = clfknn.score(X_test, y_test)
confidenceridge = clfridge.score(X_test, y_test)
confidencekrr = clfkrr.score(X_test, y_test)

winner_clf = max(confidenceknn, confidenceridge, confidencekrr)

print('Score for KNN confidence is', confidenceknn)
print('Score for Ridge Regression confidence is', confidenceridge)
print('Score for Kernel Ridge Regression confidence is', confidencekrr)
print('The highest score is', winner_clf)

forecast_set = clfknn.predict(X_lately)
dfreg['Forecast'] = np.nan

last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

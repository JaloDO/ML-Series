import pandas as pd
import quandl, math, datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

style.use('ggplot')
quandl.ApiConfig.api_key = "gpxHQLsfSPp1AyQy5Afj"

dt = quandl.get('WIKI/GOOGL')

"""
    SET OF DATA FROM ACTIVES VALUES & FINANCIAL TAXES


    Only select the meaningful features
"""

dt = dt[
    ['Adj. Open',
     'Adj. High',
     'Adj. Low',
     'Adj. Close',
     'Adj. Volume'
     ]]
##print('ORIGINAL DATA\n',dt.head())

dt['HL_PCT'] = (dt['Adj. High'] - dt['Adj. Low']) / dt['Adj. Low'] * 100.0
#HL_PCT is the percentage in high value minus low value divided by low value

dt['PCT_change'] = (dt['Adj. Close'] - dt['Adj. Open']) / dt['Adj. Open'] * 100.0
#PCT_change is the percentage in change between close value and open value

#NEW DATAFRAME, only with the meaningful features
dt = dt[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

##print('SELECTED DATA\n',dt.head())

forecast_col = 'Adj. Close'
dt.fillna(-99999, inplace=True)
#mi forecast out va a ser 34, que es el entero superior a la longitud
#de mi dataframe(3360aprox) por 0.01 = 34, (este valor es por que yo quiero)

##forecast_out = math.ceil(0.01*len(dt))
forecast_out = (10)
#en label lo unico que voy a mostrar es los valores de la columna 'Adj. Close'
#a partir de 34 dias en el futuro
dt['label'] = dt[forecast_col].shift(-forecast_out)
##adding following line, obtain better model
dt = dt.tail(200)
print(dt.tail(100))


"""
    DEFINE X-features & Y-labels to prepare data for classifier
"""
X = np.array(dt.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

dt.dropna(inplace=True)
y = np.array(dt['label'])


"""
    DEFINE A CLASSIFIER FOR TRAINNED DATA
"""
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1) #n_jobs = threads (-1 -> max)
##clf = svm.SVR(kernel='linear')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
print('Accuracy Ratio: ',accuracy)
print('Forecast Set: ',forecast_set)
print('Forecast Out: ',forecast_out)
pd.plotting.scatter_matrix(dt, alpha=0.4, figsize=(8, 8))
plt.show()


"""
    PLOT DATA in the GRAPH
"""
dt['Forecast'] = np.nan
last_date = dt.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    dt.loc[next_date] = [np.nan for _ in range(len(dt.columns)-1)] + [i]

dt['Adj. Close'].plot()
dt['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()





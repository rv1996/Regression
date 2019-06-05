import pandas as pd
import quandl,math,datetime
import numpy as np 
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
# out of the above column we can have many relationship derived colums like changes percentage, etc
df['HL_PCT'] = (df['Adj. High']-df['Adj. Low'])/df['Adj. Low']*100
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_column = "Adj. Close"

df.fillna(-99999,inplace=True) # filling the value which is not a number 
# the corresponding value will be treated as an outlier in the dataset

forecast_out = int(math.ceil(0.01*len(df))) # creating is to shift the data frame this much percentage upward..
# so that we have the featured predicting the price in future == that percent in future
# we don't actually have a label and to make a future prediction we are doing this

df['label'] = df[forecast_column].shift(-forecast_out) #creating a label by the shift the future value in future

print(df.head())


X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_predict = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

clf = LinearRegression()
clf.fit(X_train,y_train)
# printing the accuracy of the model we trained
print("using linear regression -",clf.score(X_test,y_test))

clfSVM = svm.SVR(gamma="auto")
clfSVM.fit(X_train,y_train)
print("using Support vector machine with linear kernel -",clfSVM.score(X_test,y_test))

clfSVMPoly = svm.SVR(kernel="poly",gamma="scale")
clfSVMPoly.fit(X_train,y_train)
print("using Support vector machine with polynomial kernel -",clfSVMPoly.score(X_test,y_test))

# clear svm doesn't work well in this case


prediction_set = clf.predict(X_predict)
print(prediction_set,clf.score(X_test,y_test),forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix =  last_date.timestamp()
oneday = 86400
next_unix = last_unix + oneday


for i in prediction_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += oneday
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1) ] + [i]


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel("date")
plt.ylabel("price")
plt.show()

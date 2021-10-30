# Import modules and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam


def buildmodel(dataset_train, cols, nfuture, npast):
    
    dataset_train = dataset_train[cols].astype(str)
    
#    for i in cols:
#        for j in range(0, len(dataset_train)):
#            dataset_train[i][j] = dataset_train[i][j].replace(',', '')
    
    dataset_train = dataset_train.astype(float)
    
    # Using multiple features (predictors)
    training_set = dataset_train.to_numpy()
    
    print('Shape of training set == {}.'.format(training_set.shape))
    
    # Feature Scaling
    
    
    sc = StandardScaler()
    training_set_scaled = sc.fit_transform(training_set)
    
    
    
    
    
    X_train = []
    y_train = []
    
    n_future = nfuture   # Number of days we want top predict into the future
    n_past = npast
        # Number of past days we want to use to predict the future
    
    for i in range(n_past, len(training_set_scaled) - n_future +1):
        
        X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1] - 1])
        #y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])
        y_train.append(training_set_scaled[i , 0])
        
    
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    print('X_train shape == {}.'.format(X_train.shape))
    print('y_train shape == {}.'.format(y_train.shape))
    
    #new model
    model = Sequential()
    model.add(LSTM(units = 60, return_sequences = True, input_shape=(n_past, dataset_train.shape[1]-1)))
    model.add(Dropout(0.2))
    
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 60, return_sequences = True))
    model.add(Dropout(0.2))
    
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 80, return_sequences = True))
    model.add(Dropout(0.2))
    
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 120))
    model.add(Dropout(0.2))
    
    # Adding the output layer
    model.add(Dense(units = 1))
    model.compile(optimizer = Adam(learning_rate=0.01), loss='mean_squared_error')
    
    model.fit(X_train, y_train, epochs = 100, batch_size = 32)
    
    return model, X_train, y_train

def datetime_to_timestamp(x):
    '''
        x : a given datetime value (datetime.date)
    '''
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


def evaluatemodel(dataframefull, cols, nfuture, npast, colTopredict=1):
    datasettrain, datasettest = train_test_split(dataframefull, test_size=0.05)
    datelist_train = list(datasettrain['Date'])
    datelist_train = [dt.datetime.strptime(date, '%m/%d/%Y').date() for date in datelist_train]
    
    model, X_train, y_train = buildmodel(datasettrain, cols, nfuture, npast)
    
    datelist_future = pd.date_range(datelist_train[-1], periods=datasettest.shape[0] - 1, freq='1d').tolist()
    datelist_future_ = []
    for this_timestamp in datelist_future:
        datelist_future_.append(this_timestamp.date())
        
    dataset_test = datasettest[cols].astype(str)
#    for i in cols:
#        for j in range(0, len(dataset_test)):
#            dataset_test[i][j] = dataset_test[i][j].replace(',', '')
    
    dataset_test = dataset_test.astype(float)
    
    # Using multiple features (predictors)
    test_set = dataset_test.to_numpy()
    sc_predict = StandardScaler()
    sc_predict.fit_transform(test_set[:, 0:colTopredict])
    
    testtempscaled = sc_predict.fit_transform(test_set)
    
    X_test = []
    y_test = []
    
    
    # Number of past days we want to use to predict the future
    
    for i in range(npast, len(testtempscaled) - nfuture +1):
        
        X_test.append(testtempscaled[i - n_past:i, 0:dataset_test.shape[1] - 1])
        #y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])
        y_test.append(testtempscaled[i , 0])
        
    
    
    X_test, y_test = np.array(X_test), np.array(y_test)
        
    predictions_future = model.predict(X_test)
    y_pred_future = sc_predict.inverse_transform(predictions_future)
    
    futuredf = pd.DataFrame({'Date': datelist_future, 'Open': list(y_pred_future)}, columns=['Date', 'Open'])
    
    real_stock_price = dataset_test.iloc[:, 1:2].values
    
    predicted_stock_price = futuredf.iloc[:, 1:2].values
    
    return real_stock_price, predicted_stock_price



dataset_full = pd.read_csv('C:\\stockmodel\\GOOG.csv')



# Select features (columns) to be involved intro training and predictions
cols = list(dataset_full)[1:6]
n_future = 39
n_past = 120

real_stock_price, predicted_stock_price = evaluatemodel(dataset_full, cols, n_future, n_past, 1)


plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



























































# Import modules and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
#from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard


# Importing Training Set


def buildmodel(dataset_train, cols, nfuture, npast):
    dataset_train = dataset_train[cols].astype(str)
    for i in cols:
        for j in range(0, len(dataset_train)):
            dataset_train[i][j] = dataset_train[i][j].replace(',', '')
    
    dataset_train = dataset_train.astype(float)
    
    # Using multiple features (predictors)
    training_set = dataset_train.to_numpy()
    
    print('Shape of training set == {}.'.format(training_set.shape))
    
    # Feature Scaling
    
    
    sc = StandardScaler()
    training_set_scaled = sc.fit_transform(training_set)
    
    sc_predict = StandardScaler()
    sc_predict.fit_transform(training_set[:, 0:1])
    
    
    
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
    
    model.fit(X_train, y_train, epochs = 5, batch_size = 32)
    
    return model, X_train, y_train, sc_predict
    
    #end newmodel

dataset_train = pd.read_csv('C:\\stockmodel\\GOOG.csv')

# Select features (columns) to be involved intro training and predictions
cols = list(dataset_train)[1:6]

# Extract dates (will be used in visualization)
datelist_train = list(dataset_train['Date'])
datelist_train = [dt.datetime.strptime(date, '%m/%d/%Y').date() for date in datelist_train]

print('Training set shape == {}'.format(dataset_train.shape))
print('All timestamps == {}'.format(len(datelist_train)))
print('Featured selected: {}'.format(cols))

n_future = 39
n_past = 120

model, X_train, y_train, sc_predict = buildmodel(dataset_train, cols, n_future, n_past)



# Generate list of sequence of days for predictions
datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1d').tolist()

'''
Remeber, we have datelist_train from begining.
'''

# Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE
datelist_future_ = []
for this_timestamp in datelist_future:
    datelist_future_.append(this_timestamp.date())
    
    
# Perform predictions
predictions_future = model.predict(X_train[-n_future:])

predictions_train = model.predict(X_train[n_past:])



# Inverse the predictions to original measurements

# ---> Special function: convert <datetime.date> to <Timestamp>
def datetime_to_timestamp(x):
    '''
        x : a given datetime value (datetime.date)
    '''
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


y_pred_future = sc_predict.inverse_transform(predictions_future)
y_pred_train = sc_predict.inverse_transform(predictions_train)

PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Open']).set_index(pd.Series(datelist_future))
PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Open']).set_index(pd.Series(datelist_train[2 * n_past + n_future -1:]))

# Convert <datetime.date> to <Timestamp> for PREDCITION_TRAIN
PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(datetime_to_timestamp)

PREDICTION_TRAIN.head(3)









futuredf = pd.DataFrame({'Date': datelist_future, 'Open': list(y_pred_future)}, columns=['Date', 'Open'])

dataset_test = pd.read_csv('C:\\stockmodel\\gtestdata.csv')

real_stock_price = dataset_test.iloc[:, 1:2].values

predicted_stock_price = futuredf.iloc[:, 1:2].values
print(type(real_stock_price))
print(type(predicted_stock_price))

predicted_stock_price.astype(np.float)


plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()































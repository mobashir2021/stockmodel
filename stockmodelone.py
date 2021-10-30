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


datasetfull = pd.read_csv('C:\\stockmodel\\GOOG.csv', parse_dates=True)
datasetfull.sort_values("Date")

# Select features (columns) to be involved intro training and predictions
cols = list(datasetfull)[1:6]
n_future = 33   # Number of days we want top predict into the future
n_past = 60

datasettrain, datasettest = train_test_split(datasetfull, test_size=0.01, shuffle=False)
datasettrain["Date"] = pd.to_datetime(datasettrain.Date)
datasettrain.sort_values("Date", inplace=True)
datasettest["Date"] = pd.to_datetime(datasettest.Date)
datasettest.sort_values("Date", inplace=True)

dataset_train = datasettrain[cols].astype(str)

training_set = dataset_train.to_numpy()
    
print('Shape of training set == {}.'.format(training_set.shape))

sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []


# Number of past days we want to use to predict the future
for i in range(n_past, training_set_scaled.shape[0]):
    X_train.append(training_set_scaled[i-n_past: i])
    y_train.append(training_set_scaled[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)
    
    
model = Sequential()
model.add(LSTM(units = 60, return_sequences = True, input_shape=(X_train.shape[1], X_train.shape[2])))
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
model.compile(optimizer = Adam(learning_rate=0.01), loss='mean_squared_error', metrics=['mean_absolute_percentage_error'])

history = model.fit(X_train, y_train, epochs = 1, batch_size = 32,verbose=1)
mape = history.history['mean_absolute_percentage_error']


past_60_days = dataset_train.tail(n_past)

df = past_60_days.append(datasettest, ignore_index = True)
df = df.drop(['Date'], axis = 1)

dataset_test = df[cols].astype(str)

test_set = dataset_test.to_numpy()
    
print('Shape of training set == {}.'.format(test_set.shape))

scpredict = StandardScaler()
test_set_scaled = scpredict.fit_transform(test_set)



X_test = []
y_test = []

for i in range(n_past, test_set_scaled.shape[0]):
    X_test.append(test_set_scaled[i-n_past: i])
    y_test.append(test_set_scaled[i, 0])
    
X_test, y_test = np.array(X_test), np.array(y_test)

y_pred = model.predict(X_test)

y_pred_copies = np.repeat(y_pred, X_train.shape[2], axis = -1)
y_pred_value = scpredict.inverse_transform(y_pred_copies)

dfarray = pd.DataFrame(y_pred_value, columns = ['A','B','C', 'D', 'E'])
stdvalue = dfarray['B'].std()

y_pred_value[0][0]


plt.plot(y_test, color = 'red', label = 'Real Google Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Mean Absolute Percentage Error (MAPE)
MAPE = np.mean((np.abs(np.subtract(y_test, y_pred)/ y_test))) * 100
print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(MAPE, 2)) + ' %')

# Median Absolute Percentage Error (MDAPE)
MDAPE = np.median((np.abs(np.subtract(y_test, y_pred)/ y_test)) ) * 100
print('Median Absolute Percentage Error (MDAPE): ' + str(np.round(MDAPE, 2)) + ' %')
    
    
    
    
    
    
    
    
    
    


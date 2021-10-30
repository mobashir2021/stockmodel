import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam


dataset_full = pd.read_csv('C:\\stockmodel\\GOOG.csv')

n_previous = 60

# Select features (columns) to be involved intro training and predictions


data_training, data_test = train_test_split(dataset_full, test_size=0.2)

training_data = data_training.drop(['Date'], axis = 1)

scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)

X_train = []
y_train = []

for i in range(n_previous, training_data.shape[0]):
    X_train.append(training_data[i-n_previous: i])
    y_train.append(training_data[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)


X_train.shape[1]
X_train.shape[2]

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
model.compile(optimizer = Adam(learning_rate=0.01), loss='mean_squared_error')

model.fit(X_train, y_train, epochs = 100, batch_size = 32)


past_60_days = data_training.tail(n_previous)

df = past_60_days.append(data_test, ignore_index = True)
df = df.drop(['Date'], axis = 1)

inputs = scaler.transform(df)

X_test = []
y_test = []

for i in range(n_previous, inputs.shape[0]):
    X_test.append(inputs[i-n_previous: i])
    y_test.append(inputs[i, 0])
    
X_test, y_test = np.array(X_test), np.array(y_test)

y_pred = model.predict(X_test)

#fullscaler = scaler.scale_
#scale = 1/fullscaler[0]
#
#y_pred = y_pred*scale
#y_test = y_test*scale

y_pred_copies = np.repeat(y_pred, inputs.shape[1], axis = -1)
y_pred = scaler.inverse_transform(y_pred_copies)
#y_pred[0]
#
real_stock_price = data_test.iloc[:, 1:2].values
futuredf = pd.DataFrame({ 'Open': list(y_pred[0])}, columns=[ 'Open'])
predicted_stock_price1 = futuredf.iloc[:, 0:1].values
predicted_stock_price1.astype(np.float)


plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price1, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

plt.plot(y_test, color = 'red', label = 'Real Google Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()














































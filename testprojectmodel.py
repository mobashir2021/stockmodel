import mysql.connector
from sqlalchemy import create_engine
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


db_connection_str = 'mysql+pymysql://devdb:Devdb@786#@3.136.40.143/Integration_DB'
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SELECT * FROM s_alpha', con=db_connection)

temp = df[df['stockid'] == 'COST']

dataset = temp[['date', 'open_value', 'close_value', 'low_value', 'high_value', 'volume']]
cols = list(dataset)[1:6]

datasettrain = dataset.copy()
datasettrain["date"] = pd.to_datetime(datasettrain.date)
datasettrain.sort_values("date", inplace=True)


dataset_train = datasettrain[cols].astype(str)

n_past = 60
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
model.compile(optimizer = Adam(learning_rate=0.01), loss='mean_squared_error')

model.fit(X_train, y_train, epochs = 10, batch_size = 32)


past_60_days = dataset_train.tail(n_past)

#df = past_60_days.append(datasettest, ignore_index = True)
df = past_60_days.copy()
#df = df.drop(['date'], axis = 1)

dataset_test = df[cols].astype(str)

test_set = dataset_test.to_numpy()
    
print('Shape of training set == {}.'.format(test_set.shape))

scpredict = StandardScaler()
test_set_scaled = scpredict.fit_transform(test_set)



X_test = []
y_test = []

X_test.append(np.asarray( test_set_scaled))
y_test.append(np.asarray(test_set_scaled[i, 0]))

#for i in range(n_past, test_set_scaled.shape[0]):
#    X_test.append(test_set_scaled[i-n_past: i])
#    y_test.append(test_set_scaled[i, 0])
    
X_test, y_test = np.array(X_test), np.array(y_test)

y_pred = model.predict(X_test)

y_pred_copies = np.repeat(y_pred, X_train.shape[2], axis = -1)
y_pred_value = scpredict.inverse_transform(y_pred_copies)
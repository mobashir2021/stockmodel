import mysql.connector
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import date, datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
import os
import math
import csv


#mydb = mysql.connector.connect(
#	host="3.136.40.143",
#	user="devdb",
#	password="Devdb@786#",
#	database="Integration_DB"
#	)
#cursor = mydb.cursor()
#cursor.execute("SELECT * FROM s_alpha")
#rows = cursor.fetchall()
#ij = 0
#for row in rows:
#    
#    if ij <= 5:
#        print(row)
#    ij = ij + 1
    
 
    
#num_fields = len(cursor.description)
#field_names = [i[0] for i in cursor.description]


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def predictNextValue(datasetfull, cols, datasettest, n_past = 60):
    datasetfullcopy = datasetfull.copy()
    datasettrain, datasettest = train_test_split(datasetfullcopy, test_size=0.05, shuffle=False)
    datasettrain["date"] = pd.to_datetime(datasettrain.date)
    datasettrain.sort_values("date", inplace=True)
    datasettest["date"] = pd.to_datetime(datasettest.date)
    datasettest.sort_values("date", inplace=True)

#    datasettrain["date"] = pd.to_datetime(datasettrain.date)
#    datasettrain.sort_values("date", inplace=True)
    
    
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
        y_train.append(training_set_scaled[i, 1]) #1 for High prediction
        
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
    
    history = model.fit(X_train, y_train, epochs = 100, batch_size = 32,verbose = 1)
    mape = history.history['mean_absolute_percentage_error']
    
    past_60_days = dataset_train.tail(n_past)
    
    df = past_60_days.append(datasettest, ignore_index = True)
    #df = past_60_days.copy()
    df = df.drop(['date'], axis = 1)
    
    dataset_test = df[cols].astype(str)
    
    test_set = dataset_test.to_numpy()
        
    print('Shape of training set == {}.'.format(test_set.shape))
    
    scpredict = StandardScaler()
    test_set_scaled = scpredict.fit_transform(test_set)
    
    
    
    X_test = []
    y_test = []
    
    for i in range(n_past, test_set_scaled.shape[0]):
        X_test.append(test_set_scaled[i-n_past: i])
        y_test.append(test_set_scaled[i, 1])
    
    #X_test.append(np.asarray( test_set_scaled))
   # y_test.append(np.asarray(test_set_scaled[i, 0]))
        
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    y_pred = model.predict(X_test)
    
    y_pred_copies = np.repeat(y_pred, X_train.shape[2], axis = -1)
    y_pred_value = scpredict.inverse_transform(y_pred_copies)
    
    dfarray = pd.DataFrame(y_pred_value, columns = ['A','B','C', 'D', 'E', 'F'])
    stdvalue = dfarray['D'].std()
    
    score = model.evaluate(X_test, y_test, verbose = 0) 
    
    #MAPE = mean_absolute_percentage_error(y_test, y_pred)
    
    
    #single day prediction
    past60Prediction = past_60_days.tail(n_past)
    dfpred = past60Prediction.copy()
    
    dataset_testpred = dfpred[cols].astype(str)
    
    test_setpred = dataset_testpred.to_numpy()
        
    print('Shape of training set == {}.'.format(test_setpred.shape))
    
    scpredictnew = StandardScaler()
    test_set_scaledpred = scpredictnew.fit_transform(test_setpred)
    
    
    
    X_testpred = []
    #y_testpred = []
    
#    for i in range(n_past, test_set_scaled.shape[0]):
#        X_test.append(test_set_scaled[i-n_past: i])
#        y_test.append(test_set_scaled[i, 0])
    
    X_testpred.append(np.asarray( test_set_scaledpred))
    #y_test.append(np.asarray(test_set_scaled[i, 0]))
        
    X_testpred = np.array(X_testpred)#, np.array(y_test)
    
    y_prednew = model.predict(X_testpred)
    
    y_pred_copiesnew = np.repeat(y_prednew, X_train.shape[2], axis = -1)
    y_pred_valuenew = scpredictnew.inverse_transform(y_pred_copiesnew)
    
    
    return y_pred_valuenew[0][0], score, mape

def sharpe_ratio(return_series, N, rf):
    mean = return_series.mean() * N -rf
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma

def sortino_ratio(series, N,rf):
    mean = series.mean() * N -rf
    std_neg = series[series<0].std()*np.sqrt(N)
    return mean/std_neg

def getCalculations(datasetfull):
    datasetfullcopy = datasetfull.copy()
    
    datasetfullcopy["date"] = pd.to_datetime(datasetfullcopy.date, errors='coerce')
    datasetfullcopy.sort_values("date", inplace=True)
    datasetfullcopy.set_index("date", inplace = True)
    oneyeardata = datasetfullcopy.tail(255)
    daily_returns = oneyeardata['close_value'].pct_change()
    monthly_returns = oneyeardata['close_value'].resample('M').ffill().pct_change()
    mean = monthly_returns.mean()
    stdvalue = monthly_returns.std()
    
    N = 255 #255 trading days in a year
    rf =0.01 #1% risk free rate
    
    sharpes = oneyeardata.apply(sharpe_ratio, args=(N,rf,),axis=0)
    
    sortinos = oneyeardata.apply(sortino_ratio, args=(N,rf,), axis=0)
    
    return mean, stdvalue, sharpes, sortinos
    


def getAlphaData():
    
    db_connection_str = 'mysql+pymysql://devdb:Devdb@786#@3.136.40.143/Integration_DB'
    db_connection = create_engine(db_connection_str)
    
    dffilter = pd.read_sql('SELECT * FROM s_alpha_table', con=db_connection)
    df = dffilter[dffilter['high_value'].notnull()]
    
    listofstocks = ['COST', 'DD', 'LOW', 'SEE', 'VIAC', 'WELL']
    listofstockname = ["Costco Wholesale Corp." "DuPont de Nemours Inc", "Lowe's Cos.", 
                       "Sealed Air", "ViacomCBS", "Welltower Inc."]
    
    x = dt.datetime.now()
    folder = x.strftime("%Y%m%d")
    folder1=x.strftime("%Y-%m-%d")
    
    symbol_folder = os.path.join('/home/ubuntu/AlphaData', folder)
    #symbol_folder = os.path.join('C:\\stockmodel\\data', folder)
    if not os.path.exists(symbol_folder):
    	os.mkdir(symbol_folder)
    listofvalues = []
    
    indexvalue = 0
    for stock in listofstocks:
        dfstock = df[df['stockid'] == stock]
        dataset = dfstock[['date', 'open_value', 'close_value', 'low_value', 'high_value', 'volume', 'wiki_pageviews']]
        cols = list(dataset)[1:7]
        score = []
        alpha = 0
        
        alpha , score, newmape  = predictNextValue(dataset, cols, [], 60)
        print(score)
        modelscore = 0
        sharperatio = 0
        sortinoratio = 0
        
        mean, stdvalue, sharpes, sortinos = getCalculations(dataset)
        sharperatio = sharpes[0]
        sortinoratio = sortinos[0]
        stdvalue = stdvalue * 100
        if math.isnan(sortinoratio):
            sortinoratio = 0
        
        y = date.today()
        tempdate = y.strftime("%d-%m-%Y")
        
	
        with open(symbol_folder + "/" + stock + ".csv", 'w') as dl:
            
            writer = csv.writer(dl)
            writer.writerow(["date", "Stockid", "predicted_value", "mape", "model_score", "stan_dev", "sharpe_ratio", "sortino_ratio"])
            writer.writerow([tempdate, stock, alpha, newmape[0], modelscore, stdvalue, sharperatio, sortinoratio])
        
    
        indexvalue = indexvalue + 1
    
    return [symbol_folder,folder1]

def mysql_insert():
    ingestion_start = dt1.now().strftime('%Y-%m-%d %H:%M:%S')
    mydb = mysql.connector.connect(
    host="3.136.40.143",
    user="devdb",
    password="Devdb@786#",
    database="Control_DB"
    )
    #print(mydb)
    mycursor = mydb.cursor()

    sql = "INSERT INTO JobStatus_CTb (sourceid,ingestion_start,ingestion_status) VALUES (%s,%s,%s)"
    val = ('FacebookData',str(ingestion_start),'RunningIngestion')
    mycursor.execute(sql, val)
    mydb.commit()
    return str(ingestion_start)


# In[11]:


def mysql_update(stock_up,mysql_ins):
    ingestion_end = dt1.now().strftime('%Y-%m-%d %H:%M:%S')
    mydb = mysql.connector.connect(
    host="3.136.40.143",
    user="devdb",
    password="Devdb@786#",
    database="Control_DB"
    )
    #print(mydb)
    mycursor = mydb.cursor()
    sql = 'UPDATE JobStatus_CTb SET location=%s, rundate= %s, ingestion_end= %s, ingestion_status=%s  where ingestion_start=%s'
    val = (stock_up[0], stock_up[1],str(ingestion_end),"LoadedToEC2",mysql_ins)
    mycursor.execute(sql,val)
    mydb.commit()


# In[13]:



mysql_ins = mysql_insert()
stock_up = getAlphaData()
mysql_update(stock_up,mysql_ins)

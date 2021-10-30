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
from sklearn.metrics import accuracy_score
from keras import backend as K

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

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


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
    
    model.compile(optimizer = Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])
    
    history = model.fit(X_train, y_train, epochs = 100, batch_size = 16,verbose = 1)
    mape = history.history['mean_squared_error']
    
    
    print('Mpae value')
    print(mape)
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
    
#    accuracydata = accuracy_score(y_pred, y_test)
#    accuracydata = model.score(X_test,y_test)
    print('calc accuracy')
    #print(accuracydata)
    
    
    y_pred_copies = np.repeat(y_pred, X_train.shape[2], axis = -1)
    y_pred_value = scpredict.inverse_transform(y_pred_copies)
    
    #dfarray = pd.DataFrame(y_pred_value, columns = ['A','B','C', 'D', 'E', 'F'])
    #stdvalue = dfarray['D'].std()
    
    score = model.evaluate(X_test, y_test, verbose = 0) 
    print('Score')
    print(score)
    
    
    #MAPE = mean_absolute_percentage_error(y_test, y_pred)
    
#    score = accuracydata
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
    mapevalue = mape[len(mape) - 1] * 100
    
    return y_pred_valuenew[0][0], score, mapevalue

def sharpe_ratio(return_series, N, rf):
    mean = return_series.mean() * N -rf
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma

def sortino_ratio(series, N,rf):
    mean = series.mean() * N -rf
    std_neg = series[series<0].std()*np.sqrt(N)
    return mean/std_neg

def daily_returns(prices):

    res = (prices/prices.shift(1) - 1.0)[1:]
    res.columns = ['return']

    return res

def sharpe(returns, risk_free=0):
    adj_returns = returns - risk_free
    return (np.nanmean(adj_returns) * np.sqrt(252)) / np.nanstd(adj_returns, ddof=1)
        
def downside_risk(returns, risk_free=0):
    adj_returns = returns - risk_free
    sqr_downside = np.square(np.clip(adj_returns, np.NINF, 0))
    return np.sqrt(np.nanmean(sqr_downside) * 252)


def sortino(returns, risk_free=0):
    adj_returns = returns - risk_free
    drisk = downside_risk(adj_returns)

    if drisk == 0:
        return np.nan

    return (np.nanmean(adj_returns) * np.sqrt(252)) / drisk

def getCalculations(datasetfull):
    datasetfullcopy = datasetfull.copy()
    datasetemp = datasetfull.copy()
    datasetemp["date"] = pd.to_datetime(datasetemp.date, errors='coerce')
    datasetemp.sort_values("date", inplace=True)
    datasetfullcopy["date"] = pd.to_datetime(datasetfullcopy.date, errors='coerce')
    datasetfullcopy.sort_values("date", inplace=True)
    datasetfullcopy.set_index("date", inplace = True)
    oneyeardata = datasetfullcopy.tail(255).copy()
    
#    daily_return = oneyeardata['close_value'].pct_change()
    monthly_returns = oneyeardata['close_value'].resample('M').ffill().pct_change()
    mean = monthly_returns.mean()
    stdvalue = monthly_returns.std()
    
    N = 255 #255 trading days in a year
    rf =0.01 #1% risk free rate
    
    sharpes = oneyeardata.apply(sharpe_ratio, args=(N,rf,),axis=0)
#    
#    sortinos = oneyeardata.apply(sortino_ratio, args=(N,rf,), axis=0)
    
    newdfcalc = datasetemp[['date', 'close_value']].tail(255).copy()
    
    newdfcalc.set_index("date", inplace = True)
    return1 = daily_returns(newdfcalc)
    
   
    sortinos = sortino(return1, rf)
    return mean, stdvalue, sharpes, sortinos

def roundup(x):
    return int(math.ceil(x / 100.0)) * 100
    


def getAlphaData():
    
    db_connection_str = 'mysql+pymysql://devdb:Devdb@786#@3.136.40.143/Integration_DB'
    db_connection = create_engine(db_connection_str)
    
    dffilter = pd.read_sql('SELECT * FROM s_alpha_table', con=db_connection)
    df = dffilter[dffilter['close_value'].notnull()]
    
    listofstocks = ['COST', 'DD', 'LOW', 'SEE', 'VIAC', 'WELL']
#    listofstocks = ['WELL']
    listofstockname = ["Costco Wholesale Corp." "DuPont de Nemours Inc", "Lowe's Cos.", 
                       "Sealed Air", "ViacomCBS", "Welltower Inc."]
    
    x = dt.datetime.now()
    folder = x.strftime("%Y%m%d")
    folder1=x.strftime("%Y-%m-%d")
    
    symbol_folder11 = os.path.join('/home/ubuntu/AlphaData', folder)
    symbol_folder = os.path.join('C:\\stockmodel\\data', folder)
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
        sortinoratio = sortinos
        stdvalue = stdvalue * 100
        
        modelscore = 100.0 - (score[0] * 100)
        if math.isnan(sortinoratio):
            sortinoratio = 0
        
        y = date.today()
        tempdate = y.strftime("%d-%m-%Y")
        
        finalmodelscore = newmape * 100
        if finalmodelscore > 100:
            finalmodelscore = roundup(finalmodelscore) - finalmodelscore
	
        with open(symbol_folder + "\\" + stock + ".csv", 'w') as dl:
            
            writer = csv.writer(dl)
            writer.writerow(["Stockid", "date", "predicted_value", "mape", "model_score", "stan_dev", "sharpe_ratio", "sortino_ratio"])
            writer.writerow([stock,tempdate, alpha, newmape, finalmodelscore, stdvalue, sharperatio, sortinoratio])
        
    
        indexvalue = indexvalue + 1
    
    return [symbol_folder11,folder1]

def mysql_insert():
    ingestion_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    mydb = mysql.connector.connect(
    host="3.136.40.143",
    user="devdb",
    password="Devdb@786#",
    database="Control_DB"
    )
    #print(mydb)
    mycursor = mydb.cursor()

    sql = "INSERT INTO JobStatus_CTb (sourceid,ingestion_start,ingestion_status) VALUES (%s,%s,%s)"
    val = ('AlphaData',str(ingestion_start),'RunningIngestion')
    mycursor.execute(sql, val)
    mydb.commit()
    return str(ingestion_start)


# In[11]:


def mysql_update(stock_up,mysql_ins):
    ingestion_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    mydb = mysql.connector.connect(
    host="3.136.40.143",
    user="devdb",
    password="Devdb@786#",
    database="Control_DB"
    )
    #print(mydb)
    mycursor = mydb.cursor()
    sql = 'UPDATE JobStatus_CTb SET location=%s, rundate= %s, ingestion_end= %s, ingestion_status=%s  where ingestion_start=%s'
    val = (stock_up[0], stock_up[1],str(ingestion_end),"LoadedToUbuntu",mysql_ins)
    mycursor.execute(sql,val)
    mydb.commit()


# In[13]:

#getAlphaData()

mysql_ins = mysql_insert()
print('mysql inserted ssuccesffuly')
stock_up = getAlphaData()
mysql_update(stock_up,mysql_ins)
print('mysql update ssuccesffuly')










    

    




































































































































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



db_connection_str = 'mysql+pymysql://devdb:Devdb@786#@3.136.40.143/Integration_DB'
db_connection = create_engine(db_connection_str)

dffilter = pd.read_sql('SELECT * FROM s_alpha_table', con=db_connection)
df = dffilter[dffilter['high_value'].notnull()]

listofstocks = ['COST']
dfstock = df[df['stockid'] == 'COST']
dataset = dfstock[['date',  'close_value']]

mean, stdvalue, sharpes, sortinos = getCalculations(dataset)

sharpes[0]
math.isnan(sortinos[0])
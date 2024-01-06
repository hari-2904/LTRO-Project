"""
@Project: LTRO Project - Demand Forecasting
@author: Harihareshwar Kumaravel
"""

# ///LIBRAIRIES///

import os
import pandas as pd 
import numpy as np
import statsmodels.tsa.statespace.sarimax as sarimax
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

# ///USER INPUTS///
d = '191' #Enter the point where the prediction has to be made. Make sure it is in 'str' format
m = 1 # m = 0,1,2,3 for BX,Verre,OM,Carton respectively

# ///OTHER VARIABLES///
p = 'BX'
q = 'Verre'
r = "OM"
s = "Carton"
c = []
loc = "input/Data_cleared.xlsx"  # Location where the cleaned data is present
loc = os.path.abspath(loc)
# ///FUNCTIONS///

def SelectPoints(point, m):
    df = pd.read_excel(loc , sheet_name= m)
    df = df[df['date'] >= '01-01-2018']
    df['n_point']= df['n_point'].astype(str)
    df = df[df["n_point"] == point]
    df = df.query('taux <= 1.0')
    df['taux'] = df['taux'].fillna(value = np.mean(df['taux']))
    df['date']=pd.to_datetime(df['date'])
    df['day'] = [i.day for i in df['date']]
    df['month'] = [i.month for i in df['date']]
    df['year'] = [i.year for i in df['date']]
    df['week'] = [i.week for i in df['date']]
    if m == 0:
        for i in range(10):
            c.append(p)
    elif m == 1:
        for i in range(10):
            c.append(q) 
    elif m == 1:
        for i in range(10):
            c.append(r) 
    else:
        for i in range(10):
            c.append(s)
    return df

#Spliting the data to evaluate the models using MSE(Mean Squared error)
def test_train_split(df):
    train_df = df['taux'].iloc[:round(len(df)*0.8)]
    test_df = df['taux'].iloc[round(len(df)*0.8):]
    train_df = pd.DataFrame(train_df)
    test_df = pd.DataFrame(test_df)
    return train_df,test_df

#Functions to evaluate various Time series forecasting models
def esEval(df):  #exponential smoothing evaluation. Returns RMSE value
    test_train_split(df)
    train_df = test_train_split(df)[0]
    test_df = test_train_split(df)[1]
    es_model = ExponentialSmoothing(train_df['taux'])
    es_model_fit = es_model.fit()
    # Make predictions on the test set
    es_predictions = es_model_fit.predict(start=len(train_df), end=len(train_df)+len(test_df)-1)
    es_mse =  mean_squared_error(test_df['taux'], es_predictions)
    es_mse = es_mse**0.5
    return es_mse

def arEval(df): #AutoRegressive Model evaluation. Returns RMSE value
    test_train_split(df)
    train_df = test_train_split(df)[0]
    test_df = test_train_split(df)[1]
    ar_model = ARIMA(train_df["taux"], order=(1,0,0))
    ar_model_fit = ar_model.fit()
    ar_predictions = ar_model_fit.predict(start=len(train_df), end=len(train_df)+len(test_df)-1, dynamic=False)
    ar = pd.DataFrame(ar_predictions)
    ar_mse = mean_squared_error(test_df['taux'],ar)
    ar_mse = ar_mse**0.5
    return ar_mse

def armaEval(df): #AutoRegressive Moving average model evaluation. Returns RMSE value
    test_train_split(df)
    train_df = test_train_split(df)[0]
    test_df = test_train_split(df)[1]
    arma_model = ARIMA(train_df["taux"], order=(1,0,1))
    arma_model_fit = arma_model.fit()
    arma_predictions = arma_model_fit.predict(start=len(train_df), end=len(train_df)+len(test_df)-1, dynamic=False)
    arma = pd.DataFrame(arma_predictions)
    arma_mse = mean_squared_error(test_df['taux'],arma)
    arma_mse = arma_mse**0.5
    return arma_mse

def arimaEval(df): #AutoRegressive Integrated Moving Average evaluation. Returns RMSE value
    test_train_split(df)
    train_df = test_train_split(df)[0]
    test_df = test_train_split(df)[1]
    arima_model = auto_arima(df['taux'],m=6,trace=False, seasonal=True) 
    #arima_model = auto_arima(df['taux'],trace=False)
    a =str(arima_model.df_model)
    b = order_extract(a)
    arima_model = ARIMA(train_df["taux"], order=b[0])
    arima_model_fit = arima_model.fit()
    arima_predictions = arima_model_fit.predict(start=len(train_df), end=len(train_df)+len(test_df)-1, dynamic=False)
    arima = pd.DataFrame(arima_predictions)
    arima_mse = mean_squared_error(test_df['taux'],arima)
    arima_mse = arima_mse**0.5
    return arima_mse

def order_extract(a : str): #Function to extract order and seasonal order which is the output of auto_arima function
    string = a
    order_start = string.index('order')
    order_start_paren = string.index('(', order_start)
    order_end_paren = string.index(')', order_start_paren)
    order_string = string[order_start_paren+1:order_end_paren]
    order = tuple(map(int, order_string.split(',')))
    seasonal_order_start = string.index('seasonal_order')
    seasonal_order_start_paren = string.index('(', seasonal_order_start)
    seasonal_order_end_paren = string.index(')', seasonal_order_start_paren)
    seasonal_order_string = string[seasonal_order_start_paren+1:seasonal_order_end_paren]
    seasonal_order = tuple(map(int, seasonal_order_string.split(',')))
    return order, seasonal_order

def sarimaEval(df): #Seasonal Auto Regressive Integrated Moving Average evaluation. Returns RMSE value
    test_train_split(df)
    train_df = test_train_split(df)[0]
    test_df = test_train_split(df)[1]
    arima_model = auto_arima(df['taux'],m=7,trace=False)
    a =str(arima_model.df_model)
    global b 
    b =  order_extract(a)
    model = sarimax.SARIMAX(train_df["taux"], order=b[0], seasonal_order=b[1])  #Results from "arima_model.summary()". Look at Model and fill it in here
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(train_df), end=len(train_df)+len(test_df)-1)
    sarima = pd.DataFrame(predictions)
    sarima_mse = mean_squared_error(test_df['taux'],sarima)
    sarima_mse = sarima_mse**0.5
    return sarima_mse

# Methods to forecast data
def esModel(df):  #exponential Smoothing model
    series =  df['taux'].to_list()
    alpha = 0.9
    n_pred = 10
    forecast = [series[0]] # initial forecast
    for i in range(1, len(series)):
        forecast.append((alpha * series[i] + (1 - alpha) * forecast[i-1]))
    for i in range(n_pred):
        forecast.append((alpha * forecast[-1] + (1 - alpha) * forecast[-2])) 
    pred = forecast[-10:]
    df1 = pd.DataFrame({'prediction': []})
    df2 = pd.DataFrame()
    a = df['n_point'].unique()[0]
    b = []
    df1['prediction'] = pred
    timestamp = df['date'].iloc[-1] + timedelta(days = n)
    for i in range(len(df1)):
        df2 = df2.append({'date':timestamp},ignore_index=True)
        timestamp = df2['date'].iloc[i] + timedelta(days=n)
    pred_df = pd.concat([df1, df2], axis=0, ignore_index=True)
    for i in range(len(df1)):
        pred_df['date'][i] = pred_df['date'][i+len(df1)]
    pred_df = pred_df.dropna()
    for i in range(len(df1)):
        b.append(a)
    pred_df.insert(2, 'n_point', b)
    pred_df.insert(3, 'waste_type', c)
    es = pred_df
    return es

def arModel(df):  #AutoRegressive Model
    model = ARIMA(df["taux"], order=(1,0,0))
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(df), end=len(df)+9, dynamic=False)
    df1 = pd.DataFrame({'prediction': []})
    df2 = pd.DataFrame()
    a = df['n_point'].unique()[0]
    b = []
    df1['prediction'] = predictions
    timestamp = df['date'].iloc[-1] + timedelta(days = n)
    for i in range(len(df1)):
        df2 = df2.append({'date':timestamp},ignore_index=True)
        timestamp = df2['date'].iloc[i] + timedelta(days=n)
    pred_df = pd.concat([df1, df2], axis=0, ignore_index=True)
    for i in range(len(df1)):
        pred_df['date'][i] = pred_df['date'][i+len(df1)]
    pred_df = pred_df.dropna()
    for i in range(len(df1)):
        b.append(a)
    pred_df.insert(2, 'n_point', b)
    pred_df.insert(3, 'waste_type', c)
    ar = pred_df
    return ar

def armaModel(df): #ARMA model
    model = ARIMA(df["taux"], order=(1,0,1))
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(df), end=len(df)+9, dynamic=False)
    df1 = pd.DataFrame({'prediction': []})
    df2 = pd.DataFrame()
    a = df['n_point'].unique()[0]
    b = []
    df1['prediction'] = predictions
    timestamp = df['date'].iloc[-1] + timedelta(days = n)
    for i in range(len(df1)):
        df2 = df2.append({'date':timestamp},ignore_index=True)
        timestamp = df2['date'].iloc[i] + timedelta(days=n)
    pred_df = pd.concat([df1, df2], axis=0, ignore_index=True)
    for i in range(len(df1)):
        pred_df['date'][i] = pred_df['date'][i+len(df1)]
    pred_df = pred_df.dropna()
    for i in range(len(df1)):
        b.append(a)
    pred_df.insert(2, 'n_point', b)
    pred_df.insert(3, 'waste_type', c)
    arma = pred_df
    return arma

def arimaModel(df):  #ARIMA Model
    arima_model = auto_arima(df['taux'],m=6,trace=False)
    a =str(arima_model.df_model)
    b = order_extract(a)
    model = ARIMA(df["taux"], order=b[0])
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(df), end=len(df)+9, dynamic=False)
    df1 = pd.DataFrame({'prediction': []})
    df2 = pd.DataFrame()
    a = df['n_point'].unique()[0]
    b = []
    # Make predictions using the fitted model
    df1['prediction'] = predictions
    timestamp = df['date'].iloc[-1] + timedelta(days = n)
    for i in range(len(df1)):
        df2 = df2.append({'date':timestamp},ignore_index=True)
        timestamp = df2['date'].iloc[i] + timedelta(days=n)
    pred_df = pd.concat([df1, df2], axis=0, ignore_index=True)
    for i in range(len(df1)):
        pred_df['date'][i] = pred_df['date'][i+len(df1)]
    pred_df = pred_df.dropna()
    for i in range(len(df1)):
        b.append(a)
    pred_df.insert(2, 'n_point', b)
    pred_df.insert(3, 'waste_type', c)
    arima = pred_df
    return arima

def sarimaModel(df):  #SARIMA Model
    arima_model = auto_arima(df['taux'],m=7,trace=False)
    a =str(arima_model.df_model)
    b = order_extract(a)
    model = sarimax.SARIMAX(df["taux"], order=b[0], seasonal_order=b[1])
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(df), end=len(df)+9, dynamic=False)
    df1 = pd.DataFrame({'prediction': []})
    df2 = pd.DataFrame()
    a = df['n_point'].unique()[0]
    b = []
    # Make predictions using the fitted model
    df1['prediction'] = predictions
    timestamp = df['date'].iloc[-1] + timedelta(days = n)
    for i in range(len(df1)):
        df2 = df2.append({'date':timestamp},ignore_index=True)
        timestamp = df2['date'].iloc[i] + timedelta(days=n)
    pred_df = pd.concat([df1, df2], axis=0, ignore_index=True)
    for i in range(len(df1)):
        pred_df['date'][i] = pred_df['date'][i+len(df1)]
    pred_df = pred_df.dropna()
    for i in range(len(df1)):
        b.append(a)
    pred_df.insert(2, 'n_point', b)
    pred_df.insert(3, 'waste_type', c)
    sarima = pred_df
    return sarima

def prediction(df):  # Evaluates the dataset on different models and returns the prediction data where RMSE is the lowest
    min_mse = min(esEval(df),arEval(df),armaEval(df),arimaEval(df),sarimaEval(df))
    print('RMSE using Exponential smoothing: ',esEval(df))
    print('RMSE using AR Model: ', arEval(df))
    print('RMSE using ARMA Model: ', armaEval(df))
    print('RMSE using ARIMA Model: ', arimaEval(df))
    print('RMSE using SARIMA Model: ', sarimaEval(df))
    if min_mse == esEval(df):
        print('minimun mse was found using Exponential Smoothing ')
        print('min_mse =',min_mse)
        print(esModel(df))
    elif min_mse == arEval(df):
        print('minimun mse was found using AR Model ')
        print('min_mse =',min_mse)
        print(arModel(df))
    elif min_mse == armaEval(df):
        print('minimun mse was found using ARMA Model ')
        print('min_mse =',min_mse)
        print(armaModel(df))
    elif min_mse == arimaEval(df):
        print('minimun mse was found using ARIMA Model ')
        print('min_mse =',min_mse)
        print(arimaModel(df))
    else:
        print('minimun mse was found using SARIMA Model ')
        print('min_mse =',min_mse)
        print(sarimaModel(df))

#m = 0,1,2,3 for BX,Verre,OM,Carton respectively
#Select the point to be forecasted and the type of waste. 
df = SelectPoints(d,m)  
df['days before last pick up'] = df['days before last pick up'].apply(pd.to_numeric, errors='coerce')
n = round(df['days before last pick up'].mean())
prediction(df)
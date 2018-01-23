# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:14:25 2017

@author: carl
"""
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm, time #helper libraries
stock = pd.read_csv('data/20150813/2330.csv', sep=',',header=None)
time = stock.values[:,0].astype(str)#時間
data0 = stock.values[:,1].astype(int)#成交價
data1 = stock.values[:,2].astype(int)#成交量
data2 = stock.values[:,3].astype(int)#累計成交量
data3 = stock.values[:,4].astype(str)#五檔買價
data4 = stock.values[:,5].astype(str)#五檔買量
data5 = stock.values[:,6].astype(str)#五檔賣價
data6 = stock.values[:,7].astype(str)#五檔賣量
 
datab,databa=[],[]
for st in data3:
    temp = st.strip(' ').split('_')
    temp2 = [float(x) for x in temp[:-1]] 
    datab.append(temp2)#五檔買價
for st in data4:
    temp = st.strip(' ').split('_')
    temp2 = [float(x) for x in temp[:-1]] 
    databa.append(temp2)#五檔買量
    
datas,datasa=[],[]
for st in data5:
    temp = st.strip(' ').split('_')
    temp2 = [float(x) for x in temp[:-1]] 
    datas.append(temp2)#五檔賣價
for st in data6:
    temp = st.strip(' ').split('_')
    temp2 = [float(x) for x in temp[:-1]] 
    datasa.append(temp2)#五檔賣量

#Step 2 Build Model
model = Sequential()

model.add(LSTM(
    input_dim=1,
    output_dim=50,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print('compilation time : ', time.time() - start)
model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=1,
    validation_split=0.05)
predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
lstm.plot_results_multiple(predictions, y_test, 50)
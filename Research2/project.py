# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:14:01 2018
@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import BatchNormalization

#read
data1 = pd.read_csv('BeijingPM20100101_20151231.csv')
data1 = data1[['PM_US Post', 'DEWP', 'HUMI','PRES', 'TEMP', 'Iws', 'precipitation']]
data1 = data1.dropna()
data1.columns = ['PM2.5','Dew Point','Humidity', 'Pressure', 'Temperature', 'Wind Speed', 'Precipitation']
X1,y1 = data1[['PM2.5','Dew Point','Humidity', 'Pressure', 'Temperature', 'Wind Speed', 'Precipitation']].values, data1['PM2.5']

data2 = pd.read_csv('ChengduPM20100101_20151231.csv')
data2 = data2[['PM_US Post', 'DEWP', 'HUMI','PRES', 'TEMP', 'Iws', 'precipitation']]
data2 = data2.dropna()
data2.columns = ['PM2.5','Dew Point','Humidity', 'Pressure', 'Temperature', 'Wind Speed', 'Precipitation']
X2,y2 = data2[['PM2.5','Dew Point','Humidity', 'Pressure', 'Temperature', 'Wind Speed', 'Precipitation']].values, data2['PM2.5']

data3 = pd.read_csv('ShenyangPM20100101_20151231.csv')
data3 = data3[['PM_US Post', 'DEWP', 'HUMI','PRES', 'TEMP', 'Iws', 'precipitation']]
data3 = data3.dropna()
data3.columns = ['PM2.5','Dew Point','Humidity', 'Pressure', 'Temperature', 'Wind Speed', 'Precipitation']
X3,y3 = data3[['PM2.5','Dew Point','Humidity', 'Pressure', 'Temperature', 'Wind Speed', 'Precipitation']].values, data3['PM2.5']
'''
fig = plt.figure(figsize = (10,4))
fig.subplots_adjust(top = 0.8, wspace = 0.5, hspace = 0.5)
for i in range(data1.shape[1]):
    ax = fig.add_subplot(2,4,i+1)
    #ax.hist(data1.iloc[:,i].values, color = 'red', bins=15, edgecolor = 'black', linewidth = 0.2, density = True, alpha = 0.2)
    #ax.hist(data2.iloc[:,i].values, color = 'blue', bins=15, edgecolor = 'black', linewidth = 0.2, density = True, alpha = 0.2)
    #ax.hist(data3.iloc[:,i].values, color = 'green', bins=15, edgecolor = 'black', linewidth = 0.2, density = True ,alpha = 0.2)  
    sns.kdeplot(data2.iloc[:,i].values, shade=True, color = 'green')
    sns.kdeplot(data1.iloc[:,i].values, shade=True, color = 'red') #legend=False, label = 'Beijing',
    sns.kdeplot(data3.iloc[:,i].values, shade=True, color = 'blue')
    ax.set_xlabel(data1.columns[i])
    ax.set_ylabel('density')
    if i == 0:
        ax.set_xlim([0,400])
    if i == 5:
        ax.set_xlim([0,20])  
        ax.set_ylim([0,0.08])
    if i == 6:
        ax.set_xlim([0,5])
        ax.set_ylim([0,0.018])
plt.savefig('PM2.5_3cities2.png',dpi=300)
'''
print(np.mean(data1.iloc[:,0].values))
print(np.std(data1.iloc[:,0].values))
print(np.mean(data2.iloc[:,0].values))
print(np.std(data2.iloc[:,0].values))
print(np.mean(data3.iloc[:,0].values))
print(np.std(data3.iloc[:,0].values))


def myscaler(data,scaler_name):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.externals import joblib
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    joblib.dump(scaler,scaler_name)
    return data_scaled

def get_inverse_scaler(data,scaler_name):
    from sklearn.externals import joblib
    scaler = joblib.load(scaler_name)
    data_inverse = scaler.inverse_transform(data)
    return data_inverse

def myRNN(X,y,mem = 30,nlyears = 3,nnodes = 10,activation_func = 'relu'):
    # data scaling
    X = myscaler(X, 'x_scaler')
    y = y.reshape(-1,1)
    y = myscaler(y, 'y_scaler')
    
    # create training data
    X_train = []
    y_train = []
    for i in range(mem, int(len(X)*0.8)):
        X_train.append(X[i-mem:i, :])
        y_train.append(y[i, 0])
    X_train,y_train = np.array(X_train), np.array(y_train)
    
    
    #build RNN
    RNN = Sequential()
    
    RNN.add(LSTM(nnodes,return_sequences = True,activation = activation_func,input_shape=(X_train.shape[1],X_train.shape[2])))
    #RNN.add(BatchNormalization())
    #RNN.add(Dropout(0.2))
    
    RNN.add(LSTM(nnodes,return_sequences = True,activation = activation_func))
    #RNN.add(BatchNormalization())
    #RNN.add(Dropout(0.2))
    
    RNN.add(LSTM(nnodes,return_sequences = True,activation = activation_func))
    #RNN.add(BatchNormalization())
    #RNN.add(Dropout(0.2))
    
    RNN.add(LSTM(nnodes,activation = activation_func))
    #RNN.add(BatchNormalization())
    #RNN.add(Dropout(0.2))
    
    RNN.add(Dense(output_dim = 1))
    
    
    # Compile
    RNN.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    # Fitting the RNN to the Training set
    RNN.fit(X_train, y_train, nb_epoch = 1, batch_size = 32)
    # batch_size 第一次训练数据量太大会抗争大，square n 开始试
    # nb_epoch：循环多少次（24开始试，200、1000...大了好但太慢）
    
    RNN.summary()
    print(X_train.shape,y_train.shape)
    
    
    #predict
    X_test = []
    real_PM = []
    for i in range(int(len(X)*0.8) + mem, len(X)):#前80%训练
        X_test.append(X[i-mem:i,:])
        real_PM.append(y[i,0])
    X_test,real_PM = np.array(X_test),np.array(real_PM)
    #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_PM = RNN.predict(X_test)
    predicted_PM = get_inverse_scaler(predicted_PM,'y_scaler')
    real_PM = real_PM.reshape(-1,1)
    real_PM = get_inverse_scaler(real_PM,'y_scaler')
    
    
    '''
    # create predicting data
    # data scaling
    X_predict = fit_myscaler(X_predict, 'x_predict_scaler')
    y_predict = X_predict.reshape(-1,1)
    y_predict = fit_myscaler(y_predict, 'y_predict_scaler')
    
    X_test = []
    real_PM = []
    for i in range(mem, len(X_predict)):#前80%训练
        X_test.append(X_predict[i-mem:i,:])
        real_PM.append(y_predict[i,0])
    X_test,real_PM = np.array(X_test),np.array(real_PM)
    #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    
    # predict
    predicted_PM = RNN.predict(X_test)
    predicted_PM = get_inverse_scaler(predicted_PM,'y_predict_scaler')
    real_PM = real_PM.reshape(-1,1)
    real_PM = get_inverse_scaler(real_PM,'y_predict_scaler')
    '''
    
    
    # Visualising the results
    fig = plt.figure(figsize = (10,7))
    fig.subplots_adjust(top=0.8, wspace = 0.5, hspace = 0.7)
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(real_PM[mem:,], color = 'red', label = 'Real PM2.5')
    plt.plot(predicted_PM, color = 'blue', label = 'Predicted PM2.5')
    plt.title('PM2.5 Prediction')
    plt.xlabel('Time')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.savefig('PM2.5.3cities.Figure2.png')
    
    fig = plt.figure(figsize = (10,7))
    fig.subplots_adjust(top=0.8, wspace = 0.5, hspace = 0.7)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(real_PM[mem:,], color = 'red', bins = 15, edgecolor = 'black', linewidth = 0.2, density = True, alpha = 0.3)
    ax.hist(predicted_PM, color = 'blue', bins = 15, edgecolor = 'black', linewidth = 0.2, density = True, alpha = 0.3)
    ax.set_xlim([0, 500])
    ax.set_xlabel('PM2.5')
    ax.set_ylabel('density')
    plt.legend()
    plt.savefig('PM2.5.3cities.Figure3.png')
    
    
    #return result
    
    

    
RNN1 = myRNN(X1,y1)
#RNN2 = myRNN(X2,y2)
#RNN3 = myRNN(X3,y3)





'''
for i in range(data1.shape[1]):
    print(data1.columns[i])
    print('mean:')
    print(data1.mean()[i])
    print(data2.mean()[i])
    print(data3.mean()[i])
    print('std:')
    print(data1.std()[i])
    print(data2.std()[i])
    print(data3.std()[i])
    print('\n')

# standardarization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X1)
X1_=scaler.transform(X1)


# Feature Scaling
# normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X1 = scaler.fit_transform(X1)

# k-fold
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
for train, test in kf.split(X1):
    print("%s %s" % (train, test))



## RNN
# Creating a data structure with 60 timesteps (memory) and 1 output
X_train = []
y_train = []
for i in range(60, 52584):
    X_train1.append(training_set_scaled[i-60:i, 0])
    y_train1.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

plt.plot(X_train)

# Reshaping
# matrix to tensor
# [batch size, timestep, input dimension]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages

# Initialising the RNN
my1stRNN = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
my1stRNN.add(LSTM(50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
my1stRNN.add(BatchNormalization())
#my1stRNN.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
my1stRNN.add(LSTM(50, return_sequences = True))
my1stRNN.add(BatchNormalization())
#my1stRNN.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
my1stRNN.add(LSTM(50, return_sequences = True))
my1stRNN.add(BatchNormalization())
#my1stRNN.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
my1stRNN.add(LSTM(50))
my1stRNN.add(BatchNormalization())
#my1stRNN.add(Dropout(0.2))

# Adding the output layer
my1stRNN.add(Dense(output_dim = 1))

# Compiling the RNN
my1stRNN.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
my1stRNN.fit(X_train, y_train, nb_epoch = 2, batch_size = 32)
# batch_size 第一次训练数据量太大会抗争大，square n 开始试
# nb_epoch：循环多少次（24开始试，200、1000...大了好但太慢）

my1stRNN.summary()
print(X_train.shape,y_train.shape)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
#real_stock_price = dataset_test.iloc[:, 1:2].values
real_stock_price = np.append(dataset_train.iloc[:, 1:2].values,  dataset_test.iloc[:, 1:2].values);

# Getting the predicted stock price of 2017
# dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = real_stock_price
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

plt.plot(inputs,c='red')
plt.plot(training_set_scaled,c='blue')
plt.show()

X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = my1stRNN.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price[60:,], color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

'''


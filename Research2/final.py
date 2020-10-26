# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import BatchNormalization
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# read in data
#
dataset1 = pd.read_csv('BeijingPM20100101_20151231.csv')
dataset1 = dataset1[['PM_US Post', 'DEWP', 'HUMI', 'PRES', 'TEMP', 'Iws', 'precipitation']]
dataset1 = dataset1.dropna()
dataset1.columns = ['PM2.5','Dew Point','Humidity', 'Pressure', 'Temperature', 'Wind Speed', 'Precipitation']
X, y  = dataset1[['PM2.5', 'Dew Point', 'Humidity', 'Pressure', 'Temperature', 'Wind Speed', 'Precipitation']].values, dataset1['PM2.5']

#
dataset2 = pd.read_csv('ChengduPM20100101_20151231.csv')
dataset2 = dataset2[['PM_US Post', 'DEWP', 'HUMI', 'PRES', 'TEMP', 'Iws', 'precipitation']]
dataset2 = dataset2.dropna()
dataset2.columns = ['PM2.5','Dew Point','Humidity', 'Pressure', 'Temperature', 'Wind Speed', 'Precipitation']
X2, y2  = dataset2[['PM2.5', 'Dew Point', 'Humidity', 'Pressure', 'Temperature', 'Wind Speed', 'Precipitation']].values, dataset2['PM2.5']

#
dataset3 = pd.read_csv('ShenyangPM20100101_20151231.csv')
dataset3 = dataset3[['PM_US Post', 'DEWP', 'HUMI', 'PRES', 'TEMP', 'Iws', 'precipitation']]
dataset3 = dataset3.dropna()
dataset3.columns = ['PM2.5','Dew Point','Humidity', 'Pressure', 'Temperature', 'Wind Speed', 'Precipitation']
X3, y3  = dataset3[['PM2.5', 'Dew Point', 'Humidity', 'Pressure', 'Temperature', 'Wind Speed', 'Precipitation']].values, dataset3['PM2.5']


fig = plt.figure(figsize = (20,8))
fig.subplots_adjust(top = 0.8, wspace = 0.25, hspace = 0.25)
for i in range(dataset1.shape[1]):
    ax = fig.add_subplot(2,4,i+1)
    #ax.hist(dataset1.iloc[:,i].values, color = 'red', bins=15, edgecolor = 'black', linewidth = 0.2, density = True, alpha = 0.2)
    #ax.hist(dataset2.iloc[:,i].values, color = 'blue', bins=15, edgecolor = 'black', linewidth = 0.2, density = True, alpha = 0.2)
    #ax.hist(dataset3.iloc[:,i].values, color = 'green', bins=15, edgecolor = 'black', linewidth = 0.2, density = True ,alpha = 0.2)  
    sns.kdeplot(dataset2.iloc[:,i].values, shade=True, color = 'darkseagreen', label = 'Chengdu')
    sns.kdeplot(dataset1.iloc[:,i].values, shade=True, color = 'lightcoral', label = 'Beijing') #legend=False, label = 'Beijing',
    sns.kdeplot(dataset3.iloc[:,i].values, shade=True, color = 'cornflowerblue', label = 'Shenyang')
    ax.set_xlabel(dataset1.columns[i])
    ax.set_ylabel('density')
    if i == 0:
        ax.set_xlim([0,400])
    if i == 3:
        ax.set_ylim([0,0.05])
    if i == 5:
        ax.set_xlim([0,20])  
        ax.set_ylim([0,0.18])
    if i == 6:
        ax.set_xlim([0,5])
        ax.set_ylim([0,0.025])
#plt.savefig('PM2.5_3cities2.png',dpi=300)

print(np.mean(dataset1.iloc[:,0].values))
print(np.std(dataset1.iloc[:,0].values))
print(np.mean(dataset2.iloc[:,0].values))
print(np.std(dataset2.iloc[:,0].values))
print(np.mean(dataset3.iloc[:,0].values))
print(np.std(dataset3.iloc[:,0].values))


def fit_myscaler(dataset, scaler_name):
    from sklearn.externals import joblib
    
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_scaled = scaler.fit_transform(dataset)
    joblib.dump(scaler, scaler_name)
    
    return  dataset_scaled

def get_inverse_scaler(dataset, scaler_name):
    from sklearn.externals import joblib
    
    scaler = joblib.load(scaler_name)
    dataset_inverse = scaler.inverse_transform(dataset)
    
    return dataset_inverse
    

def myRNN(X, y, X2, y2, X3, y3, mem=30, nbatches = 1000, nepochs = 50, nlayers = 3, nnodes = 10, activation_func = 'relu'):
    # data scaling
    X = fit_myscaler(X, 'x_scaler')
    y = y.values.reshape(-1,1)
    y = fit_myscaler(y, 'y_scaler')
    
    X2 = fit_myscaler(X2, 'x_scaler2')
    y2 = y2.values.reshape(-1,1)
    y2 = fit_myscaler(y2, 'y_scaler2')
    
    X3 = fit_myscaler(X3, 'x_scaler3')
    y3 = y3.values.reshape(-1,1)
    y3 = fit_myscaler(y3, 'y_scaler3')
    
    
    # create training data
    X_train = []
    y_train = []
    for i in range(mem, int(len(X)*0.8)):
        X_train.append(X[i-mem:i, :])
        y_train.append(y[i, 0])
    X_train,y_train = np.array(X_train), np.array(y_train)
    
    # build RNN    
    RNN = Sequential()
    RNN.add(LSTM(nnodes, return_sequences = True, activation = activation_func, input_shape = (X_train.shape[1], X_train.shape[2])))
    RNN.add(BatchNormalization())
    for i in range(nlayers - 2):
        RNN.add(LSTM(nnodes, return_sequences = True, activation = activation_func))
        RNN.add(BatchNormalization())
    #RNN.add(Dropout(0.2))
    RNN.add(LSTM(nnodes,activation = activation_func))

    RNN.add(Dense(output_dim = 1))
    
    # Compile
    RNN.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    # Fitting the RNN to the Training set
    RNN.fit(X_train, y_train, nb_epoch = nepochs, batch_size = nbatches)
    
    RNN.summary()
    print(X_train.shape,y_train.shape)
    
    #predict
    X_test = []
    y_test = []
    for i in range(int(len(X)*0.8) + mem, len(X)):#前80%训练
        X_test.append(X[i-mem:i,:])
        y_test.append(y[i,0])
    X_test,y_test = np.array(X_test),np.array(y_test)
    #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_predict = RNN.predict(X_test)
    y_predict = get_inverse_scaler(y_predict,'y_scaler')
    y_test = y_test.reshape(-1,1)
    y_test = get_inverse_scaler(y_test,'y_scaler')
    
    X_test2 = []
    y_test2 = []
    for i in range(mem, len(X2)):#前80%训练
        X_test2.append(X2[i-mem:i,:])
        y_test2.append(y2[i,0])
    X_test2,y_test2 = np.array(X_test2),np.array(y_test2)
    #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_predict2 = RNN.predict(X_test2)
    y_predict2 = get_inverse_scaler(y_predict2,'y_scaler2')
    y_test2 = y_test2.reshape(-1,1)
    y_test2 = get_inverse_scaler(y_test2,'y_scaler2')
    
    X_test3 = []
    y_test3 = []
    for i in range(mem, len(X3)):#前80%训练
        X_test3.append(X3[i-mem:i,:])
        y_test3.append(y3[i,0])
    X_test3,y_test3 = np.array(X_test3),np.array(y_test3)
    #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_predict3 = RNN.predict(X_test3)
    y_predict3 = get_inverse_scaler(y_predict3,'y_scaler3')
    y_test3 = y_test3.reshape(-1,1)
    y_test3 = get_inverse_scaler(y_test3,'y_scaler3')
    
    # return results
    return y_test,y_predict, y_test2,y_predict2,y_test3,y_predict3


'''
# experiment with best parameter
BJ_BJ_test_all = []
BJ_BJ_pred_all = []
BJ_CD_test_all = []
BJ_CD_pred_all = []
BJ_SY_test_all = []
BJ_SY_pred_all = []

CD_CD_test_all = []
CD_CD_pred_all = []
CD_BJ_test_all = []
CD_BJ_pred_all = []
CD_SY_test_all = []
CD_SY_pred_all = []

SY_SY_test_all = []
SY_SY_pred_all = []
SY_BJ_test_all = []
SY_BJ_pred_all = []
SY_CD_test_all = []
SY_CD_pred_all = []

i = 100 #batch
j = 80 #node
k = 50 #epoch

BJ_BJ_test, BJ_BJ_pred, BJ_CD_test, BJ_CD_pred, BJ_SY_test, BJ_SY_pred = myRNN(np.delete(X,3,axis=1),  y,  np.delete(X2,3,axis=1),y2, np.delete(X3,3,axis=1), y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
CD_CD_test, CD_CD_pred, CD_BJ_test, CD_BJ_pred, CD_SY_test, CD_SY_pred = myRNN(np.delete(X2,3,axis=1), y2, np.delete(X,3,axis=1),y,   np.delete(X3,3,axis=1), y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
SY_SY_test, SY_SY_pred, SY_BJ_test, SY_BJ_pred, SY_CD_test, SY_CD_pred = myRNN(np.delete(X3,3,axis=1), y3, np.delete(X,3,axis=1),y,   np.delete(X2,3,axis=1), y2, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')

BJ_BJ_test_all.append(BJ_BJ_test)
BJ_BJ_pred_all.append(BJ_BJ_pred)
BJ_CD_test_all.append(BJ_CD_test)
BJ_CD_pred_all.append(BJ_CD_pred)
BJ_SY_test_all.append(BJ_SY_test)
BJ_SY_pred_all.append(BJ_SY_pred)

CD_CD_test_all.append(CD_CD_test)
CD_CD_pred_all.append(CD_CD_pred)
CD_BJ_test_all.append(CD_BJ_test)
CD_BJ_pred_all.append(CD_BJ_pred)
CD_SY_test_all.append(CD_SY_test)
CD_SY_pred_all.append(CD_SY_pred)

SY_SY_test_all.append(SY_SY_test)
SY_SY_pred_all.append(SY_SY_pred)
SY_BJ_test_all.append(SY_BJ_test)
SY_BJ_pred_all.append(SY_BJ_pred)
SY_CD_test_all.append(SY_CD_test)
SY_CD_pred_all.append(SY_CD_pred)

np.savez('PM2.5_BJ_CD_SY_test1.npz', BJ_BJ_test_all = BJ_BJ_test_all,BJ_BJ_pred_all = BJ_BJ_pred_all,BJ_CD_test_all = BJ_CD_test_all,BJ_CD_pred_all = BJ_CD_pred_all,BJ_SY_test_all = BJ_SY_test_all,BJ_SY_pred_all = BJ_SY_pred_all,
         CD_CD_test_all = CD_CD_test_all,CD_CD_pred_all = CD_CD_pred_all,CD_BJ_test_all = CD_BJ_test_all,CD_BJ_pred_all = CD_BJ_pred_all,CD_SY_test_all = CD_SY_test_all,CD_SY_pred_all = CD_SY_pred_all,
         SY_SY_test_all = SY_SY_test_all,SY_SY_pred_all = SY_SY_pred_all,SY_BJ_test_all = SY_BJ_test_all,SY_BJ_pred_all = SY_BJ_pred_all,SY_CD_test_all = SY_CD_test_all,SY_CD_pred_all = SY_CD_pred_all)
'''


data = np.load('PM2.5_BJ_CD_SY_test1.npz')
BJ_BJ_test_all = data['BJ_BJ_test_all']
BJ_BJ_pred_all = data['BJ_BJ_pred_all']
BJ_CD_test_all = data['BJ_CD_test_all']
BJ_CD_pred_all = data['BJ_CD_pred_all']
BJ_SY_test_all = data['BJ_SY_test_all']
BJ_SY_pred_all = data['BJ_SY_pred_all']

CD_CD_test_all = data['CD_CD_test_all']
CD_CD_pred_all = data['CD_CD_pred_all']
CD_BJ_test_all = data['CD_BJ_test_all']
CD_BJ_pred_all = data['CD_BJ_pred_all']
CD_SY_test_all = data['CD_SY_test_all']
CD_SY_pred_all = data['CD_SY_pred_all']

SY_SY_test_all = data['SY_SY_test_all']
SY_SY_pred_all = data['SY_SY_pred_all']
SY_BJ_test_all = data['SY_BJ_test_all']
SY_BJ_pred_all = data['SY_BJ_pred_all']
SY_CD_test_all = data['SY_CD_test_all']
SY_CD_pred_all = data['SY_CD_pred_all']

sns.jointplot(BJ_BJ_test_all,BJ_BJ_pred_all,kind='reg',color='darkseagreen')
#plt.savefig('PM2.5_3cities.Figure_best.png',dpi=600)


mse = np.full([3,3],np.nan)

mse[0,0] = np.mean((BJ_BJ_test_all - BJ_BJ_pred_all)**2)
mse[0,1] = np.mean((BJ_CD_test_all - BJ_CD_pred_all)**2)
mse[0,2] = np.mean((BJ_SY_test_all - BJ_SY_pred_all)**2)
'''
mse[1,0] = np.mean((CD_CD_test_all - CD_CD_pred_all)**2)
mse[1,1] = np.mean((CD_BJ_test_all - CD_BJ_pred_all)**2)
mse[1,2] = np.mean((CD_SY_test_all - CD_SY_pred_all)**2)

mse[2,0] = np.mean((SY_SY_test_all - SY_SY_pred_all)**2)
mse[2,1] = np.mean((SY_BJ_test_all - SY_BJ_pred_all)**2)
mse[2,2] = np.mean((SY_CD_test_all - SY_CD_pred_all)**2)
'''
correlation = np.full([3,3],np.nan)

[correlation[0,0], pvalue] = pearsonr(BJ_BJ_test_all[0],BJ_BJ_pred_all[0])
[correlation[0,1], pvalue] = pearsonr(BJ_CD_test_all[0],BJ_CD_pred_all[0])
[correlation[0,2], pvalue] = pearsonr(BJ_SY_test_all[0],BJ_SY_pred_all[0])
'''
[correlation[1,0], pvalue] = pearsonr(CD_CD_test_all[0],CD_CD_pred_all[0])
[correlation[1,1], pvalue] = pearsonr(CD_BJ_test_all[0],CD_BJ_pred_all[0])
[correlation[1,2], pvalue] = pearsonr(CD_SY_test_all[0],CD_SY_pred_all[0])

[correlation[2,0], pvalue] = pearsonr(SY_SY_test_all[0],SY_SY_pred_all[0])
[correlation[2,1], pvalue] = pearsonr(SY_BJ_test_all[0],SY_BJ_pred_all[0])
[correlation[2,2], pvalue] = pearsonr(SY_CD_test_all[0],SY_CD_pred_all[0])
'''
# bar plot
fig,ax = plt.subplots(1,2,figsize=(15,10))
ind = [1,2,3]
width = 0.2
p1 = ax[0].bar([1], mse[0,0]/np.max(mse), width=0.2, color='r',alpha=0.3)
p2 = ax[0].bar(np.array([1]) + 0.2, mse[0,1]/np.max(mse),  width=0.2, color='g',alpha=0.3)
p3 = ax[0].bar(np.array([1]) + 2*width, mse[0,2]/np.max(mse),  width=0.2, color='b',alpha=0.3)
p4 = ax[0].bar([2], mse[1,0]/np.max(mse), width=0.2, color='g',alpha=0.3)
p5 = ax[0].bar(np.array([2]) + 0.2, mse[1,1]/np.max(mse),  width=0.2, color='r',alpha=0.3)
p6 = ax[0].bar(np.array([2]) + 2*width, mse[1,2]/np.max(mse),  width=0.2, color='b',alpha=0.3)
p4 = ax[0].bar([3], mse[2,0]/np.max(mse), width=0.2, color='b',alpha=0.3)
p5 = ax[0].bar(np.array([3]) + 0.2, mse[2,1]/np.max(mse),  width=0.2, color='r',alpha=0.3)
p6 = ax[0].bar(np.array([3]) + 2*width, mse[2,2]/np.max(mse),  width=0.2, color='g',alpha=0.3)
ax[0].set_title('Predicted normalzed error')
ax[0].set_xticks(np.array(ind) + width / 2)
ax[0].set_xticklabels(('train in BJ', 'train in CD', 'train in SY'), rotation=20)
ax[0].legend((p1[0], p2[0], p3[0]), ('test in BJ', 'test in CD', 'test in SY'))

width = 0.2
p1 = ax[1].bar([1], correlation[0,0], width=0.2, color='r',alpha=0.3)
p2 = ax[1].bar(np.array([1]) + 0.2, correlation[0,1],  width=0.2, color='g',alpha=0.3)
p3 = ax[1].bar(np.array([1]) + 2*width, correlation[0,2],  width=0.2, color='b',alpha=0.3)
p4 = ax[1].bar([2], correlation[1,0], width=0.2, color='g',alpha=0.3)
p5 = ax[1].bar(np.array([2]) + 0.2, correlation[1,1],  width=0.2, color='r',alpha=0.3)
p6 = ax[1].bar(np.array([2]) + 2*width, correlation[1,2],  width=0.2, color='b',alpha=0.3)
p7 = ax[1].bar([3], correlation[2,0], width=0.2, color='b',alpha=0.3)
p8 = ax[1].bar(np.array([3]) + 0.2, correlation[2,1],  width=0.2, color='r',alpha=0.3)
p9 = ax[1].bar(np.array([3]) + 2*width, correlation[2,2],  width=0.2, color='g',alpha=0.3)
ax[1].set_title('Pearson correlation')
ax[1].set_xticks(np.array(ind) + width / 2)
ax[1].set_xticklabels(('train in BJ', 'train in CD', 'train in SY'), rotation=20)
#ax[1].legend((p1[0], p2[0], p3[0]), ('->BJ', '->CD', '->SY'))

#plt.savefig('PM2.5_3cities.Figure_test.png',dpi=600)


'''
# experiment on envrionmental drivers
BJ_BJ_test_all = []
BJ_BJ_pred_all = []
BJ_CD_test_all = []
BJ_CD_pred_all = []
BJ_SY_test_all = []
BJ_SY_pred_all = []

CD_CD_test_all = []
CD_CD_pred_all = []
CD_BJ_test_all = []
CD_BJ_pred_all = []
CD_SY_test_all = []
CD_SY_pred_all = []

SY_SY_test_all = []
SY_SY_pred_all = []
SY_BJ_test_all = []
SY_BJ_pred_all = []
SY_CD_test_all = []
SY_CD_pred_all = []

i = 200 #batch
j = 30 #node
k = 50 #epoch
for l in [0,1,2,3,4,5,6]:
    if l == 0:
        BJ_BJ_test, BJ_BJ_pred, BJ_CD_test, BJ_CD_pred, BJ_SY_test, BJ_SY_pred = myRNN(X,  y,  X2,y2, X3, y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
        CD_CD_test, CD_CD_pred, CD_BJ_test, CD_BJ_pred, CD_SY_test, CD_SY_pred = myRNN(X2, y2, X,y,   X3, y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
        SY_SY_test, SY_SY_pred, SY_BJ_test, SY_BJ_pred, SY_CD_test, SY_CD_pred = myRNN(X3, y3, X,y,   X2, y2, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
    else:
        BJ_BJ_test, BJ_BJ_pred, BJ_CD_test, BJ_CD_pred, BJ_SY_test, BJ_SY_pred = myRNN(np.delete(X,l,axis=1),  y,  np.delete(X2,l,axis=1),y2, np.delete(X3,l,axis=1), y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
        CD_CD_test, CD_CD_pred, CD_BJ_test, CD_BJ_pred, CD_SY_test, CD_SY_pred = myRNN(np.delete(X2,l,axis=1), y2, np.delete(X,l,axis=1),y,   np.delete(X3,l,axis=1), y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
        SY_SY_test, SY_SY_pred, SY_BJ_test, SY_BJ_pred, SY_CD_test, SY_CD_pred = myRNN(np.delete(X3,l,axis=1), y3, np.delete(X,l,axis=1),y,   np.delete(X2,l,axis=1), y2, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
    BJ_BJ_test_all.append(BJ_BJ_test)
    BJ_BJ_pred_all.append(BJ_BJ_pred)
    BJ_CD_test_all.append(BJ_CD_test)
    BJ_CD_pred_all.append(BJ_CD_pred)
    BJ_SY_test_all.append(BJ_SY_test)
    BJ_SY_pred_all.append(BJ_SY_pred)
    
    CD_CD_test_all.append(CD_CD_test)
    CD_CD_pred_all.append(CD_CD_pred)
    CD_BJ_test_all.append(CD_BJ_test)
    CD_BJ_pred_all.append(CD_BJ_pred)
    CD_SY_test_all.append(CD_SY_test)
    CD_SY_pred_all.append(CD_SY_pred)
    
    SY_SY_test_all.append(SY_SY_test)
    SY_SY_pred_all.append(SY_SY_pred)
    SY_BJ_test_all.append(SY_BJ_test)
    SY_BJ_pred_all.append(SY_BJ_pred)
    SY_CD_test_all.append(SY_CD_test)
    SY_CD_pred_all.append(SY_CD_pred)
np.savez('PM2.5_BJ_CD_SY_best_param.npz', BJ_BJ_test_all = BJ_BJ_test_all,BJ_BJ_pred_all = BJ_BJ_pred_all,BJ_CD_test_all = BJ_CD_test_all,BJ_CD_pred_all = BJ_CD_pred_all,BJ_SY_test_all = BJ_SY_test_all,BJ_SY_pred_all = BJ_SY_pred_all,
         CD_CD_test_all = CD_CD_test_all,CD_CD_pred_all = CD_CD_pred_all,CD_BJ_test_all = CD_BJ_test_all,CD_BJ_pred_all = CD_BJ_pred_all,CD_SY_test_all = CD_SY_test_all,CD_SY_pred_all = CD_SY_pred_all,
         SY_SY_test_all = SY_SY_test_all,SY_SY_pred_all = SY_SY_pred_all,SY_BJ_test_all = SY_BJ_test_all,SY_BJ_pred_all = SY_BJ_pred_all,SY_CD_test_all = SY_CD_test_all,SY_CD_pred_all = SY_CD_pred_all)
'''


'''
# plot
data = np.load('PM2.5_BJ_CD_SY_driver.npz')
BJ_BJ_test_all = data['BJ_BJ_test_all']
BJ_BJ_pred_all = data['BJ_BJ_pred_all']
BJ_CD_test_all = data['BJ_CD_test_all']
BJ_CD_pred_all = data['BJ_CD_pred_all']
BJ_SY_test_all = data['BJ_SY_test_all']
BJ_SY_pred_all = data['BJ_SY_pred_all']

CD_CD_test_all = data['CD_CD_test_all']
CD_CD_pred_all = data['CD_CD_pred_all']
CD_BJ_test_all = data['CD_BJ_test_all']
CD_BJ_pred_all = data['CD_BJ_pred_all']
CD_SY_test_all = data['CD_SY_test_all']
CD_SY_pred_all = data['CD_SY_pred_all']

SY_SY_test_all = data['SY_SY_test_all']
SY_SY_pred_all = data['SY_SY_pred_all']
SY_BJ_test_all = data['SY_BJ_test_all']
SY_BJ_pred_all = data['SY_BJ_pred_all']
SY_CD_test_all = data['SY_CD_test_all']
SY_CD_pred_all = data['SY_CD_pred_all']

mse_BJ = np.full([7,3],np.nan)
mse_CD = np.full([7,3],np.nan)
mse_SY = np.full([7,3],np.nan)
for i in range(7):
    mse_BJ[i,0] = np.mean((BJ_BJ_test_all[i] - BJ_BJ_pred_all[i])**2)
    mse_BJ[i,1] = np.mean((BJ_CD_test_all[i] - BJ_CD_pred_all[i])**2)
    mse_BJ[i,2] = np.mean((BJ_SY_test_all[i] - BJ_SY_pred_all[i])**2)
    
    mse_CD[i,0] = np.mean((CD_CD_test_all[i] - CD_CD_pred_all[i])**2)
    mse_CD[i,1] = np.mean((CD_BJ_test_all[i] - CD_BJ_pred_all[i])**2)
    mse_CD[i,2] = np.mean((CD_SY_test_all[i] - CD_SY_pred_all[i])**2)
    
    mse_SY[i,0] = np.mean((SY_SY_test_all[i] - SY_SY_pred_all[i])**2)
    mse_SY[i,1] = np.mean((SY_BJ_test_all[i] - SY_BJ_pred_all[i])**2)
    mse_SY[i,2] = np.mean((SY_CD_test_all[i] - SY_CD_pred_all[i])**2)

correlation_BJ = np.full([7,3],np.nan)
correlation_CD = np.full([7,3],np.nan)
correlation_SY = np.full([7,3],np.nan)
for i in range(7):
    [correlation_BJ[i,0], pvalue] = pearsonr(BJ_BJ_test_all[i],BJ_BJ_pred_all[i])
    [correlation_BJ[i,1], pvalue] = pearsonr(BJ_CD_test_all[i],BJ_CD_pred_all[i])
    [correlation_BJ[i,2], pvalue] = pearsonr(BJ_SY_test_all[i],BJ_SY_pred_all[i])

    [correlation_CD[i,0], pvalue] = pearsonr(CD_CD_test_all[i],CD_CD_pred_all[i])
    [correlation_CD[i,1], pvalue] = pearsonr(CD_BJ_test_all[i],CD_BJ_pred_all[i])
    [correlation_CD[i,2], pvalue] = pearsonr(CD_SY_test_all[i],CD_SY_pred_all[i])

    [correlation_SY[i,0], pvalue] = pearsonr(SY_SY_test_all[i],SY_SY_pred_all[i])
    [correlation_SY[i,1], pvalue] = pearsonr(SY_BJ_test_all[i],SY_BJ_pred_all[i])
    [correlation_SY[i,2], pvalue] = pearsonr(SY_CD_test_all[i],SY_CD_pred_all[i])

# scatter plot
#fig = plt.figure(figsize=(10,5))
#fig.subplots_adjust(top = 0.8, wspace = 0.2, hspace = 0.2)
#fig.add_subplot(1,3,1)
#sns.jointplot(BJ_BJ_test_all[0],BJ_BJ_pred_all[0],kind='reg',color='green')
#fig.add_subplot(1,3,2)
#sns.jointplot(CD_CD_test_all[0],CD_CD_pred_all[0],kind='reg',color='green')
#fig.add_subplot(1,3,3)
#sns.jointplot(SY_SY_test_all[0],SY_SY_pred_all[0],kind='reg',color='green')



# bar plot
fig,ax = plt.subplots(2,3,figsize=(15,14))
ind = [1,2,3,4,5,6,7]
width = 0.2
p1 = ax[0,0].bar(ind, mse_BJ[:,0]/np.max(mse_BJ), width=0.2, color='r',alpha=0.4)
p2 = ax[0,0].bar(np.array(ind) + 0.2, mse_BJ[:,1]/np.max(mse_BJ),  width=0.2, color='g',alpha=0.4)
p3 = ax[0,0].bar(np.array(ind) + 2*width, mse_BJ[:,2]/np.max(mse_BJ),  width=0.2, color='b',alpha=0.4)
ax[0,0].set_title('Predicted normalzed error')
ax[0,0].set_xticks(np.array(ind) + width / 2)
ax[0,0].set_xticklabels(('all', 'remove Dew Point', 'remove Humidity', 'remove Pressure', 'remove Temperature', 'remove Wind Speed', 'remove Precipitation'),rotation=20)
ax[0,0].legend((p1[0], p2[0], p3[0]), ('BJ->BJ', 'BJ->CD', 'BJ->SY'))

ind = [1,2,3,4,5,6,7]
width = 0.2
p1 = ax[0,1].bar(ind, mse_CD[:,0]/np.max(mse_CD), width=0.2, color='g',alpha=0.4)
p2 = ax[0,1].bar(np.array(ind) + 0.2, mse_CD[:,1]/np.max(mse_CD),  width=0.2, color='r',alpha=0.4)
p3 = ax[0,1].bar(np.array(ind) + 2*width, mse_CD[:,2]/np.max(mse_CD),  width=0.2, color='b',alpha=0.4)
ax[0,1].set_title('Predicted normalzed error')
ax[0,1].set_xticks(np.array(ind) + width / 2)
ax[0,1].set_xticklabels(('all', 'remove Dew Point', 'remove Humidity', 'remove Pressure', 'remove Temperature', 'remove Wind Speed', 'remove Precipitation'),rotation=20)
ax[0,1].legend((p1[0], p2[0], p3[0]), ('CD->CD', 'CD->BJ', 'CD->SY'))

ind = [1,2,3,4,5,6,7]
width = 0.2
p1 = ax[0,2].bar(ind, mse_SY[:,0]/np.max(mse_SY), width=0.2, color='b',alpha=0.4)
p2 = ax[0,2].bar(np.array(ind) + 0.2, mse_SY[:,1]/np.max(mse_SY),  width=0.2, color='r',alpha=0.4)
p3 = ax[0,2].bar(np.array(ind) + 2*width, mse_SY[:,2]/np.max(mse_SY),  width=0.2, color='g',alpha=0.4)
ax[0,2].set_title('Predicted normalzed error')
ax[0,2].set_xticks(np.array(ind) + width / 2)
ax[0,2].set_xticklabels(('all', 'remove Dew Point', 'remove Humidity', 'remove Pressure', 'remove Temperature', 'remove Wind Speed', 'remove Precipitation'),rotation=20)
ax[0,2].legend((p1[0], p2[0], p3[0]), ('SY->SY', 'SY->BJ', 'SY->CD'))


ind = [1,2,3,4,5,6,7]
width = 0.2
p1 = ax[1,0].bar(ind, correlation_BJ[:,0], width=0.2, color='r',alpha=0.4)
p2 = ax[1,0].bar(np.array(ind) + 0.2, correlation_BJ[:,1],  width=0.2, color='g',alpha=0.4)
p3 = ax[1,0].bar(np.array(ind) + 2*width, correlation_BJ[:,2],  width=0.2, color='b',alpha=0.4)
ax[1,0].set_title('Pearson correlation')
ax[1,0].set_xticks(np.array(ind) + width / 2)
ax[1,0].set_xticklabels(('all', 'remove Dew Point', 'remove Humidity', 'remove Pressure', 'remove Temperature', 'remove Wind Speed', 'remove Precipitation'),rotation=20)
ax[1,0].legend((p1[0], p2[0], p3[0]), ('BJ->BJ', 'BJ->CD', 'BJ->SY'))

ind = [1,2,3,4,5,6,7]
width = 0.2
p1 = ax[1,1].bar(ind, correlation_CD[:,0], width=0.2, color='g',alpha=0.4)
p2 = ax[1,1].bar(np.array(ind) + 0.2, correlation_CD[:,1],  width=0.2, color='r',alpha=0.4)
p3 = ax[1,1].bar(np.array(ind) + 2*width, correlation_CD[:,2],  width=0.2, color='b',alpha=0.4)
ax[1,1].set_title('Pearson correlation')
ax[1,1].set_xticks(np.array(ind) + width / 2)
ax[1,1].set_xticklabels(('all', 'remove Dew Point', 'remove Humidity', 'remove Pressure', 'remove Temperature', 'remove Wind Speed', 'remove Precipitation'),rotation=20)
ax[1,1].legend((p1[0], p2[0], p3[0]), ('CD->CD', 'CD->BJ', 'CD->SY'))

ind = [1,2,3,4,5,6,7]
width = 0.2
p1 = ax[1,2].bar(ind, correlation_SY[:,0], width=0.2, color='b',alpha=0.4)
p2 = ax[1,2].bar(np.array(ind) + 0.2, correlation_SY[:,1],  width=0.2, color='r',alpha=0.4)
p3 = ax[1,2].bar(np.array(ind) + 2*width, correlation_SY[:,2],  width=0.2, color='g',alpha=0.4)
ax[1,2].set_title('Pearson correlation')
ax[1,2].set_xticks(np.array(ind) + width / 2)
ax[1,2].set_xticklabels(('all', 'remove Dew Point', 'remove Humidity', 'remove Pressure', 'remove Temperature', 'remove Wind Speed', 'remove Precipitation'),rotation=20)
ax[1,2].legend((p1[0], p2[0], p3[0]), ('SY->SY', 'SY->BJ', 'SY->CD'))

plt.savefig('PM2.5_3cities.Figure_drivers.png',dpi=600)
'''


'''
# experiment on nodes
BJ_BJ_test_all = []
BJ_BJ_pred_all = []
BJ_CD_test_all = []
BJ_CD_pred_all = []
BJ_SY_test_all = []
BJ_SY_pred_all = []

CD_CD_test_all = []
CD_CD_pred_all = []
CD_BJ_test_all = []
CD_BJ_pred_all = []
CD_SY_test_all = []
CD_SY_pred_all = []

SY_SY_test_all = []
SY_SY_pred_all = []
SY_BJ_test_all = []
SY_BJ_pred_all = []
SY_CD_test_all = []
SY_CD_pred_all = []

i = 5000 #batch
k = 50 #epoch
l = 0 
for j in [10,20,30,40,50]:
    if l == 0:
        BJ_BJ_test, BJ_BJ_pred, BJ_CD_test, BJ_CD_pred, BJ_SY_test, BJ_SY_pred = myRNN(X,  y,  X2,y2, X3, y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
        CD_CD_test, CD_CD_pred, CD_BJ_test, CD_BJ_pred, CD_SY_test, CD_SY_pred = myRNN(X2, y2, X,y,   X3, y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
        SY_SY_test, SY_SY_pred, SY_BJ_test, SY_BJ_pred, SY_CD_test, SY_CD_pred = myRNN(X3, y3, X,y,   X2, y2, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
    else:
        BJ_BJ_test, BJ_BJ_pred, BJ_CD_test, BJ_CD_pred, BJ_SY_test, BJ_SY_pred = myRNN(np.delete(X,l,axis=1),  y,  np.delete(X2,l,axis=1),y2, np.delete(X3,l,axis=1), y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
        CD_CD_test, CD_CD_pred, CD_BJ_test, CD_BJ_pred, CD_SY_test, CD_SY_pred = myRNN(np.delete(X2,l,axis=1), y2, np.delete(X,l,axis=1),y,   np.delete(X3,l,axis=1), y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
        SY_SY_test, SY_SY_pred, SY_BJ_test, SY_BJ_pred, SY_CD_test, SY_CD_pred = myRNN(np.delete(X3,l,axis=1), y3, np.delete(X,l,axis=1),y,   np.delete(X2,l,axis=1), y2, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
    BJ_BJ_test_all.append(BJ_BJ_test)
    BJ_BJ_pred_all.append(BJ_BJ_pred)
    BJ_CD_test_all.append(BJ_CD_test)
    BJ_CD_pred_all.append(BJ_CD_pred)
    BJ_SY_test_all.append(BJ_SY_test)
    BJ_SY_pred_all.append(BJ_SY_pred)
    
    CD_CD_test_all.append(CD_CD_test)
    CD_CD_pred_all.append(CD_CD_pred)
    CD_BJ_test_all.append(CD_BJ_test)
    CD_BJ_pred_all.append(CD_BJ_pred)
    CD_SY_test_all.append(CD_SY_test)
    CD_SY_pred_all.append(CD_SY_pred)
    
    SY_SY_test_all.append(SY_SY_test)
    SY_SY_pred_all.append(SY_SY_pred)
    SY_BJ_test_all.append(SY_BJ_test)
    SY_BJ_pred_all.append(SY_BJ_pred)
    SY_CD_test_all.append(SY_CD_test)
    SY_CD_pred_all.append(SY_CD_pred)
np.savez('PM2.5_BJ_CD_SY_mynode.npz', BJ_BJ_test_all = BJ_BJ_test_all,BJ_BJ_pred_all = BJ_BJ_pred_all,BJ_CD_test_all = BJ_CD_test_all,BJ_CD_pred_all = BJ_CD_pred_all,BJ_SY_test_all = BJ_SY_test_all,BJ_SY_pred_all = BJ_SY_pred_all,
         CD_CD_test_all = CD_CD_test_all,CD_CD_pred_all = CD_CD_pred_all,CD_BJ_test_all = CD_BJ_test_all,CD_BJ_pred_all = CD_BJ_pred_all,CD_SY_test_all = CD_SY_test_all,CD_SY_pred_all = CD_SY_pred_all,
         SY_SY_test_all = SY_SY_test_all,SY_SY_pred_all = SY_SY_pred_all,SY_BJ_test_all = SY_BJ_test_all,SY_BJ_pred_all = SY_BJ_pred_all,SY_CD_test_all = SY_CD_test_all,SY_CD_pred_all = SY_CD_pred_all)
'''

'''
# plot
data = np.load('PM2.5_BJ_CD_SY_mynode.npz')
BJ_BJ_test_all = data['BJ_BJ_test_all']
BJ_BJ_pred_all = data['BJ_BJ_pred_all']
BJ_CD_test_all = data['BJ_CD_test_all']
BJ_CD_pred_all = data['BJ_CD_pred_all']
BJ_SY_test_all = data['BJ_SY_test_all']
BJ_SY_pred_all = data['BJ_SY_pred_all']

CD_CD_test_all = data['CD_CD_test_all']
CD_CD_pred_all = data['CD_CD_pred_all']
CD_BJ_test_all = data['CD_BJ_test_all']
CD_BJ_pred_all = data['CD_BJ_pred_all']
CD_SY_test_all = data['CD_SY_test_all']
CD_SY_pred_all = data['CD_SY_pred_all']

SY_SY_test_all = data['SY_SY_test_all']
SY_SY_pred_all = data['SY_SY_pred_all']
SY_BJ_test_all = data['SY_BJ_test_all']
SY_BJ_pred_all = data['SY_BJ_pred_all']
SY_CD_test_all = data['SY_CD_test_all']
SY_CD_pred_all = data['SY_CD_pred_all']

mse_BJ = np.full([5,3],np.nan)
mse_CD = np.full([5,3],np.nan)
mse_SY = np.full([5,3],np.nan)
for i in range(5):
    mse_BJ[4-i,0] = np.mean((BJ_BJ_test_all[i] - BJ_BJ_pred_all[i])**2)
    mse_BJ[4-i,1] = np.mean((BJ_CD_test_all[i] - BJ_CD_pred_all[i])**2)
    mse_BJ[4-i,2] = np.mean((BJ_SY_test_all[i] - BJ_SY_pred_all[i])**2)
    
    mse_CD[4-i,0] = np.mean((CD_CD_test_all[i] - CD_CD_pred_all[i])**2)
    mse_CD[4-i,1] = np.mean((CD_BJ_test_all[i] - CD_BJ_pred_all[i])**2)
    mse_CD[4-i,2] = np.mean((CD_SY_test_all[i] - CD_SY_pred_all[i])**2)
    
    mse_SY[4-i,0] = np.mean((SY_SY_test_all[i] - SY_SY_pred_all[i])**2)
    mse_SY[4-i,1] = np.mean((SY_BJ_test_all[i] - SY_BJ_pred_all[i])**2)
    mse_SY[4-i,2] = np.mean((SY_CD_test_all[i] - SY_CD_pred_all[i])**2)

correlation_BJ = np.full([5,3],np.nan)
correlation_CD = np.full([5,3],np.nan)
correlation_SY = np.full([5,3],np.nan)
for i in range(5):
    [correlation_BJ[4-i,0], pvalue] = pearsonr(BJ_BJ_test_all[i],BJ_BJ_pred_all[i])
    [correlation_BJ[4-i,1], pvalue] = pearsonr(BJ_CD_test_all[i],BJ_CD_pred_all[i])
    [correlation_BJ[4-i,2], pvalue] = pearsonr(BJ_SY_test_all[i],BJ_SY_pred_all[i])

    [correlation_CD[4-i,0], pvalue] = pearsonr(CD_CD_test_all[i],CD_CD_pred_all[i])
    [correlation_CD[4-i,1], pvalue] = pearsonr(CD_BJ_test_all[i],CD_BJ_pred_all[i])
    [correlation_CD[4-i,2], pvalue] = pearsonr(CD_SY_test_all[i],CD_SY_pred_all[i])

    [correlation_SY[4-i,0], pvalue] = pearsonr(SY_SY_test_all[i],SY_SY_pred_all[i])
    [correlation_SY[4-i,1], pvalue] = pearsonr(SY_BJ_test_all[i],SY_BJ_pred_all[i])
    [correlation_SY[4-i,2], pvalue] = pearsonr(SY_CD_test_all[i],SY_CD_pred_all[i])


# bar plot
fig,ax = plt.subplots(2,3,figsize=(15,14))
ind = [1,2,3,4,5]
width = 0.2
p1 = ax[0,0].bar(ind, mse_BJ[:,0]/np.max(mse_BJ[:,:]), width=0.2, color='r',alpha=0.4)
p2 = ax[0,0].bar(np.array(ind) + 0.2, mse_BJ[:,1]/np.max(mse_BJ[:,:]),  width=0.2, color='b',alpha=0.4)
p3 = ax[0,0].bar(np.array(ind) + 2*width, mse_BJ[:,2]/np.max(mse_BJ[:,:]),  width=0.2, color='g',alpha=0.4)
ax[0,0].set_title('Predicted normalzed error')
ax[0,0].set_xticks(np.array(ind) + width / 2)
ax[0,0].set_xticklabels(('10', '20', '30', '40', '50'),rotation=00)
ax[0,0].set_xlabel('node per layer')
ax[0,0].legend((p1[0], p2[0], p3[0]), ('BJ->BJ', 'BJ->CD', 'BJ->SY'))

ind = [1,2,3,4,5]
width = 0.2
p1 = ax[0,1].bar(ind, mse_CD[:,0]/np.max(mse_CD[:,:]), width=0.2, color='g',alpha=0.4)
p2 = ax[0,1].bar(np.array(ind) + 0.2, mse_CD[:,1]/np.max(mse_CD[:,:]),  width=0.2, color='r',alpha=0.4)
p3 = ax[0,1].bar(np.array(ind) + 2*width, mse_CD[:,2]/np.max(mse_CD[:,:]),  width=0.2, color='b',alpha=0.4)
ax[0,1].set_title('Predicted normalzed error')
ax[0,1].set_xticks(np.array(ind) + width / 2)
ax[0,1].set_xticklabels(('10', '20', '30', '40', '50'),rotation=00)
ax[0,1].set_xlabel('node per layer')
ax[0,1].legend((p1[0], p2[0], p3[0]), ('CD->CD', 'CD->BJ', 'CD->SY'))

ind = [1,2,3,4,5]
width = 0.2
p1 = ax[0,2].bar(ind, mse_SY[:,0]/np.max(mse_SY[:,:]), width=0.2, color='b',alpha=0.4)
p2 = ax[0,2].bar(np.array(ind) + 0.2, mse_SY[:,1]/np.max(mse_SY[:,:]),  width=0.2, color='r',alpha=0.4)
p3 = ax[0,2].bar(np.array(ind) + 2*width, mse_SY[:,2]/np.max(mse_SY[:,:]),  width=0.2, color='g',alpha=0.4)
ax[0,2].set_title('Predicted normalzed error')
ax[0,2].set_xticks(np.array(ind) + width / 2)
ax[0,2].set_xticklabels(('10', '20', '30', '40', '50'),rotation=00)
ax[0,2].set_xlabel('node per layer')
ax[0,2].legend((p1[0], p2[0], p3[0]), ('SY->SY', 'SY->BJ', 'SY->CD'))


ind = [1,2,3,4,5]
width = 0.2
p1 = ax[1,0].bar(ind, correlation_BJ[:,0], width=0.2, color='r',alpha=0.4)
p2 = ax[1,0].bar(np.array(ind) + 0.2, correlation_BJ[:,1],  width=0.2, color='b',alpha=0.4)
p3 = ax[1,0].bar(np.array(ind) + 2*width, correlation_BJ[:,2],  width=0.2, color='g',alpha=0.4)
ax[1,0].set_title('Pearson correlation')
ax[1,0].set_xticks(np.array(ind) + width / 2)
ax[1,0].set_xticklabels(('10', '20', '30', '40', '50'),rotation=00)
ax[1,0].set_xlabel('node per layer')
ax[1,0].legend((p1[0], p2[0], p3[0]), ('BJ->BJ', 'BJ->CD', 'BJ->SY'))

ind = [1,2,3,4,5]
width = 0.2
p1 = ax[1,1].bar(ind, correlation_CD[:,0], width=0.2, color='g',alpha=0.4)
p2 = ax[1,1].bar(np.array(ind) + 0.2, correlation_CD[:,1],  width=0.2, color='r',alpha=0.4)
p3 = ax[1,1].bar(np.array(ind) + 2*width, correlation_CD[:,2],  width=0.2, color='b',alpha=0.4)
ax[1,1].set_title('Pearson correlation')
ax[1,1].set_xticks(np.array(ind) + width / 2)
ax[1,1].set_xticklabels(('10', '20', '30', '40', '50'),rotation=00)
ax[1,1].set_xlabel('node per layer')
ax[1,1].legend((p1[0], p2[0], p3[0]), ('CD->CD', 'CD->BJ', 'CD->SY'))

ind = [1,2,3,4,5]
width = 0.2
p1 = ax[1,2].bar(ind, correlation_SY[:,0], width=0.2, color='b',alpha=0.4)
p2 = ax[1,2].bar(np.array(ind) + 0.2, correlation_SY[:,1],  width=0.2, color='r',alpha=0.4)
p3 = ax[1,2].bar(np.array(ind) + 2*width, correlation_SY[:,2],  width=0.2, color='g',alpha=0.4)
ax[1,2].set_title('Pearson correlation')
ax[1,2].set_xticks(np.array(ind) + width / 2)
ax[1,2].set_xticklabels(('10', '20', '30', '40', '50'),rotation=00)
ax[1,2].set_xlabel('node per layer')
ax[1,2].legend((p1[0], p2[0], p3[0]), ('SY->SY', 'SY->BJ', 'SY->CD'))

plt.savefig('PM2.5_3cities.Figure_mynode.png',dpi=600)
'''

'''
# experiment on batchsize
BJ_BJ_test_all = []
BJ_BJ_pred_all = []
BJ_CD_test_all = []
BJ_CD_pred_all = []
BJ_SY_test_all = []
BJ_SY_pred_all = []

CD_CD_test_all = []
CD_CD_pred_all = []
CD_BJ_test_all = []
CD_BJ_pred_all = []
CD_SY_test_all = []
CD_SY_pred_all = []

SY_SY_test_all = []
SY_SY_pred_all = []
SY_BJ_test_all = []
SY_BJ_pred_all = []
SY_CD_test_all = []
SY_CD_pred_all = []

j = 10
k = 50
l=0
for i in [2000,5000,10000,20000,40000]:
    if l == 0:
        BJ_BJ_test, BJ_BJ_pred, BJ_CD_test, BJ_CD_pred, BJ_SY_test, BJ_SY_pred = myRNN(X,  y,  X2,y2, X3, y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
        CD_CD_test, CD_CD_pred, CD_BJ_test, CD_BJ_pred, CD_SY_test, CD_SY_pred = myRNN(X2, y2, X,y,   X3, y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
        SY_SY_test, SY_SY_pred, SY_BJ_test, SY_BJ_pred, SY_CD_test, SY_CD_pred = myRNN(X3, y3, X,y,   X2, y2, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
    else:
        BJ_BJ_test, BJ_BJ_pred, BJ_CD_test, BJ_CD_pred, BJ_SY_test, BJ_SY_pred = myRNN(np.delete(X,l,axis=1),  y,  np.delete(X2,l,axis=1),y2, np.delete(X3,l,axis=1), y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
        CD_CD_test, CD_CD_pred, CD_BJ_test, CD_BJ_pred, CD_SY_test, CD_SY_pred = myRNN(np.delete(X2,l,axis=1), y2, np.delete(X,l,axis=1),y,   np.delete(X3,l,axis=1), y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
        SY_SY_test, SY_SY_pred, SY_BJ_test, SY_BJ_pred, SY_CD_test, SY_CD_pred = myRNN(np.delete(X3,l,axis=1), y3, np.delete(X,l,axis=1),y,   np.delete(X2,l,axis=1), y2, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
    BJ_BJ_test_all.append(BJ_BJ_test)
    BJ_BJ_pred_all.append(BJ_BJ_pred)
    BJ_CD_test_all.append(BJ_CD_test)
    BJ_CD_pred_all.append(BJ_CD_pred)
    BJ_SY_test_all.append(BJ_SY_test)
    BJ_SY_pred_all.append(BJ_SY_pred)
    
    CD_CD_test_all.append(CD_CD_test)
    CD_CD_pred_all.append(CD_CD_pred)
    CD_BJ_test_all.append(CD_BJ_test)
    CD_BJ_pred_all.append(CD_BJ_pred)
    CD_SY_test_all.append(CD_SY_test)
    CD_SY_pred_all.append(CD_SY_pred)
    
    SY_SY_test_all.append(SY_SY_test)
    SY_SY_pred_all.append(SY_SY_pred)
    SY_BJ_test_all.append(SY_BJ_test)
    SY_BJ_pred_all.append(SY_BJ_pred)
    SY_CD_test_all.append(SY_CD_test)
    SY_CD_pred_all.append(SY_CD_pred)
np.savez('PM2.5_BJ_CD_SY_batch.npz', BJ_BJ_test_all = BJ_BJ_test_all,BJ_BJ_pred_all = BJ_BJ_pred_all,BJ_CD_test_all = BJ_CD_test_all,BJ_CD_pred_all = BJ_CD_pred_all,BJ_SY_test_all = BJ_SY_test_all,BJ_SY_pred_all = BJ_SY_pred_all,
         CD_CD_test_all = CD_CD_test_all,CD_CD_pred_all = CD_CD_pred_all,CD_BJ_test_all = CD_BJ_test_all,CD_BJ_pred_all = CD_BJ_pred_all,CD_SY_test_all = CD_SY_test_all,CD_SY_pred_all = CD_SY_pred_all,
         SY_SY_test_all = SY_SY_test_all,SY_SY_pred_all = SY_SY_pred_all,SY_BJ_test_all = SY_BJ_test_all,SY_BJ_pred_all = SY_BJ_pred_all,SY_CD_test_all = SY_CD_test_all,SY_CD_pred_all = SY_CD_pred_all)
'''

'''
# plot
data = np.load('PM2.5_BJ_CD_SY_mybatch.npz')
BJ_BJ_test_all = data['BJ_BJ_test_all']
BJ_BJ_pred_all = data['BJ_BJ_pred_all']
BJ_CD_test_all = data['BJ_CD_test_all']
BJ_CD_pred_all = data['BJ_CD_pred_all']
BJ_SY_test_all = data['BJ_SY_test_all']
BJ_SY_pred_all = data['BJ_SY_pred_all']

CD_CD_test_all = data['CD_CD_test_all']
CD_CD_pred_all = data['CD_CD_pred_all']
CD_BJ_test_all = data['CD_BJ_test_all']
CD_BJ_pred_all = data['CD_BJ_pred_all']
CD_SY_test_all = data['CD_SY_test_all']
CD_SY_pred_all = data['CD_SY_pred_all']

SY_SY_test_all = data['SY_SY_test_all']
SY_SY_pred_all = data['SY_SY_pred_all']
SY_BJ_test_all = data['SY_BJ_test_all']
SY_BJ_pred_all = data['SY_BJ_pred_all']
SY_CD_test_all = data['SY_CD_test_all']
SY_CD_pred_all = data['SY_CD_pred_all']

mse_BJ = np.full([6,3],np.nan)
mse_CD = np.full([6,3],np.nan)
mse_SY = np.full([6,3],np.nan)
for i in range(6):
    mse_BJ[i,0] = np.mean((BJ_BJ_test_all[i] - BJ_BJ_pred_all[i])**2)
    mse_BJ[i,1] = np.mean((BJ_CD_test_all[i] - BJ_CD_pred_all[i])**2)
    mse_BJ[i,2] = np.mean((BJ_SY_test_all[i] - BJ_SY_pred_all[i])**2)
    
    mse_CD[i,0] = np.mean((CD_CD_test_all[i] - CD_CD_pred_all[i])**2)
    mse_CD[i,1] = np.mean((CD_BJ_test_all[i] - CD_BJ_pred_all[i])**2)
    mse_CD[i,2] = np.mean((CD_SY_test_all[i] - CD_SY_pred_all[i])**2)
    
    mse_SY[i,0] = np.mean((SY_SY_test_all[i] - SY_SY_pred_all[i])**2)
    mse_SY[i,1] = np.mean((SY_BJ_test_all[i] - SY_BJ_pred_all[i])**2)
    mse_SY[i,2] = np.mean((SY_CD_test_all[i] - SY_CD_pred_all[i])**2)

correlation_BJ = np.full([6,3],np.nan)
correlation_CD = np.full([6,3],np.nan)
correlation_SY = np.full([6,3],np.nan)
for i in range(6):
    [correlation_BJ[i,0], pvalue] = pearsonr(BJ_BJ_test_all[i],BJ_BJ_pred_all[i])
    [correlation_BJ[i,1], pvalue] = pearsonr(BJ_CD_test_all[i],BJ_CD_pred_all[i])
    [correlation_BJ[i,2], pvalue] = pearsonr(BJ_SY_test_all[i],BJ_SY_pred_all[i])

    [correlation_CD[i,0], pvalue] = pearsonr(CD_CD_test_all[i],CD_CD_pred_all[i])
    [correlation_CD[i,1], pvalue] = pearsonr(CD_BJ_test_all[i],CD_BJ_pred_all[i])
    [correlation_CD[i,2], pvalue] = pearsonr(CD_SY_test_all[i],CD_SY_pred_all[i])

    [correlation_SY[i,0], pvalue] = pearsonr(SY_SY_test_all[i],SY_SY_pred_all[i])
    [correlation_SY[i,1], pvalue] = pearsonr(SY_BJ_test_all[i],SY_BJ_pred_all[i])
    [correlation_SY[i,2], pvalue] = pearsonr(SY_CD_test_all[i],SY_CD_pred_all[i])

# bar plot
fig,ax = plt.subplots(2,3,figsize=(15,14))
ind = [1,2,3,4,5,6]
width = 0.2
p1 = ax[0,0].bar(ind, mse_BJ[:,0]/np.max(mse_BJ[:,:]), width=0.2, color='r',alpha=0.4)
p2 = ax[0,0].bar(np.array(ind) + 0.2, mse_BJ[:,1]/np.max(mse_BJ[:,:]),  width=0.2, color='b',alpha=0.4)
p3 = ax[0,0].bar(np.array(ind) + 2*width, mse_BJ[:,2]/np.max(mse_BJ[:,:]),  width=0.2, color='g',alpha=0.4)
ax[0,0].set_title('Predicted normalzed error')
ax[0,0].set_xticks(np.array(ind) + width / 2)
ax[0,0].set_xticklabels(('100', '200', '1000', '2000', '5000', '10000'),rotation=00)
ax[0,0].set_xlabel('batch size')
ax[0,0].legend((p1[0], p2[0], p3[0]), ('BJ->BJ', 'BJ->CD', 'BJ->SY'))

ind = [1,2,3,4,5,6]
width = 0.2
p1 = ax[0,1].bar(ind, mse_CD[:,0]/np.max(mse_CD[:,:]), width=0.2, color='g',alpha=0.4)
p2 = ax[0,1].bar(np.array(ind) + 0.2, mse_CD[:,1]/np.max(mse_CD[:,:]),  width=0.2, color='r',alpha=0.4)
p3 = ax[0,1].bar(np.array(ind) + 2*width, mse_CD[:,2]/np.max(mse_CD[:,:]),  width=0.2, color='b',alpha=0.4)
ax[0,1].set_title('Predicted normalzed error')
ax[0,1].set_xticks(np.array(ind) + width / 2)
ax[0,1].set_xticklabels(('100', '200', '1000', '2000', '5000', '10000'),rotation=00)
ax[0,1].set_xlabel('batch size')
ax[0,1].legend((p1[0], p2[0], p3[0]), ('CD->CD', 'CD->BJ', 'CD->SY'))

ind = [1,2,3,4,5,6]
width = 0.2
p1 = ax[0,2].bar(ind, mse_SY[:,0]/np.max(mse_SY[:,:]), width=0.2, color='b',alpha=0.4)
p2 = ax[0,2].bar(np.array(ind) + 0.2, mse_SY[:,1]/np.max(mse_SY[:,:]),  width=0.2, color='r',alpha=0.4)
p3 = ax[0,2].bar(np.array(ind) + 2*width, mse_SY[:,2]/np.max(mse_SY[:,:]),  width=0.2, color='g',alpha=0.4)
ax[0,2].set_title('Predicted normalzed error')
ax[0,2].set_xticks(np.array(ind) + width / 2)
ax[0,2].set_xticklabels(('100', '200', '1000', '2000', '5000', '10000'),rotation=00)
ax[0,2].set_xlabel('batch size')
ax[0,2].legend((p1[0], p2[0], p3[0]), ('SY->SY', 'SY->BJ', 'SY->CD'))


ind = [1,2,3,4,5,6]
width = 0.2
p1 = ax[1,0].bar(ind, correlation_BJ[:,0], width=0.2, color='r',alpha=0.4)
p2 = ax[1,0].bar(np.array(ind) + 0.2, correlation_BJ[:,1],  width=0.2, color='b',alpha=0.4)
p3 = ax[1,0].bar(np.array(ind) + 2*width, correlation_BJ[:,2],  width=0.2, color='g',alpha=0.4)
ax[1,0].set_title('Pearson correlation')
ax[1,0].set_xticks(np.array(ind) + width / 2)
ax[1,0].set_xticklabels(('100', '200', '1000', '2000', '5000', '10000'),rotation=00)
ax[1,0].set_xlabel('batch size')
ax[1,0].legend((p1[0], p2[0], p3[0]), ('BJ->BJ', 'BJ->CD', 'BJ->SY'))

ind = [1,2,3,4,5,6]
width = 0.2
p1 = ax[1,1].bar(ind, correlation_CD[:,0], width=0.2, color='g',alpha=0.4)
p2 = ax[1,1].bar(np.array(ind) + 0.2, correlation_CD[:,1],  width=0.2, color='r',alpha=0.4)
p3 = ax[1,1].bar(np.array(ind) + 2*width, correlation_CD[:,2],  width=0.2, color='b',alpha=0.4)
ax[1,1].set_title('Pearson correlation')
ax[1,1].set_xticks(np.array(ind) + width / 2)
ax[1,1].set_xticklabels(('100', '200', '1000', '2000', '5000', '10000'),rotation=00)
ax[1,1].set_xlabel('batch size')
ax[1,1].legend((p1[0], p2[0], p3[0]), ('CD->CD', 'CD->BJ', 'CD->SY'))

ind = [1,2,3,4,5,6]
width = 0.2
p1 = ax[1,2].bar(ind, correlation_SY[:,0], width=0.2, color='b',alpha=0.4)
p2 = ax[1,2].bar(np.array(ind) + 0.2, correlation_SY[:,1],  width=0.2, color='r',alpha=0.4)
p3 = ax[1,2].bar(np.array(ind) + 2*width, correlation_SY[:,2],  width=0.2, color='g',alpha=0.4)
ax[1,2].set_title('Pearson correlation')
ax[1,2].set_xticks(np.array(ind) + width / 2)
ax[1,2].set_xticklabels(('100', '200', '1000', '2000', '5000', '10000'),rotation=00)
ax[1,2].set_xlabel('batch size')
ax[1,2].legend((p1[0], p2[0], p3[0]), ('SY->SY', 'SY->BJ', 'SY->CD'))

plt.savefig('PM2.5_3cities.Figure_mybatch.png',dpi=600)
'''

'''
# experiment on epoch
BJ_BJ_test_all = []
BJ_BJ_pred_all = []
BJ_CD_test_all = []
BJ_CD_pred_all = []
BJ_SY_test_all = []
BJ_SY_pred_all = []

CD_CD_test_all = []
CD_CD_pred_all = []
CD_BJ_test_all = []
CD_BJ_pred_all = []
CD_SY_test_all = []
CD_SY_pred_all = []

SY_SY_test_all = []
SY_SY_pred_all = []
SY_BJ_test_all = []
SY_BJ_pred_all = []
SY_CD_test_all = []
SY_CD_pred_all = []

i = 10000
j = 10
l=0
for k in [10,50,100,150,200]:
    if l == 0:
        BJ_BJ_test, BJ_BJ_pred, BJ_CD_test, BJ_CD_pred, BJ_SY_test, BJ_SY_pred = myRNN(X,  y,  X2,y2, X3, y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
        CD_CD_test, CD_CD_pred, CD_BJ_test, CD_BJ_pred, CD_SY_test, CD_SY_pred = myRNN(X2, y2, X,y,   X3, y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
        SY_SY_test, SY_SY_pred, SY_BJ_test, SY_BJ_pred, SY_CD_test, SY_CD_pred = myRNN(X3, y3, X,y,   X2, y2, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
    else:
        BJ_BJ_test, BJ_BJ_pred, BJ_CD_test, BJ_CD_pred, BJ_SY_test, BJ_SY_pred = myRNN(np.delete(X,l,axis=1),  y,  np.delete(X2,l,axis=1),y2, np.delete(X3,l,axis=1), y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
        CD_CD_test, CD_CD_pred, CD_BJ_test, CD_BJ_pred, CD_SY_test, CD_SY_pred = myRNN(np.delete(X2,l,axis=1), y2, np.delete(X,l,axis=1),y,   np.delete(X3,l,axis=1), y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
        SY_SY_test, SY_SY_pred, SY_BJ_test, SY_BJ_pred, SY_CD_test, SY_CD_pred = myRNN(np.delete(X3,l,axis=1), y3, np.delete(X,l,axis=1),y,   np.delete(X2,l,axis=1), y2, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
    BJ_BJ_test_all.append(BJ_BJ_test)
    BJ_BJ_pred_all.append(BJ_BJ_pred)
    BJ_CD_test_all.append(BJ_CD_test)
    BJ_CD_pred_all.append(BJ_CD_pred)
    BJ_SY_test_all.append(BJ_SY_test)
    BJ_SY_pred_all.append(BJ_SY_pred)
    
    CD_CD_test_all.append(CD_CD_test)
    CD_CD_pred_all.append(CD_CD_pred)
    CD_BJ_test_all.append(CD_BJ_test)
    CD_BJ_pred_all.append(CD_BJ_pred)
    CD_SY_test_all.append(CD_SY_test)
    CD_SY_pred_all.append(CD_SY_pred)
    
    SY_SY_test_all.append(SY_SY_test)
    SY_SY_pred_all.append(SY_SY_pred)
    SY_BJ_test_all.append(SY_BJ_test)
    SY_BJ_pred_all.append(SY_BJ_pred)
    SY_CD_test_all.append(SY_CD_test)
    SY_CD_pred_all.append(SY_CD_pred)
    
np.savez('PM2.5_BJ_CD_SY_epoch.npz', BJ_BJ_test_all = BJ_BJ_test_all,BJ_BJ_pred_all = BJ_BJ_pred_all,BJ_CD_test_all = BJ_CD_test_all,BJ_CD_pred_all = BJ_CD_pred_all,BJ_SY_test_all = BJ_SY_test_all,BJ_SY_pred_all = BJ_SY_pred_all,
         CD_CD_test_all = CD_CD_test_all,CD_CD_pred_all = CD_CD_pred_all,CD_BJ_test_all = CD_BJ_test_all,CD_BJ_pred_all = CD_BJ_pred_all,CD_SY_test_all = CD_SY_test_all,CD_SY_pred_all = CD_SY_pred_all,
         SY_SY_test_all = SY_SY_test_all,SY_SY_pred_all = SY_SY_pred_all,SY_BJ_test_all = SY_BJ_test_all,SY_BJ_pred_all = SY_BJ_pred_all,SY_CD_test_all = SY_CD_test_all,SY_CD_pred_all = SY_CD_pred_all)
'''

'''
# plot
data = np.load('PM2.5_BJ_CD_SY_myepoch.npz')
BJ_BJ_test_all = data['BJ_BJ_test_all']
BJ_BJ_pred_all = data['BJ_BJ_pred_all']
BJ_CD_test_all = data['BJ_CD_test_all']
BJ_CD_pred_all = data['BJ_CD_pred_all']
BJ_SY_test_all = data['BJ_SY_test_all']
BJ_SY_pred_all = data['BJ_SY_pred_all']

CD_CD_test_all = data['CD_CD_test_all']
CD_CD_pred_all = data['CD_CD_pred_all']
CD_BJ_test_all = data['CD_BJ_test_all']
CD_BJ_pred_all = data['CD_BJ_pred_all']
CD_SY_test_all = data['CD_SY_test_all']
CD_SY_pred_all = data['CD_SY_pred_all']

SY_SY_test_all = data['SY_SY_test_all']
SY_SY_pred_all = data['SY_SY_pred_all']
SY_BJ_test_all = data['SY_BJ_test_all']
SY_BJ_pred_all = data['SY_BJ_pred_all']
SY_CD_test_all = data['SY_CD_test_all']
SY_CD_pred_all = data['SY_CD_pred_all']

mse_BJ = np.full([6,3],np.nan)
mse_CD = np.full([6,3],np.nan)
mse_SY = np.full([6,3],np.nan)
for i in range(6):
    mse_BJ[i,0] = np.mean((BJ_BJ_test_all[i] - BJ_BJ_pred_all[i])**2)
    mse_BJ[i,1] = np.mean((BJ_CD_test_all[i] - BJ_CD_pred_all[i])**2)
    mse_BJ[i,2] = np.mean((BJ_SY_test_all[i] - BJ_SY_pred_all[i])**2)
    
    mse_CD[i,0] = np.mean((CD_CD_test_all[i] - CD_CD_pred_all[i])**2)
    mse_CD[i,1] = np.mean((CD_BJ_test_all[i] - CD_BJ_pred_all[i])**2)
    mse_CD[i,2] = np.mean((CD_SY_test_all[i] - CD_SY_pred_all[i])**2)
    
    mse_SY[i,0] = np.mean((SY_SY_test_all[i] - SY_SY_pred_all[i])**2)
    mse_SY[i,1] = np.mean((SY_BJ_test_all[i] - SY_BJ_pred_all[i])**2)
    mse_SY[i,2] = np.mean((SY_CD_test_all[i] - SY_CD_pred_all[i])**2)

correlation_BJ = np.full([6,3],np.nan)
correlation_CD = np.full([6,3],np.nan)
correlation_SY = np.full([6,3],np.nan)
for i in range(6):
    [correlation_BJ[i,0], pvalue] = pearsonr(BJ_BJ_test_all[i],BJ_BJ_pred_all[i])
    [correlation_BJ[i,1], pvalue] = pearsonr(BJ_CD_test_all[i],BJ_CD_pred_all[i])
    [correlation_BJ[i,2], pvalue] = pearsonr(BJ_SY_test_all[i],BJ_SY_pred_all[i])

    [correlation_CD[i,0], pvalue] = pearsonr(CD_CD_test_all[i],CD_CD_pred_all[i])
    [correlation_CD[i,1], pvalue] = pearsonr(CD_BJ_test_all[i],CD_BJ_pred_all[i])
    [correlation_CD[i,2], pvalue] = pearsonr(CD_SY_test_all[i],CD_SY_pred_all[i])

    [correlation_SY[i,0], pvalue] = pearsonr(SY_SY_test_all[i],SY_SY_pred_all[i])
    [correlation_SY[i,1], pvalue] = pearsonr(SY_BJ_test_all[i],SY_BJ_pred_all[i])
    [correlation_SY[i,2], pvalue] = pearsonr(SY_CD_test_all[i],SY_CD_pred_all[i])

# bar plot
fig,ax = plt.subplots(2,3,figsize=(15,14))
ind = [1,2,3,4,5,6]
width = 0.2
p1 = ax[0,0].bar(ind, mse_BJ[:,0]/np.max(mse_BJ[:,:]), width=0.2, color='r', alpha = 0.4)
p2 = ax[0,0].bar(np.array(ind) + 0.2, mse_BJ[:,1]/np.max(mse_BJ[:,:]),  width=0.2, color='g', alpha = 0.4)
p3 = ax[0,0].bar(np.array(ind) + 2*width, mse_BJ[:,2]/np.max(mse_BJ[:,:]),  width=0.2, color='b', alpha = 0.4)
ax[0,0].set_title('Predicted normalzed error')
ax[0,0].set_xticks(np.array(ind) + width / 2)
ax[0,0].set_xticklabels(('10', '20', '40', '60', '80', '100'),rotation=00)
ax[0,0].set_xlabel('epoch')
ax[0,0].legend((p1[0], p2[0], p3[0]), ('BJ->BJ', 'BJ->CD', 'BJ->SY'))

ind = [1,2,3,4,5,6]
width = 0.2
p1 = ax[0,1].bar(ind, mse_CD[:,0]/np.max(mse_CD[:,:]), width=0.2, color='g', alpha = 0.4)
p2 = ax[0,1].bar(np.array(ind) + 0.2, mse_CD[:,1]/np.max(mse_CD[:,:]),  width=0.2, color='r', alpha = 0.4)
p3 = ax[0,1].bar(np.array(ind) + 2*width, mse_CD[:,2]/np.max(mse_CD[:,:]),  width=0.2, color='b', alpha = 0.4)
ax[0,1].set_title('Predicted normalzed error')
ax[0,1].set_xticks(np.array(ind) + width / 2)
ax[0,1].set_xticklabels(('10', '20', '40', '60', '80', '100'),rotation=00)
ax[0,1].set_xlabel('epoch')
ax[0,1].legend((p1[0], p2[0], p3[0]), ('CD->CD', 'CD->BJ', 'CD->SY'))

ind = [1,2,3,4,5,6]
width = 0.2
p1 = ax[0,2].bar(ind, mse_SY[:,0]/np.max(mse_SY[:,:]), width=0.2, color='b', alpha = 0.4)
p2 = ax[0,2].bar(np.array(ind) + 0.2, mse_SY[:,1]/np.max(mse_SY[:,:]),  width=0.2, color='r', alpha = 0.4)
p3 = ax[0,2].bar(np.array(ind) + 2*width, mse_SY[:,2]/np.max(mse_SY[:,:]),  width=0.2, color='g', alpha = 0.4)
ax[0,2].set_title('Predicted normalzed error')
ax[0,2].set_xticks(np.array(ind) + width / 2)
ax[0,2].set_xticklabels(('10', '20', '40', '60', '80', '100'),rotation=00)
ax[0,2].set_xlabel('epoch')
ax[0,2].legend((p1[0], p2[0], p3[0]), ('SY->SY', 'SY->BJ', 'SY->CD'))


ind = [1,2,3,4,5,6]
width = 0.2
p1 = ax[1,0].bar(ind, correlation_BJ[:,0], width=0.2, color='r', alpha = 0.4)
p2 = ax[1,0].bar(np.array(ind) + 0.2, correlation_BJ[:,1],  width=0.2, color='g', alpha = 0.4)
p3 = ax[1,0].bar(np.array(ind) + 2*width, correlation_BJ[:,2],  width=0.2, color='b', alpha = 0.4)
ax[1,0].set_title('Pearson correlation')
ax[1,0].set_xticks(np.array(ind) + width / 2)
ax[1,0].set_xticklabels(('10', '20', '40', '60', '80', '100'),rotation=00)
ax[1,0].set_xlabel('epoch')
ax[1,0].legend((p1[0], p2[0], p3[0]), ('BJ->BJ', 'BJ->CD', 'BJ->SY'))

ind = [1,2,3,4,5,6]
width = 0.2
p1 = ax[1,1].bar(ind, correlation_CD[:,0], width=0.2, color='g', alpha = 0.4)
p2 = ax[1,1].bar(np.array(ind) + 0.2, correlation_CD[:,1],  width=0.2, color='r', alpha = 0.4)
p3 = ax[1,1].bar(np.array(ind) + 2*width, correlation_CD[:,2],  width=0.2, color='b', alpha = 0.4)
ax[1,1].set_title('Pearson correlation')
ax[1,1].set_xticks(np.array(ind) + width / 2)
ax[1,1].set_xticklabels(('10', '20', '40', '60', '80', '100'),rotation=00)
ax[1,1].set_xlabel('epoch')
ax[1,1].legend((p1[0], p2[0], p3[0]), ('CD->CD', 'CD->BJ', 'CD->SY'))

ind = [1,2,3,4,5,6]
width = 0.2
p1 = ax[1,2].bar(ind, correlation_SY[:,0], width=0.2, color='b', alpha = 0.4)
p2 = ax[1,2].bar(np.array(ind) + 0.2, correlation_SY[:,1],  width=0.2, color='r', alpha = 0.4)
p3 = ax[1,2].bar(np.array(ind) + 2*width, correlation_SY[:,2],  width=0.2, color='g', alpha = 0.4)
ax[1,2].set_title('Pearson correlation')
ax[1,2].set_xticks(np.array(ind) + width / 2)
ax[1,2].set_xticklabels(('10', '20', '40', '60', '80', '100'),rotation=00)
ax[1,2].set_xlabel('epoch')
ax[1,2].legend((p1[0], p2[0], p3[0]), ('SY->SY', 'SY->BJ', 'SY->CD'))

plt.savefig('PM2.5_3cities.Figure_epoch.png',dpi=600)
'''

'''
for i in [200]: # 100,200,1000,2000,5000,10000
    for j in [30]: # 10,20,30,40,50
        for k in [50]: # 10,20,40,60,80,100
            for l in [0]: # 0,1,2,3,4,5,6
                if l == 0:
                    BJ_BJ_test, BJ_BJ_pred, BJ_CD_test, BJ_CD_pred, BJ_SY_test, BJ_SY_pred = myRNN(X,  y,  X2,y2, X3, y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
                    CD_CD_test, CD_CD_pred, CD_BJ_test, CD_BJ_pred, CD_SY_test, CD_SY_pred = myRNN(X2, y2, X,y,   X3, y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
                    SY_SY_test, SY_SY_pred, SY_BJ_test, SY_BJ_pred, SY_CD_test, SY_CD_pred = myRNN(X3, y3, X,y,   X2, y2, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
                    
                    BJ_BJ_test_all.append(BJ_BJ_test)
                    BJ_BJ_pred_all.append(BJ_BJ_pred)
                    BJ_CD_test_all.append(BJ_CD_test)
                    BJ_CD_pred_all.append(BJ_CD_pred)
                    BJ_SY_test_all.append(BJ_SY_test)
                    BJ_SY_pred_all.append(BJ_SY_pred)
                    
                    CD_CD_test_all.append(CD_CD_test)
                    CD_CD_pred_all.append(CD_CD_pred)
                    CD_BJ_test_all.append(CD_BJ_test)
                    CD_BJ_pred_all.append(CD_BJ_pred)
                    CD_SY_test_all.append(CD_SY_test)
                    CD_SY_pred_all.append(CD_SY_pred)
                    
                    SY_SY_test_all.append(SY_SY_test)
                    SY_SY_pred_all.append(SY_SY_pred)
                    SY_BJ_test_all.append(SY_BJ_test)
                    SY_BJ_pred_all.append(SY_BJ_pred)
                    SY_CD_test_all.append(SY_CD_test)
                    SY_CD_pred_all.append(SY_CD_pred)
                else:
                    BJ_BJ_test, BJ_BJ_pred, BJ_CD_test, BJ_CD_pred, BJ_SY_test, BJ_SY_pred = myRNN(np.delete(X,l,axis=1),  y,  np.delete(X2,l,axis=1),y2, np.delete(X3,l,axis=1), y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
                    CD_CD_test, CD_CD_pred, CD_BJ_test, CD_BJ_pred, CD_SY_test, CD_SY_pred = myRNN(np.delete(X2,l,axis=1), y2, np.delete(X,l,axis=1),y,   np.delete(X3,l,axis=1), y3, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
                    SY_SY_test, SY_SY_pred, SY_BJ_test, SY_BJ_pred, SY_CD_test, SY_CD_pred = myRNN(np.delete(X3,l,axis=1), y3, np.delete(X,l,axis=1),y,   np.delete(X2,l,axis=1), y2, mem=30, nbatches = i, nepochs = k, nlayers = 3, nnodes = j, activation_func = 'relu')
                    
                    BJ_BJ_test_all.append(BJ_BJ_test)
                    BJ_BJ_pred_all.append(BJ_BJ_pred)
                    BJ_CD_test_all.append(BJ_CD_test)
                    BJ_CD_pred_all.append(BJ_CD_pred)
                    BJ_SY_test_all.append(BJ_SY_test)
                    BJ_SY_pred_all.append(BJ_SY_pred)
                    
                    CD_CD_test_all.append(CD_CD_test)
                    CD_CD_pred_all.append(CD_CD_pred)
                    CD_BJ_test_all.append(CD_BJ_test)
                    CD_BJ_pred_all.append(CD_BJ_pred)
                    CD_SY_test_all.append(CD_SY_test)
                    CD_SY_pred_all.append(CD_SY_pred)
                    
                    SY_SY_test_all.append(SY_SY_test)
                    SY_SY_pred_all.append(SY_SY_pred)
                    SY_BJ_test_all.append(SY_BJ_test)
                    SY_BJ_pred_all.append(SY_BJ_pred)
                    SY_CD_test_all.append(SY_CD_test)
                    SY_CD_pred_all.append(SY_CD_pred)
                
                print([i,j,k,l])

np.savez('PM2.5_BJ_CD_SY_best_param2.npz', BJ_BJ_test_all = BJ_BJ_test_all,BJ_BJ_pred_all = BJ_BJ_pred_all,BJ_CD_test_all = BJ_CD_test_all,BJ_CD_pred_all = BJ_CD_pred_all,BJ_SY_test_all = BJ_SY_test_all,BJ_SY_pred_all = BJ_SY_pred_all,
         CD_CD_test_all = CD_CD_test_all,CD_CD_pred_all = CD_CD_pred_all,CD_BJ_test_all = CD_BJ_test_all,CD_BJ_pred_all = CD_BJ_pred_all,CD_SY_test_all = CD_SY_test_all,CD_SY_pred_all = CD_SY_pred_all,
         SY_SY_test_all = SY_SY_test_all,SY_SY_pred_all = SY_SY_pred_all,SY_BJ_test_all = SY_BJ_test_all,SY_BJ_pred_all = SY_BJ_pred_all,SY_CD_test_all = SY_CD_test_all,SY_CD_pred_all = SY_CD_pred_all)
'''

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 02:21:48 2018
@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data_batch = np.load('PM2.5_BJ_CD_SY_batch.npz')

BJ_BJ_test_all_batch = data_batch['BJ_BJ_test_all']
BJ_BJ_pred_all_batch = data_batch['BJ_BJ_pred_all']
BJ_CD_test_all_batch = data_batch['BJ_CD_test_all']
BJ_CD_pred_all_batch = data_batch['BJ_CD_pred_all']
BJ_SY_test_all_batch = data_batch['BJ_SY_test_all']
BJ_SY_pred_all_batch = data_batch['BJ_SY_pred_all']

CD_CD_test_all_batch = data_batch['CD_CD_test_all']
CD_CD_pred_all_batch = data_batch['CD_CD_pred_all']
CD_BJ_test_all_batch = data_batch['CD_BJ_test_all']
CD_BJ_pred_all_batch = data_batch['CD_BJ_pred_all']
CD_SY_test_all_batch = data_batch['CD_SY_test_all']
CD_SY_pred_all_batch = data_batch['CD_SY_pred_all']

SY_SY_test_all_batch = data_batch['SY_SY_test_all']
SY_SY_pred_all_batch = data_batch['SY_SY_pred_all']
SY_BJ_test_all_batch = data_batch['SY_BJ_test_all']
SY_BJ_pred_all_batch = data_batch['SY_BJ_pred_all']
SY_CD_test_all_batch = data_batch['SY_CD_test_all']
SY_CD_pred_all_batch = data_batch['SY_CD_pred_all']

data_epoch = np.load('PM2.5_BJ_CD_SY_epoch.npz')

BJ_BJ_test_all_epoch = data_epoch['BJ_BJ_test_all']
BJ_BJ_pred_all_epoch = data_epoch['BJ_BJ_pred_all']
BJ_CD_test_all_epoch = data_epoch['BJ_CD_test_all']
BJ_CD_pred_all_epoch = data_epoch['BJ_CD_pred_all']
BJ_SY_test_all_epoch = data_epoch['BJ_SY_test_all']
BJ_SY_pred_all_epoch = data_epoch['BJ_SY_pred_all']

CD_CD_test_all_epoch = data_epoch['CD_CD_test_all']
CD_CD_pred_all_epoch = data_epoch['CD_CD_pred_all']
CD_BJ_test_all_epoch = data_epoch['CD_BJ_test_all']
CD_BJ_pred_all_epoch = data_epoch['CD_BJ_pred_all']
CD_SY_test_all_epoch = data_epoch['CD_SY_test_all']
CD_SY_pred_all_epoch = data_epoch['CD_SY_pred_all']

SY_SY_test_all_epoch = data_epoch['SY_SY_test_all']
SY_SY_pred_all_epoch = data_epoch['SY_SY_pred_all']
SY_BJ_test_all_epoch = data_epoch['SY_BJ_test_all']
SY_BJ_pred_all_epoch = data_epoch['SY_BJ_pred_all']
SY_CD_test_all_epoch = data_epoch['SY_CD_test_all']
SY_CD_pred_all_epoch = data_epoch['SY_CD_pred_all']


# '三个城市 九种预测方式对比:'
sns.jointplot(BJ_BJ_test_all_batch[1],BJ_BJ_pred_all_batch[1],kind='reg',color='red')
sns.jointplot(BJ_CD_test_all_batch[1],BJ_CD_pred_all_batch[1],kind='reg',color='red')
sns.jointplot(BJ_SY_test_all_batch[1],BJ_SY_pred_all_batch[1],kind='reg',color='red')

sns.jointplot(CD_CD_test_all_batch[1],CD_CD_pred_all_batch[1],kind='reg',color='green')
sns.jointplot(CD_BJ_test_all_batch[1],CD_BJ_pred_all_batch[1],kind='reg',color='green')
sns.jointplot(CD_SY_test_all_batch[1],CD_SY_pred_all_batch[1],kind='reg',color='green')

sns.jointplot(SY_SY_test_all_batch[1],SY_SY_pred_all_batch[1],kind='reg',color='blue')
sns.jointplot(SY_BJ_test_all_batch[1],SY_BJ_pred_all_batch[1],kind='reg',color='blue')
sns.jointplot(SY_CD_test_all_batch[1],SY_CD_pred_all_batch[1],kind='reg',color='blue')

# batch = 100,500,1000,2000,5000,10000 以北京为例(100,500最好): 
for i in range(6):
    sns.jointplot(BJ_BJ_test_all_batch[i],BJ_BJ_pred_all_batch[i],kind='reg',color='red')
#sns_batch.savefig('batch.png')

# 'epoch = 10,20,40,60,80,100 以北京为例(10,20明显不好，其他差异不显著): 
for i in range(6):
    sns.jointplot(BJ_BJ_test_all_epoch[i],BJ_BJ_pred_all_epoch[i],kind='reg',color='green')
    

    
'''
fig = plt.figure(figsize = (20, 10))
fig.subplots_adjust(top = 0.8, wspace = 0.2, hspace = 0.2)
for i in range(6):
    fig.add_subplot(3, 6, i + 1)
    sns.jointplot(BJ_BJ_test_all[i],BJ_BJ_pred_all[i],kind='reg',color='green')
    fig.add_subplot(3, 6, i + 7)
    sns.jointplot(BJ_CD_test_all[i],BJ_CD_pred_all[i],kind='reg',color='green')
    fig.add_subplot(3, 6, i + 13)
    sns.jointplot(BJ_SY_test_all[i],BJ_SY_pred_all[i],kind='reg',color='green')
'''
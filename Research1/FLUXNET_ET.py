import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
from TE_library_QING import calculate_TE,getSurrogateData

# feature scaling
def feature_scaling(data):
    
    from sklearn.preprocessing import StandardScaler
    
    sc = StandardScaler()
    data = sc.fit_transform(data)
    
    return data,sc

# detrend
def detrend_ma(data,window):
    
    from scipy.ndimage.filters import uniform_filter1d
    
    for i in range(data.shape[1]):
        data[:,i] = data[:,i] - uniform_filter1d(data[:,i], size=window)
    
    return data

# time, LE, NEE , GPP, FSDS, Prcp, T, FLDS, SH, VPD
data = pd.read_csv('FLX_IT-CA1_DD_2011-2014.csv')
# 2011-2013 1462 samples
grass_data = data.iloc[:,[1,3,4,5,6,8,9]].values
grass_data = grass_data[~np.isnan(grass_data).any(axis=1)]
grass_data[:,3] = np.log(grass_data[:,3] + 1.0) # prec
for i in range(len(grass_data)):
    if grass_data[i,5] < 0:
        grass_data[i,5] = 0
grass_data = np.insert(grass_data,7,grass_data[:,0]/ (grass_data[:,0] + grass_data[:,5]),axis=1)# evaporative fraction

data = pd.read_csv('FLX_IT-CA2_DD_2011-2014.csv')
# 2011-2013 1462 samples
forest_data = data.iloc[:,[1,3,4,5,6,8,9]].values
forest_data = forest_data[~np.isnan(forest_data).any(axis=1)]
forest_data[:,3] = np.log(forest_data[:,3] + 1.0)
for i in range(len(forest_data)):
    if forest_data[i,5] < 0:
        forest_data[i,5] = 0
forest_data = np.insert(forest_data,7,forest_data[:,0]/ (forest_data[:,0] + forest_data[:,5]),axis=1)# evaporative fraction

varnames = list(data.columns[[1,3,4,5,6,8,9]].values)
varnames.append('EF')# evaporative fraction

fig = plt.figure(figsize = (12, 4))
fig.subplots_adjust(top = 0.8, wspace = 0.2, hspace = 0.6)
for i in range(len(varnames)):
    ax = fig.add_subplot(2,4,i+1)
    ax.plot(np.linspace(2011,2014,grass_data.shape[0]),grass_data[:,i],color='r',label = 'grass')
    ax.plot(np.linspace(2011,2014,forest_data.shape[0]),forest_data[:,i],color='b',label = 'forest')
    ax.set_title(varnames[i])
    start, end = ax.get_xlim()
    if i == 7:
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('FLUXNET_ET.Figure1.png',dpi=900)

inputdata_grass = pd.DataFrame(grass_data,columns=varnames)
inputdata_forest = pd.DataFrame(forest_data,columns=varnames)
import seaborn as sns
sns.pairplot(inputdata_grass[['GPP','FSDS','Prcp','VPD','EF']],size=1.8,aspect=1.8,
             plot_kws=dict(edgecolor='k',linewidth=0.5),
             diag_kind='kde',diag_kws=dict(shade=True))
plt.savefig('FLUXNET_ET.Figure2.1.grass.png',dpi=900)
sns.pairplot(inputdata_forest[['GPP','FSDS','Prcp','VPD','EF']],size=1.8,aspect=1.8,
             plot_kws=dict(edgecolor='k',linewidth=0.5),
             diag_kind='kde',diag_kws=dict(shade=True))
plt.savefig('FLUXNET_ET.Figure2.2.forest.png',dpi=900)

corr_grass = inputdata_grass.corr()
corr_forest = inputdata_forest.corr()

sns.heatmap(corr_grass, vmin=-1.0, vmax=1.0, annot=True, fmt='.2f',
            cmap='coolwarm',linewidths=0.05, linecolor='white',
            cbar=True,cbar_kws={'orientation':"vertical"}, square=False)
plt.savefig('FLUXNET_ET.Figure2.3.grass.png',dpi=900)

sns.heatmap(corr_forest, vmin=-1.0, vmax=1.0, annot=True, fmt='.2f',
            cmap='coolwarm',linewidths=0.05, linecolor='white',
            cbar=True,cbar_kws={'orientation':"vertical"}, square=False)
plt.savefig('FLUXNET_ET.Figure2.4.forest.png',dpi=900)

sns.jointplot(x='LE',y='GPP',data=inputdata_grass, kind='kde', space=0.2) # kind = reg, hex, kde

# detrend
#grass_data = detrend_ma(data=grass_data,window=100)
#forest_data = detrend_ma(data=forest_data,window=100)

# feature scaling
#grass_data,grass_sc = feature_scaling(data=grass_data)
#forest_data,forest_sc = feature_scaling(data=forest_data)

# original data, LE, GPP, FSDS, P, T, SH, VPD, EF
# only consider, GPP, FSDS, P, T, VPD, EF
#grass_data = grass_data[:,[1,2,3,4,6,7]]
#forest_data = forest_data[:,[1,2,3,4,6,7]]
#varnames = varnames[1:5]+varnames[6:8]

# mutual information analysis
## step 1: ============= user-defined parameters
opts = {}

# run grass site or forest site
opts['site'] = 'grass'

# 0 = use ordinary ranking to estimate PDF
# 1 = use Gaussian Kernel Estimation
opts['pdf'] = 1

# number of bins for gaussian kernel estimations
opts['GKE_n'] = 11

# Number of surrogates to create and/or test (default = 100), for statistical testing
opts['nTests'] = 20

# the number of data points signifiying the previous history of Y.
opts['nYw'] = [1]

# lags (in units of time steps) to evaluate (default = 0:10). Note: 0 will always be included, whether it is entered here or not.
opts['lagVect'] = np.arange(5)

# one-tail z-score for 95% significance given number of tests, 1.66 for 100, 1.68 for 50, 1.71 for 25
opts['oneTailZ'] = 1.66

# TE link 'firstSignal': only consider TE from first var to others; 'allSignals': consider nxn TE
opts['TE_link'] =  'allSignals'

## step 2: ============= read in data and preprocess
if opts['site'] == 'grass':
    Data = grass_data
elif opts['site'] == 'forest':
    Data = forest_data
rawData = Data
opts['varNames']= varnames

# number of row, column of the data; number of variables to run TE through = nSignals x nSignals
[nData,nSignals] = np.shape(Data)

# nLags includes the zero lag which is first
nLags = len(opts['lagVect'])

## step 3: ============= calculate TE
# initialize
HXt       = np.full([nSignals,nSignals,nLags],np.nan)
HYw       = np.full([nSignals,nSignals,nLags],np.nan)
HYf       = np.full([nSignals,nSignals,nLags],np.nan)
HXtYw     = np.full([nSignals,nSignals,nLags],np.nan)
HXtYf     = np.full([nSignals,nSignals,nLags],np.nan)
HYwYf     = np.full([nSignals,nSignals,nLags],np.nan)
HXtYwYf   = np.full([nSignals,nSignals,nLags],np.nan)
I         = np.full([nSignals,nSignals,nLags],np.nan)
T         = np.full([nSignals,nSignals,nLags],np.nan)
IR        = np.full([nSignals,nSignals,nLags],np.nan)
TR        = np.full([nSignals,nSignals,nLags],np.nan)
Tplus     = np.full([nSignals,nLags],0.0)
Tminus    = np.full([nSignals,nLags],0.0)
Tnet      = np.full([nSignals,nLags],0.0)

pool = mp.Pool(processes=4)
calculate_TE_partial=partial(calculate_TE, Data=Data, opts=opts) # prod_x has only one argument x (Data and opts are fixed) 
results = pool.map(calculate_TE_partial, range(nLags)) 
pool.close()
pool.join()

for i in range(nLags):
    HXt[:,:,i]     = results[i]['HXt']
    HYw[:,:,i]     = results[i]['HYw']
    HYf[:,:,i]     = results[i]['HYf']
    HXtYw[:,:,i]   = results[i]['HXtYw']
    HXtYf[:,:,i]   = results[i]['HXtYf']
    HYwYf[:,:,i]   = results[i]['HYwYf']
    HXtYwYf[:,:,i] = results[i]['HXtYwYf']
    I[:,:,i]       = results[i]['I']
    T[:,:,i]       = results[i]['T']
    IR[:,:,i]      = results[i]['IR']
    TR[:,:,i]      = results[i]['TR']
    Tplus[:,i]     = results[i]['Tplus']
    Tminus[:,i]    = results[i]['Tminus']
    Tnet[:,i]      = results[i]['Tnet']

# calculate suroogate TE
surrogateData = getSurrogateData(rawData=rawData,opts=opts)
HYf_surr  = np.full([nSignals,nSignals,nLags,opts['nTests']],np.nan)
I_surr    = np.full([nSignals,nSignals,nLags,opts['nTests']],np.nan)
IR_surr   = np.full([nSignals,nSignals,nLags,opts['nTests']],np.nan)
T_surr    = np.full([nSignals,nSignals,nLags,opts['nTests']],np.nan)
TR_surr   = np.full([nSignals,nSignals,nLags,opts['nTests']],np.nan)
# loop over number of surrogate tests
for i in range(opts['nTests']):
    # cauculate surrogate test TE and store necessary variables
    pool = mp.Pool(processes=4)
    calculate_TE_partial=partial(calculate_TE, Data=surrogateData[:,:,i], opts=opts) # prod_x has only one argument x (Data and opts are fixed) 
    results_sur = pool.map(calculate_TE_partial, range(nLags)) 
    pool.close()
    pool.join()
    for j in range(nLags):
        HYf_surr[:,:,j,i] = results_sur[j]['HYf']
        I_surr[:,:,j,i] = results_sur[j]['I']
        T_surr[:,:,j,i] = results_sur[j]['T']

# derive relative mutual info from X to Y
TR_surr = T_surr / HYf_surr
# derive relative TE from X to Y
IR_surr = I_surr / HYf_surr
# calculate statistics for significant test
meanShuffT             = np.mean(T_surr,axis=3)
sigmaShuffT            = np.std(T_surr,axis=3)
meanShuffI             = np.mean(I_surr,axis=3)
sigmaShuffI            = np.std(I_surr,axis=3)
meanShuffTR            = np.mean(TR_surr,axis=3)
sigmaShuffTR           = np.std(TR_surr,axis=3)
meanShuffIR            = np.mean(IR_surr,axis=3)
sigmaShuffIR           = np.mean(IR_surr,axis=3)
SigThreshT             = meanShuffT + opts['oneTailZ']*sigmaShuffT
SigThreshI             = meanShuffI + opts['oneTailZ']*sigmaShuffI
SigThreshTR            = meanShuffTR + opts['oneTailZ']*sigmaShuffTR
SigThreshIR            = meanShuffIR + opts['oneTailZ']*sigmaShuffIR

## save data
if opts['site'] == 'grass':
    filename = 'FLX_CA1_grass_TE.npz'
elif opts['site'] == 'forest':
    filename = 'FLX_CA2_forest_TE.npz'
np.savez(filename, HXt=HXt, HYw=HYw, HYf=HYf, HXtYw=HXtYw, HXtYf=HXtYf, HYwYf=HYwYf, HXtYwYf=HXtYwYf, \
         I=I, T=T, IR=IR, TR=TR, Tplus=Tplus, Tminus=Tminus, Tnet=Tnet, \
         meanShuffT=meanShuffT, sigmaShuffT=sigmaShuffT, meanShuffI=meanShuffI, sigmaShuffI=sigmaShuffI, \
         meanShuffTR=meanShuffTR, sigmaShuffTR=sigmaShuffTR, meanShuffIR=meanShuffIR, sigmaShuffIR=sigmaShuffIR, \
         SigThreshT=SigThreshT, SigThreshI=SigThreshI, SigThreshTR=SigThreshTR, SigThreshIR=SigThreshIR , \
         Data=Data, varNames=opts['varNames'], nTests=opts['nTests'], nYw=opts['nYw'])

## step 5: ============= simple visualization and output data for Gelphi visualization: e.g. Tnet, T, I
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# load in TE
if opts['site'] == 'grass':
    filename = 'FLX_CA1_grass_TE.npz'
elif opts['site'] == 'forest':
    filename = 'FLX_CA2_forest_TE.npz'
data = np.load(filename)
I=data['I']
T=data['T']
Tplus=data['Tplus']
Tminus=data['Tminus']
Tnet=data['Tnet']
SigThreshT=data['SigThreshT']
SigThreshI=data['SigThreshI']
varNames=data['varNames']
meanShuffT=data['meanShuffT']
sigmaShuffT=data['sigmaShuffT']

# get dimension of data
[nData,nSignals] = np.shape(data['Data'])
# plot T vs T threshold
plt.figure(figsize = (25,20))
index = 1
for sX in range(nSignals):
    for sY in range(nSignals):
        if True: #sX != sY:
            plt.subplot(nSignals,nSignals,index)
            if np.max(T[sX,sY,1:] - SigThreshT[sX,sY,1:]) > 0:
                 plt.plot(T[sX,sY,1:],c='b',label='T')
            else:
                plt.plot(T[sX,sY,1:],c='k',label='T')
            plt.plot(range(4),meanShuffT[sX,sY,1:]+1.66*sigmaShuffT[sX,sY,1:],c='r',label='T shreshold+')
            plt.plot(range(4),meanShuffT[sX,sY,1:]-1.66*sigmaShuffT[sX,sY,1:],c='g',label='T shreshold-')
            plt.fill_between(range(4),meanShuffT[sX,sY,1:]-1.66*sigmaShuffT[sX,sY,1:],meanShuffT[sX,sY,1:]+1.66*sigmaShuffT[sX,sY,1:],color='grey', alpha='0.2')
            if sX == 0:
                plt.title(varNames[sY])
            if sY == 0:
                plt.ylabel(varNames[sX])
            #if index == 2:
            #    plt.legend()
        index = index + 1
if opts['site'] == 'grass':
    filename = 'FLUXNET_ET.TE._FLX_CA1_grass.png'
elif opts['site'] == 'forest':
    filename = 'FLUXNET_ET.TE._FLX_CA2_forest.png'
plt.savefig(filename,dpi=600)

# output data for Gelphi visualization
# node size: Tpus, Tminus, or Tnet
# edge size and color: T, I
if opts['site'] == 'grass':
    filename1 = 'Gelphi_node_FLX_CA1_grass.csv'
    filename2 = 'Gelphi_edge_FLX_CA1_grass.csv'
elif opts['site'] == 'forest':
    filename1 = 'Gelphi_node_FLX_CA2_forest.csv'
    filename2 = 'Gelphi_edge_FLX_CA2_forest.csv'
Gelphi_node = []
for sX in range(nSignals):
    Gelphi_node.append([ varNames[sX], varNames[sX], np.mean(Tplus[sX,:]), np.mean(Tminus[sX,:]), np.mean(Tnet[sX,:])])
Gelphi_node_df = pd.DataFrame(data=Gelphi_node, columns=['Id','Label', 'Tplus', 'Tminus', 'Tnet'])
Gelphi_node_df.to_csv(filename1,encoding='utf-8', index=False)

Gelphi_edge = []
index = 1
for sX in range(nSignals):
    for sY in range(nSignals):
        if sX != sY:
            # exclude 0 time lag of TE_sX->sY
            diff_T = T[sX,sY,1:] - SigThreshT[sX,sY,1:]
            if  np.max(diff_T) > 0:
                Gelphi_edge.append([varNames[sX], varNames[sY], np.mean(I[sX,sY,:]), np.mean(SigThreshI[sX,sY,:]), np.mean(T[sX,sY,:]), np.mean(SigThreshT[sX,sY,:]), \
                                np.where(diff_T>0)[0][0] + 1,  diff_T[np.where(diff_T>0)[0][0]], np.where(diff_T>0)[0][-1] + 1, diff_T[np.where(diff_T>0)[0][-1]], \
                                int(np.mean(np.where(diff_T>0)[0])), np.mean(diff_T[np.where(diff_T>0)[0]]), np.where(diff_T>0)[0].size ])
Gelphi_edge_df = pd.DataFrame(data=Gelphi_edge, columns=['Source','Target','I', 'SigThreshI','T', 'SigThreshT', \
               'firtLag', 'firtT', 'lastLag', 'lastT', 'avgLagT', 'avgT', '#LagT'])
Gelphi_edge_df.to_csv(filename2,encoding='utf-8', index=False)


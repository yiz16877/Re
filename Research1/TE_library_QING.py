#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:03:21 2020

@author: yi
"""

mport numpy as np
import matplotlib.pyplot as plt

#=========================================================== fixed bin
# estimate 1D probability
# mean and std
mu = 0
sigma = 1
nData = 500
# genreate data
np.random.seed(0)
data = sigma * np.random.randn(nData) + mu
#plt.hist(data)
#plt.show()
# prescribe bins
def get_pdf(bins=11,data=data):
    binMin = np.min(data)
    binMax = np.max(data)
    binEdges = np.linspace(binMin,binMax,bins+1)
    
    # calssify data
    classifiedData = np.full([nData],np.nan)
    smat = data.copy()
    # define classified matrix
    cmat = np.full([nData],np.nan)
    # loop over local bins
    for e in range(bins):
        if np.where( smat <= binEdges[e+1] )[0].size > 0:
            # Find data in the variable falling within this bin and Assign classification
            cmat[np.where( smat <= binEdges[e+1] )[0]] = e
            # remove these that were classified from further consideration
            smat[np.where( smat <= binEdges[e+1] )[0]] = np.nan
            # assign classified data variable
    # classifiedData: 0,1,2,3,...
    classifiedData = cmat
    
    # calculate PDF
    C = np.full([bins],0)
    for i in range(nData):
        C[int(classifiedData[i])]  = C[int(classifiedData[i])] + 1
    nCounts = sum(C)
    pX=(C+1e-20)/nCounts
    # plot
    plt.plot(pX)

plt.subplot(2,2,1)
get_pdf(12,data)
plt.subplot(2,2,2)
get_pdf(11,data)
plt.subplot(2,2,3)
get_pdf(10,data)
plt.subplot(2,2,4)
get_pdf(9,data)

#=========================================================== Gaussian Kernel Estimation
# estimate 1D probability
# mean and std
mu = 0
sigma = 1
nData = 500
# genreate data
np.random.seed(0)
data = sigma * np.random.randn(nData) + mu

def get_pdf_gke(bins=11,data=data):
    binMin = np.min(data)
    binMax = np.max(data)
    binEdges = np.linspace(binMin,binMax,bins+1)
    
    # bandwith
    band_X = 1.06*np.std(data)*nData**-0.2
                    
    # calculate PDF
    pX = np.full([bins+1],1e-20)
    for k in range(bins+1):
        pX[k] = sum(1.0/nData/(2.0*np.pi)**0.5/band_X * np.exp(-0.5*( (data - binEdges[k])/band_X )**2))
    pX = pX/np.sum(pX)
    # plot
    plt.plot(pX)

plt.subplot(2,2,1)
get_pdf(12,data)
plt.subplot(2,2,2)
get_pdf(11,data)
plt.subplot(2,2,3)
get_pdf(10,data)
plt.subplot(2,2,4)
get_pdf(9,data)

plt.subplot(2,2,1)
get_pdf_gke(14,data)
plt.subplot(2,2,2)
get_pdf_gke(12,data)
plt.subplot(2,2,3)
get_pdf_gke(10,data)
plt.subplot(2,2,4)
get_pdf_gke(8,data)
plt.show()

#=========================================================== fixed bin 2D
# estimate 2D probability
# mean and std
mean = [0, 0]
cov = [[1, 50], [50, 100]]  # diagonal covariance
nData = 500
np.random.seed(0)
x, y = np.random.multivariate_normal(mean, cov, nData).T
plt.hist2d(x, y, bins=20)
plt.show()
# prescribe bins
def get_pdf2d(bins=11,x=x,y=y):
    xbinMin = np.min(x)
    xbinMax = np.max(x)
    xbinEdges = np.linspace(xbinMin,xbinMax,bins+1)
    ybinMin = np.min(y)
    ybinMax = np.max(y)
    ybinEdges = np.linspace(ybinMin,ybinMax,bins+1)
    
    # calssify data
    xclassified = np.full([nData],np.nan)
    yclassified = np.full([nData],np.nan)
    xsmat = x.copy()
    ysmat = y.copy()
    # define classified matrix
    xcmat = np.full([nData],np.nan)
    ycmat = np.full([nData],np.nan)
    # loop over local bins
    for e in range(bins):
        if np.where( xsmat <= xbinEdges[e+1] )[0].size > 0:
            # Find data in the variable falling within this bin and Assign classification
            xcmat[np.where( xsmat <= xbinEdges[e+1] )[0]] = e
            # remove these that were classified from further consideration
            xsmat[np.where( xsmat <= xbinEdges[e+1] )[0]] = np.nan
            # assign classified data variable
        if np.where( ysmat <= ybinEdges[e+1] )[0].size > 0:
            # Find data in the variable falling within this bin and Assign classification
            ycmat[np.where( ysmat <= ybinEdges[e+1] )[0]] = e
            # remove these that were classified from further consideration
            ysmat[np.where( ysmat <= ybinEdges[e+1] )[0]] = np.nan
            # assign classified data variable
    # classifiedData: 0,1,2,3,...
    xclassified = xcmat
    yclassified = ycmat
    
    # calculate PDF
    C = np.full([bins,bins],0)
    for i in range(nData):
        C[int(xclassified[i]),int(yclassified[i])]  = C[int(xclassified[i]),int(yclassified[i])] + 1
    nCounts = sum(C)
    pXpY=(C+1e-20)/nCounts
    # plot
    plt.contourf(pXpY)

plt.subplot(2,2,1)
get_pdf2d(12,x,y)
plt.subplot(2,2,2)
get_pdf2d(11,x,y)
plt.subplot(2,2,3)
get_pdf2d(10,x,y)
plt.subplot(2,2,4)
get_pdf2d(9,x,y)

#=========================================================== Gaussian Kernel Estimation 2D
# estimate 2D probability
mean = [0, 0]
cov = [[1, 50], [50, 100]]  # diagonal covariance
nData = 500
np.random.seed(0)
x, y = np.random.multivariate_normal(mean, cov, nData).T
plt.hist2d(x, y, bins=20)
plt.show()

def get_pdf_gke2d(bins=11,x=x,y=y):
    xbinMin = np.min(x)
    xbinMax = np.max(x)
    xbinEdges = np.linspace(xbinMin,xbinMax,bins+1)
    ybinMin = np.min(y)
    ybinMax = np.max(y)
    ybinEdges = np.linspace(ybinMin,ybinMax,bins+1)
    
    # bandwith
    band_X = 1.06*np.std(x)*nData**-0.2
    band_Y = 1.06*np.std(y)*nData**-0.2   
    
    # calculate PDF
    pXpY = np.full([bins+1,bins+1],1e-20)
    for k in range(bins+1):
        for j in range(bins+1):
            pXpY[k,j] = sum(1.0/nData/(2.0*np.pi)**0.5/band_X/band_Y * \
              np.exp(-0.5*( (x - xbinEdges[k])/band_X )**2) * \
              np.exp(-0.5*( (y - ybinEdges[j])/band_X )**2))
    pXpY = pXpY/np.sum(pXpY)
    # plot
    plt.contourf(pXpY)

plt.subplot(2,2,1)
get_pdf_gke2d(14,x,y)
plt.subplot(2,2,2)
get_pdf_gke2d(12,x,y)
plt.subplot(2,2,3)
get_pdf_gke2d(10,x,y)
plt.subplot(2,2,4)
get_pdf_gke2d(8,x,y)
plt.show()
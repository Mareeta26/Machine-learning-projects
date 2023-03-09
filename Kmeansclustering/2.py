#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:43:26 2020

@author: mareeta
"""

"""Iris.xls contains 150 data samples of three Iris categories, labeled by outcome values 0, 1, and 2. Each data sample has four attributes: sepal length, sepal width, petal length, and petal width.
Implement the K-means clustering algorithm to group the samples into K=3 clusters. Randomly choose three
samples as the initial cluster centers.  Exit the iterations if the following criterion is met: 𝐽(Iter − 1) − 𝐽(Iter) < ε,
−5
 , and Iter is the iteration number. 
where ε = 10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
#reading excel file
iris_data=  pd.read_excel('Iris.xls', usecols=[0,1,2,3,4,5])

#no of centers
K =3 
X = iris_data.iloc[:, [1,2,3,4]].values
y = iris_data.iloc[:, [5]].values

# number of training data
n = np.size(X,0)
# number of features
f = np.size(X,1)
#center array
center=np.array([]).reshape(f,0) 


distances= np.zeros((n,K))
C = np.zeros(n)
# choose 3 random samples as cluster center
idx = np.random.randint(150, size=3)
center=X[idx,:]

J_old= np.zeros((K,1))
J_new = np.zeros((K,1))
J = np.zeros((20,1))
#iter 0 
for i in range(K):
    distances[:,i] = np.linalg.norm(X - center[i], axis=1)
C = np.argmin(distances, axis = 1)
for i in range(K):
    J_old[i] = np.sum(np.linalg.norm(X[C==i]-center[i],axis =1))
for i in range(K):
    center[i] = np.mean(X[C == i], axis=0)

Iter= 0
print(np.sum(J_old))
J[0]= np.sum(J_old)
#iterations
    
while True:
    Iter =  Iter +1
    for i in range(K):
        distances[:,i] = np.linalg.norm(X - center[i], axis=1)
#assign minimum J 
    C = np.argmin(distances, axis = 1)
    #calculate J
    for i in range(K):
        J_new[i]= np.sum(np.linalg.norm(X[C==i]-center[i],axis =1))
    #recompute centers
    for i in range(K):
        center[i] = np.mean(X[C == i], axis=0)
    print(np.sum(J_new))
    J[Iter -1] = np.sum(J_new)
    #stopping criteria
    if np.sum(J_old) - np.sum(J_new) < math.pow(10,-5): 
        break
    else:
        J_old = J_new.copy()
        
J = J[~np.all(J == 0, axis=1)]
k= range(0,Iter)
plt.plot(k,J,'go--', linewidth=2, markersize=12)

plt.xlabel('k (Iterations)')
plt.ylabel('J - objective function value')
plt.show()
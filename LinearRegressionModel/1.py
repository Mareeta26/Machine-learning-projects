#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 05:20:12 2020

@author: mareeta
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#reading csv file
prima_data=  pd.read_csv('pima-indians-diabetes.csv',header=None, usecols=[0,1,2,3,4,5,6,7,8])

#split diabetic and non-diabetic samples
diabetic= prima_data[(prima_data[8] == 1)]
no_diabetic= prima_data[(prima_data[8] == 0)]

accuracy = np.zeros(1000)
j=0
total_accuracy = np.zeros(5)
#for loop for different n values
for n in range(40,240,40):
    for k in range(0,1000):
    
        # get n random samples from both diabetic and non diabetic samples
        X_dia = diabetic.sample(n)
        XX_dia = diabetic[~diabetic.isin(X_dia)].dropna()
        X_nodia = no_diabetic.sample(n)
        XX_nodia = no_diabetic[~no_diabetic.isin(X_nodia)].dropna()
        result_X = [X_dia, X_nodia]
        result_XX = [XX_dia, XX_nodia]
        
        #train and test data
        X = pd.concat(result_X)
        XX = pd.concat(result_XX)
        
        y_train = X.iloc[:,[8]]
        y_test = XX.iloc[:,[8]]
        X_train =X.iloc[:, [0,1,2,3,4,5,6,7]]
        X_test = XX.iloc[:, [0,1,2,3,4,5,6,7]]
              
        lm = LinearRegression()        
        lm.fit(X_train,y_train)       
        predictions = lm.predict(X_test)
        predictions[predictions >= 0.5] = 1  
        predictions[predictions < 0.5 ] = 0
        accuracy[k]  = (accuracy_score(y_test, predictions, normalize=False) / len(predictions))
        #total_predictions= pd.concat((total_predictions,pd.DataFrame(predictions)),axis=1)
    
    total_accuracy[j] = np.mean(accuracy)*100
    print("Prediction Accuracy rate for n = ", n, "is" , total_accuracy[j],"%")
    j = j +1
    
xs=[40,80,120,160,200]  
ys = total_accuracy
plt.plot(xs,ys,'bo-')  
for x,y in zip(xs,ys):

    label = "{:.2f}".format(y)

    plt.annotate(label,
                 (x,y),
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center')
    
plt.xlabel('n (number of training samples)')
plt.ylabel('Prediction accuracy rate %')
plt.show()
              
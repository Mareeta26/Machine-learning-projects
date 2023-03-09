#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:50:57 2020

@author: mareeta
"""

from sklearn.decomposition import PCA as pca
import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0)
        if img is not None:
            images.append(img)
    return images

images=[]

#load images
for i in range(1,11,1):
    images.append(load_images_from_folder("/Users/mareeta/Desktop/phd/GAN/COEN240/hw6/att_faces_10/s"+str(i)))

#training and test data
img =  np.zeros((100,10304)) 
img_ind = np.zeros((100,1))
ctr=0  

for i in range(10):
    for j in range(10):
        img[ctr]= images[i][j].reshape(-1)
        img_ind[ctr]=i
        ctr +=1   
accuracy_pca = np.zeros((7,20))
accuracy_lda = np.zeros((7,20))

for exp in range(20):   
    X_train =  np.empty((0,10304))
    Y_train = np.empty((0,1))
    
    X_test =  np.empty((0,10304))
    Y_test = np.empty((0,1))
    
    tmp_xtrain = np.zeros((10,10304))  
    tmp_ytrain = np.zeros((10,1))
    tmp_xtest = np.zeros((2,10304))  
    tmp_ytest = np.zeros((2,1))
    
    for ind in range(0,100,10):
        tmp_xtrain,  tmp_xtest, tmp_ytrain, tmp_ytest = train_test_split(img[ind:ind+10,:], img_ind[ind:ind+10,:], test_size=0.2, random_state=42)
        X_train= np.append(X_train,tmp_xtrain,axis=0)
        Y_train = np.append(Y_train,tmp_ytrain,axis=0)
        X_test=np.append(X_test,tmp_xtest,axis =0)
        Y_test= np.append(Y_test,tmp_ytest,axis=0)
    
    d0 = 40
    #PCA
    ctr=0
    d_val = [1,2,3,6,10,20,30]
    for d in d_val:
        #PCA to d
        pca0 = pca(n_components=d)
        pca_train = pca0.fit_transform(X_train) 
        pca_test = pca0.transform(X_test) 
        
        #nearest neighbour -pca
        model_pca = KNeighborsClassifier(1)
        model_pca.fit(pca_train,Y_train.ravel())
        predict_pca = model_pca.predict(pca_test)
        
        #LDA
        #pca to d0=40
        pca1 = pca(n_components=d0)
        pca1_train = pca1.fit_transform(X_train)
        pca1_test = pca1.transform(X_test)
        
        #lda
        lda0 = lda(n_components=d)
        lda_train = lda0.fit_transform(pca1_train,Y_train)
        lda_test = lda0.transform(pca1_test)
        
        #NEAREST NEIGHBOUR -LDA
        model_lda = KNeighborsClassifier(1)
        model_lda.fit(lda_train,Y_train.ravel())
        predict_lda= model_lda.predict(lda_test)
        
        
        correct_count_lda = 0.0
        correct_count_pca = 0.0
        for i in range(len(Y_test)):
            if predict_pca[i] == Y_test[i]:
                correct_count_pca += 1.0
            if predict_lda[i] == Y_test[i]:
                correct_count_lda += 1.0
        accuracy_pca[ctr][exp] = correct_count_pca/float(len(Y_test))
        accuracy_lda[ctr][exp] = correct_count_lda/float(len(Y_test))
        #print("Accuracy for d:",d," is ", accuracy[ctr])
        ctr +=1

accuracy_mean_pca=np.mean(accuracy_pca,axis =1)*100 
accuracy_mean_lda=np.mean(accuracy_lda,axis =1)*100
plt.plot(d_val,accuracy_mean_pca,'go--',linewidth=2,markersize=2,label="PCA")
plt.plot(d_val,accuracy_mean_lda,'rx-.',linewidth=2,markersize=5,label="LDA")
plt.xlabel("Number of projection vectors (d)")
plt.ylabel("Recognition accuracy rate(%)")
plt.title("PCA vs LDA")
plt.grid(True)
plt.legend()
plt.show()

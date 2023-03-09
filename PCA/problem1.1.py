#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 12:56:38 2020

@author: mareeta
"""
from sklearn.decomposition import PCA as pca
import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
accuracy = np.zeros((7,20))

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
    
    
    #PCA
    
    ctr=0
    d_val = [1,2,3,6,10,20,30]
    for d in d_val:
        pca0 = pca(n_components=d)
        pca_train = pca0.fit_transform(X_train) 
        pca_test = pca0.transform(X_test) 
        model = KNeighborsClassifier(1)
        model.fit(pca_train,Y_train.ravel())
        predict = model.predict(pca_test)
        correct_count = 0.0
        for i in range(len(Y_test)):
            if predict[i] == Y_test[i]:
                correct_count += 1.0
        accuracy[ctr][exp] = correct_count/float(len(Y_test))
        #print("Accuracy for d:",d," in exp", exp,"is ", accuracy[ctr][exp])
        ctr +=1
accuracy_mean=np.mean(accuracy,axis =1)*100                   

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(d_val,accuracy_mean,'go--',linewidth=2,markersize=10)
plt.xlabel("Number of projection vectors(d)")
plt.ylabel("Recognition accuracy rate (%)")
plt.title("PCA")
for i,j in np.broadcast(d_val,accuracy_mean):
    ax.annotate(round(j,4),xy=(i+1,j))
plt.show()

   
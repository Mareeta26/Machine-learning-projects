#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:03:43 2020

@author: mareeta
"""

import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train,(60000,28*28))
x_test = np.reshape(x_test,(10000,28*28))
x_train, x_test = x_train / 255.0, x_test / 255.0
logreg = LogisticRegression(solver='saga', multi_class='multinomial',max_iter = 100,verbose=2)
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    stratify=y_train, random_state=42,test_size=0.25)
logreg.fit(X_train, y_train)
y_test_hat = logreg.predict(X_test)
num_correct = 0
for i in range(len(y_test)):
    if y_test_hat[i]==y_test[i]:
        num_correct +=1
        
Accuracy_rate = num_correct/len(y_test)
print("Accuracy Rate = ", Accuracy_rate)

cm = confusion_matrix(y_test, y_test_hat, labels=[0, 1, 2])
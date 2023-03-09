#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:53:21 2020

@author: mareeta
"""

from PIL import Image
import numpy as np
np.seterr(over='ignore')
import math


def calculate_variance(arr):
    arr = arr[~np.isnan(arr)]
    average = np.nanmean(arr)
    variance = 0
    for i in arr:
        variance = variance + ((average - i) ** 2)
    
    return variance/len(arr)
    
def normpdf(x, mean, var):
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom                                              
#read images and masks
face_im = Image.open("family.jpg").convert('RGB')
face_mask =  Image.open("family.png").convert('RGB')
portrait_im =  Image.open("portrait.jpg").convert('RGB')
portrait_mask =  Image.open("portrait.png").convert('RGB')

#transforming to array
train_im = np.asarray(face_im,dtype='int64')
train_mask = np.asarray(face_mask, dtype='int64')
test_im = np.asarray(portrait_im, dtype='int64')
test_mask = np.asarray(portrait_mask, dtype='int64')


#count bg and skin pixels of train_mask
n_bg = 0
n_skin = 0
width, height,channels = train_im.shape

for pixel in face_mask.getdata():
    if pixel == (0, 0, 0):
        n_bg += 1
    else:
        n_skin += 1

#prior probabilities
p_h0 = n_bg/(n_skin+n_bg)
p_h1 = n_skin/(n_skin+n_bg)

#threshold
gamma = p_h0 / p_h1

r0_k = np.zeros(n_bg)
g0_k = np.zeros(n_bg)

r1_k = np.zeros(n_skin)
g1_k = np.zeros(n_skin)
i= 0
j = 0

red = 0
green = 0
blue = 0

for x in range(width):
  for y in range(height):
      red, green, blue = train_im[x,y]
      #bg pixels
      if all(train_mask[x,y] == (0,0,0)):
          r0_k[i] = red / (red + green + blue)
          g0_k[i] = green/ (red + green + blue)
          i = i+1
      #skin_pixels 
      else:
          r1_k[j] = red/ (red + green + blue)
          g1_k[j] = green/ (red + green + blue)
          j = j +1 
          
#constructing classifier        
mean_0r = np.nanmean(r0_k)
mean_0g = np.nanmean(g0_k)
sigma_0r = calculate_variance(r0_k)
sigma_0g = calculate_variance(g0_k)

mean_1r = np.mean(r1_k)
mean_1g = np.mean(g1_k)
sigma_1r = calculate_variance(r1_k)
sigma_1g = calculate_variance(g1_k)

#testing image

width, height,channels = test_im.shape
result = np.zeros((width, height,3),dtype=np.uint8)
for x in range(width):
  for y in range(height):
      red, green, blue = test_im[x,y]
      r = red / (red + green + blue)
      g= green/ (red + green + blue)
      x_h0 = normpdf(r,mean_0r,sigma_0r)*normpdf(g,mean_0g,sigma_0g) 
      x_h1 = normpdf(r,mean_1r,sigma_1r)* normpdf(g,mean_1g,sigma_1g) 
      #x_h0= joint_pdf(mean_0r,sigma_0r,red)*joint_pdf(mean_0g,sigma_0g,green)
      #x_h1= joint_pdf(mean_1r,sigma_1r,red)*joint_pdf(mean_1g,sigma_1g,green)
      if (x_h1/x_h0) > gamma:
          result[x,y] = [255, 255,255]
      else:
          result[x,y] = [0,0,0]
    
img = Image.fromarray(result, 'RGB')
img.show()

#accuracy
tp = 0
tn= 0
fp = 0
fn =0

n_bg =0
n_skin =0 
for pixel in portrait_mask.getdata():
    if pixel == (0, 0, 0): 
        n_bg += 1
    else:
        n_skin += 1

#results
for x in range(width):
  for y in range(height):
      if all(test_mask[x,y] == [0,0,0]) & all(result[x,y] == [0,0,0]):
          tn += 1
      if all(test_mask[x,y] == [255,255,255]) & all(result[x,y] == [255,255,255]):
          tp += 1
      if all(test_mask[x,y] == [255,255,255]) & all(result[x,y] == [0,0,0]):
          fn += 1
      if all(test_mask[x,y] == [0,0,0]) & all(result[x,y] == [255,255,255]):  
          fp += 1

print("True Positive rate",tp*100/n_skin,"%" )
print ("True Negative rate",tn*100/n_bg,"%")
print ("False Positve rate",fp*100/n_bg,"%")
print ("False Negative rate",fn*100/n_skin,"%")


# hello-world
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:58:48 2015

@author: hlq
"""

import theano
import theano.tensor as T
import numpy as np
np.random.seed(1337)  # for reproducibility



from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Layer,Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb


import cPickle
import os
import sys  
import math
import time

'''
print("begin")
f=open(r'/home/hlq/ASA/casp8/casp08_data.11.25.pkl','r')
data=cPickle.load(f)
f.close()
data_set_x,data_set_y=data
length=len(data_set_y)
print("len:",length)
print("data_set_x[0]:",data_set_x[0])
print("data_set_y[0]:",data_set_y[0])


ls=[]

print(ls)
ls[0].append([1,2,3])
print(ls)
ls[0].append([1,2,3])
print(ls)
ls[1].append([2,3,4])
ls[1].append([2,3,4])
print(ls)

for i in range(100):
    ls.append([])
print(ls)


ls[0].append([1,2,3])
ls[0].append([2,3,4,5])


array=np.asarray(ls)
print(type(array))
'''

'''
a=[[],[],[1],[2],[]]
print type(a)
print len(a)
for i in range(len(a)-1,-1,-1):
    if a[i]==[]:
        a.remove([])

print len(a)
'''
'''
a=np.array([[1],[2],[3],[4],[5],[6]])
print a.ndim 

a=a.reshape(6,1,1)
print a
print a.ndim
'''
'''
pklFilePath="/home/hlq/Python/mypkl/"
fileList=os.listdir(pklFilePath)


if(os.path.exists(pklFilePath)==False):
    os.makedirs(pklFilePath)

def readPklFile(filePath):
    f=open(pklFilePath+filePath,'rb')
    x,y=cPickle.load(f)
    
    
    print(len(x))
    print("y len:"+"%i"%(len(y)))
    print(type(y))
    print(len(y))
    print(y)
  
    length=len(y)
    y=y.reshape(length,1,1)
    return x,y
    

i=1
if i<3:
    print 1
else:
    print 2

x_train,y_train=readPklFile("1a0a.pkl")
#print x_train
print x_train[0]
print x_train.ndim
print type(x_train)
print type(y_train)
print len(x_train)

if(len(x_train)==0):
    print "NULL"
if x_train==[]:
    print 1
else:
    print 2
'''



data=[[[0.1]],[[0.2]],[[0.6]]]
print len(data)
for i in range(len(data)):
    print data[i][0][0]
    if data[i][0][0]<0.5:
        data[i][0][0] = 0
    else:
        data[i][0][0] =1


for element in data:
    print element
    
print 1.


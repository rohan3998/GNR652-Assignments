import pandas as pd
import numpy as np
import math
import cvxopt
from cvxopt import solvers
from cvxopt import matrix
import scipy.io as sio
from sklearn import preprocessing
import scipy.sparse

mat=sio.loadmat('Indian_pines.mat')
x=mat['indian_pines_corrected']
x=x.reshape((145*145,200))
#print(x.shape) #21025x200
label=sio.loadmat('Indian_pines_gt.mat')
y=label['indian_pines_gt']
#print(y.shape)# 145x145
y=y.reshape((-1))
#print(y.shape)# 21025x1
#zero=[]
#for d in range(x.shape[0]):
#	if y[d]==0:
#		zero.append(d)
#print ("len_of_label_zeros->",len(zero)) #10776
#x=np.delete(x,zero,0)
#y=np.delete(y,zero,0)
#print (x.shape)#10249x200
#print (y.shape)#10249x1
np.set_printoptions(threshold=np.inf)
#print(mat['indian_pines_corrected'][1,1,:])
xtrain1=x[1:10000,:]
#print(xtrain1.shape)
xtest1=x[10001:20000,:]
#print(xtest1.shape)
ytrain=y[1:10000]
#print(ytrain.shape)
ytest=y[10001:20000]
#print(ytest.shape)
#print(np.shape(xtrain1))
theta=np.zeros((17,200))
one1=np.ones((17,1))
theta=np.concatenate((one1,theta),axis=1)
one=np.ones((9999,1))
#print(np.shape(one1))
#print(np.shape(xtrain1[1,:,:]))
xtrain=np.concatenate((one,xtrain1),axis=1)
#print(xtrain.shape)#5124x201
xtrain=preprocessing.scale(xtrain)
xtrain=np.array(xtrain)
#print(xtrain)
xtest=np.concatenate((one,xtest1),axis=1)
#print(xtest.shape)#5124x201
xtest=preprocessing.scale(xtest)
xtest=np.array(xtest)
theta=np.transpose(theta)

def getLoss(w,x,y,lam):
    m = x.shape[0] 
    y_mat = oneHotIt(y) 
    scores = np.dot(x,w) 
    prob = softmax(scores) 
    loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam/2)*np.sum(w*w) 
    grad = (-1 / m) * np.dot(x.T,(y_mat - prob)) + lam*w 
    return loss,grad

def oneHotIt(Y):
    m = Y.shape[0]
    OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    OHX = np.array(OHX.todense()).T
    return OHX
    
def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def getProbsAndPreds(someX):
    probs = softmax(np.dot(someX,theta))
    preds = np.argmax(probs,axis=1)
    return probs,preds

def getAccuracy(someX,someY):
    prob,prede = getProbsAndPreds(someX)
    accuracy = sum(prede == someY)/(float(len(someY)))
    return accuracy


lam = 1
iterations = 700
learningRate = 1e-4
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(theta,xtrain,ytrain,lam)
    losses.append(loss)
    theta = theta - (learningRate * grad)
print ('loss : ', loss)
print ('Test Accuracy: ', 100*getAccuracy(xtest,ytest))
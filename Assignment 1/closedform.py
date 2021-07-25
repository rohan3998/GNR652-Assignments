import csv
import math
import numpy as np
from numpy.linalg import inv
file = np.genfromtxt('dataset.csv', delimiter=',')
one1 = np.ones((108, 1))
np.random.shuffle(file)
xtrain = file[:108, :17]
xtrain = np.append(one1, xtrain, axis=1)
xtrainT=xtrain.transpose()
ytrain = file[:108, [17]]

one2 = np.ones((27, 1))
xtest = file[108:, :17]
xtest = np.append(one2, xtest, axis=1)
xtestT=xtest.transpose()
ytest = file[108:, [17]]
Id=np.ones((18,18))
lamda = 0.01
alpha=0.0001
theta = np.matmul(inv(np.matmul(xtrainT,xtrain) +lamda*Id ),np.matmul(xtrainT,ytrain))
mse = np.sum(np.square(np.matmul(xtest, theta)- ytest))/27
#print (theta)
print ("Mean square error for closed form is") 
print(mse)  



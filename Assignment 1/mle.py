import csv
import math
import numpy as np
from numpy.linalg import inv
file = np.genfromtxt('dataset.csv', delimiter=',')
one1 = np.ones((108, 1))
np.random.shuffle(file)
xtrain = file[:108, :17]
#print(xtrain)
xtrain = np.append(one1, xtrain, axis=1)
xtrainT = xtrain.transpose()
ytrain = file[:108, [17]]

one2 = np.ones((27, 1))
xtest = file[108:, :17]
xtest = np.append(one2, xtest, axis=1)
xtestT=xtest.transpose()
ytest = file[108:, [17]]
theta = np.ones((18, 1))
alpha = 0.0001
lamda = 0.01


def grad(x,y,theta):
	return (np.matmul(x.transpose(),np.matmul(x,theta)-y)+lamda*theta)/108


for i in range (1,100000):
	#func = np.matmul(xtrain, theta)- ytrain   
    #grad = np.matmul(xtrainT, func)
    theta=theta-alpha*grad(xtrain,ytrain,theta);


mse = np.sum(np.square(np.matmul(xtest, theta)- ytest))/27
var=np.mean(np.square(np.matmul(xtrain,theta)-ytrain))
stddev=math.sqrt(var)

print("standard deviation is ")
print (stddev)


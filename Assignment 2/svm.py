import numpy as np
import math
import cvxopt
from cvxopt import solvers
from cvxopt import matrix
from numpy.linalg import matrix_rank as Rank

myfile=np.genfromtxt('creditcard.csv',delimiter=',')
np.random.shuffle(myfile)
X_pos=myfile[:490,:30]
np.random.shuffle(X_pos)
Y_pos=myfile[:490,[31]]
np.random.shuffle(Y_pos)
X_neg=myfile[500:,:30]
np.random.shuffle(X_neg)
Y_neg=myfile[500:,[31]]
np.random.shuffle(Y_neg)
X1_train=X_pos[2:102,:30]
Y1_train=Y_pos[2:102,[0]]
X2_train=X_pos[594:695,:30]
Y2_train=Y_pos[594:695,[0]]
X1_test=X_neg[220:320,:30]
Y1_test=Y_neg[220:320,[0]]
X2_test=X_neg[1000:7100,:30]
Y2_test=Y_neg[1000:7100,[0]]

def ctd(A):
	B=np.zeros((len(A),len(A)))
	for i in range(0,len(A)):
		B[i,i]=A[i]
	return B

X_train=np.concatenate((X1_train,X2_train),axis=0)
Y_train=np.concatenate((Y1_train,Y2_train),axis=0)
X_test=np.concatenate((X1_test,X2_test),axis=0)
Y_test=np.concatenate((Y1_test,Y2_test),axis=0)
Z=np.ones((1,len(X_train[1])))
#y1=np.matmul(Y_train,Z)
M=len(X_train)
N=len(X_train[1])
#y1=matrix(y1)
p=np.zeros((M,N))
for i in range (0,M):
		p[i]=Y_train[i]*X_train[i]


X1=matrix(X_train)
Y1=matrix(Y_train)

def kernel(x,y):
	z=ctd(y)
	return np.matmul(np.matmul(z,x),np.matmul(z,x).transpose())

A=Y_train.transpose()
a=matrix(A)
b=np.zeros((1,1))
B=matrix(b)
G=-np.identity(M)
g=matrix(G)
h=np.zeros((M,1))
H=matrix(h)
P=kernel(X_train,Y_train)
p=matrix(P)
q=np.ones((M,1))
Q=matrix(q)


sol=solvers.qp(p,Q,g,H,a,B)


M1=len(X_test)

beta=np.matmul(ctd(sol['x']),Y_train)
w=np.matmul(X_train.transpose(),beta)
b1=1-np.matmul(X_test[150],w)
disc=np.matmul(X_test,w)-b1*np.ones((M1,1))
Y_exp=np.zeros((M1,1))
for i in range(0,M1):
	if disc[i]>0:
		Y_exp[i]=1
	else:
		Y_exp[i]=-1

count=0

for i in range(0,M1):
	if Y_exp[i]==Y_test[i]:
		count=count+1



print("Accuracy =" )
print((count/M1) * 100)



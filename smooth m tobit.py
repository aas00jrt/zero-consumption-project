__author__ = 'aas00jrt'
import numpy as np
rnorm = np.random.normal
runif = np.random.rand
zeros = np.zeros
dot=np.dot
eye = np.identity
transpose = np.transpose
diag = np.diag
shape = np.shape
chol = np.linalg.cholesky
from scipy.stats import distributions
import matplotlib.pyplot as plt
from math import *
from random import random
import statistics as stat
import scipy
import scipy.linalg
import scipy.stats as sp
from sys import exit

#### Generate data
t = 500
# Covariance

Sigmatrue = np.matrix(np.zeros((3,3)))
Sigmatrue[0,0] = 2
Sigmatrue[0,1] = 0.5
Sigmatrue[0,2] = 0.3
Sigmatrue[1,0] = Sigmatrue[0,1]
Sigmatrue[1,1] = 3
Sigmatrue[1,2] = 0.2
Sigmatrue[2,0] = Sigmatrue[0,2]
Sigmatrue[2,1] = Sigmatrue[1,2]
Sigmatrue[2,2] = 1.5

z1=rnorm(0,2,t)
z2=rnorm(0,2,t)

# Sigmatrue = np.matrix(np.zeros((4,4)))
# Sigmatrue[0,0] = 2
# Sigmatrue[0,1] = 0.5
# Sigmatrue[0,2] = 0.3
# Sigmatrue[0,3] = -0.2
# Sigmatrue[1,0] = Sigmatrue[0,1]
# Sigmatrue[1,1] = 3
# Sigmatrue[1,2] = 0.2
# Sigmatrue[1,3] = 0.1
# Sigmatrue[2,0] = Sigmatrue[0,2]
# Sigmatrue[2,1] = Sigmatrue[1,2]
# Sigmatrue[2,2] = 1.5
# Sigmatrue[2,3] = -0.3
# Sigmatrue[3,0] = Sigmatrue[0,3]
# Sigmatrue[3,1] = Sigmatrue[1,3]
# Sigmatrue[3,2] = Sigmatrue[2,3]
# Sigmatrue[3,3] = 1.3

def ldl(a):
    n=(shape(a)[0])
    l=eye(n)
    d=zeros((n,n))
    for i in range(0,n):
        did = diag(d)
        if i > 0:
            if i==1:
                lint=l[i,0]*l[i,0]
                dint=d[0,0]
            else:
                lint=l[i,0:i]*l[i,0:i]
                dint=did[0:i]
            ldint=np.dot(transpose(lint),dint)
        else:
            ldint=0
        d[i,i]=a[i,i]-ldint
        for j in range(i+1,n):
            if i > 0:
                if i==1:
                    lint=l[j,0]*l[i,0]
                    lint=dot((lint),did[0])
                else:
                    lint=l[j,0:i]*l[i,0:i]
                    lint=dot((lint),did[0:i])/did[i]
            else:
                lint=0
            l[j,i]=(a[j,i]-lint)/did[i]
    return(l,d)

out=ldl(Sigmatrue)
l=out[0]
d=out[1]
ldl=dot(dot(l,d),transpose(l))

print("Sigmatrue", Sigmatrue)
print("ldl", ldl)
print("l", l)
print("d", d)

AAAAA
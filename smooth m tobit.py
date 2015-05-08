__author__ = 'aas00jrt'
import numpy as np
# TODO the code for the wishart draw comes out of github and should probably be checked
import invwishart as iw
rnorm = np.random.normal
runif = np.random.rand
zeros = np.zeros
ones=np.ones
dot=np.dot
eye = np.identity
transpose = np.transpose
diag = np.diag
shape = np.shape
chol = np.linalg.cholesky
inv=np.linalg.inv
conc = np.concatenate
xp=np.expand_dims
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

#True values
d1t = 2
d2t = 3
gt=3
b1t=4
b2t=5
gibbsno=10
z1=xp(rnorm(0,2,t),axis=1)
z2=xp(rnorm(0,2,t),axis=1)
w=ones((t,1))

#starting values
phi=eye(3)
d1 = 2
d2 = 3
g=3
b1=4
b2=5

l=chol(Sigmatrue)
e=rnorm(0,1,(t,3))
e=(l*e.T).T
x1=z1*d1t+e[:,0]
x2=z2*d2t+e[:,1]
y=w*gt+x1*b1t+x2*b2t+e[:,2]
print(shape(w))
print(shape(x1))
print(shape(x2))
temp=conc((w,x1),1)
xmat = conc((temp,x2),1)
shapex=shape(xmat)
nobs=shapex[1]

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

for i in range(0,gibbsno):
    sigdraw=iw.invwishartrand(t-1,phi)
    e1=x1-z1*d1
    e2=x2-z2*d2
    e12=conc((e1.T,e2.T),axis=0)
    sig12=xp(sigdraw[0:2,2],axis=0)
    sig21=sig12.T
    sig22=sigdraw[0:2,0:2]
    cmeane3=(dot(dot(sig12,inv(sig22)),e12)).T
    cvare3=sigdraw[2,2]-dot(dot(sig12,inv(sig22)),sig21)
    ytilde1=(y-cmeane3)/cvare3
    xtilde=xmat/cvare3
    bmean=dot(inv(dot(xtilde.T,xtilde)),dot(xtilde.T,ytilde1))
    bdraw=rnorm(bmean,1)
    e3=y-dot(xmat,bdraw)
    gamma=bdraw[2]
    ytilde2=y-xmat[:,2]*gamma
    xxy=conc(conc(x1,x2),ytilde2)
    z1z=conc(z1,zeros(nobs,1))
    zz2=conc(zeros(nobs,1),z2)
    b1z1=bdraw[0]*z1
    b2z2=bdraw[1]*z2
    bigz=conc(conc(z1z,zz2),conc(b1z1,b2z2,1))
    a=[[1, 0, 0],[1, 0, 0],[bdraw[[0],bdraw[1],0]]],
    omega=dot(dot(a,sigdraw),a.T)
    iu=inv(chol(omega))
    uxxy=dot(iu.T,xxy)
    ubigz=dot(iu.T,bigz)
    dmean=dot(inv(dot(ubigz.T,ubigz)),dot(ubigz.T,uxxy))
    ddraw=rnorm(dmean,1)



out=ldl(Sigmatrue)
l=out[0]
d=out[1]
ldl=dot(dot(l,d),transpose(l))

# print("Sigmatrue", Sigmatrue)
# print("ldl", ldl)
# print("l", l)
# print("d", d)
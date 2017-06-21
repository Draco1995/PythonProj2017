# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:12:49 2017

@author: MSI
"""

# Demo file for Spyder Tutorial
# Hans Fangohr, University of Southampton, UK

import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import sympy as sy
from mpl_toolkits.mplot3d import Axes3D

def read():
    mon_flux = open("donneesProjetMAP311.txt","r")
    y = mon_flux.readlines()
    return y

def print_myself(x):
    for string in x:
        print(string,end="")
def print_matrix(x):
    for i in x:
        for j in i:
            print(j,end = "\t")
        print()


def text_to_matrix(x):
    y = x[:]
    for string in y:
        if string.startswith("%") or string.startswith("\n"):
            x.remove(string)
    y = []
    for string in x:
        if string.startswith("x"):
            matrix = []
            continue
        elif string.startswith("]"):
            y.append(matrix)
        else:
            matrix.append(string.split())
    """
    print_matrix(y[0])
    print()
    print_matrix(y[1])
    """
    for i in range(2):
        for j in range(len(y[i])):
            for k in range(len(y[i][j])):
                y[i][j][k] = float(y[i][j][k])
    return y

def centreGravite(a):
    l = a[0].copy()
    for i in a[1:]:
        l += i
    return l/len(a)

def repereCentree(a):
    centre = centreGravite(A)
    X = A.copy()
    for i in range(len(A)):
        for j in range(len(A[0])):    
            X[i][j] = A[i][j] - centre[j]
    return X

def repereRenormalisee(a):
    et = ecartType(a)
    for i in range(len(A)):
        a[i] /= et
    return a

def ecartType(a):
    b = a.copy()
    b = np.transpose(a)
    c = []
    for i in b:
        c += [i.std()]
    return c

def question2(A):
    X = repereCentree(A)
    X = repereRenormalisee(X)
  #  print(centreGravite(X))
  #  print(ecartType(X))
    return X
'''
def max_index(a):
    index = 0
    max_valeur = a[0]
    for i in range(len(a)-1):
        if max_valeur<a[i+1]:
            max_valeur = a[i+1]
            index = i+1
    return index
'''     

def inertieDuNuage(X):
    S = np.transpose(X)*X
    a,b = np.linalg.eig(S)
    '''
    print(a)
    print()
    print(b)
    '''
    #index = max_index(a)
    return a,b


def question3(X):
    valeur_propre,vecteur_propre = inertieDuNuage(X)
    '''
    print(valeur_propre)
    print()
    print(vecteur_propre)
    '''
    return valeur_propre,vecteur_propre

def question4(X,valeur_propre,vecteur_propre):
    A = X.copy()
    plt.scatter(X[:,2],X[:,3])
    plt.figure()
    plt.scatter(X[:,3],X[:,6])
    plt.figure()
    plt.scatter(X[:,2],X[:,6])
    plt.figure()
    plt.scatter(X[:,1],X[:,6])
    plt.figure()
    plt.scatter(X[:,3],X[:,5])
    plt.figure()
    plt.scatter(X[:,1],X[:,4])
    flg = plt.figure()
    ax = flg.gca(projection = "3d")
    ax.scatter(X[:,2].tolist(),X[:,3].tolist(),X[:,6].tolist())
    flg = plt.figure()
    ax = flg.gca(projection = "3d")
    ax.scatter(X[:,2].tolist(),X[:,3].tolist(),X[:,1].tolist())
    flg = plt.figure()
    ax = flg.gca(projection = "3d")
    ax.scatter(X[:,4].tolist(),X[:,3].tolist(),X[:,0].tolist())
    flg = plt.figure()
    ax = flg.gca(projection = "3d")
    ax.scatter(X[:,0].tolist(),X[:,5].tolist(),X[:,6].tolist())
    
def question5(valeur_propre,vecteur_propre):
    print()
    print(valeur_propre[:3])
    print(vecteur_propre[:3])

def question6(X,valeur_propre,vecteur_propre):
    x = [[],[],[]]
    for i in range(3):
        x[i].append(np.dot(X,np.transpose(vecteur_propre[i])))
    plt.scatter(x[0],x[1])
    flg = plt.figure()
    ax = flg.gca(projection = "3d")
    ax.scatter(x[0],x[1],x[2])
    
def estiCov(X):
    length = X.shape[1]
    sample = len(X[:,0])
    sigma = np.matrix(np.eye(length))
    for p in range(length):
        a = X[:,p]
        a -= np.mean(a)
        for q in range(length):
            b = X[:,q]
            b -= np.mean(b)
            sigma[p,q] = np.dot(np.transpose(b),a)[0,0]/(sample-1)
    return sigma
            
def question8(X,number):
    i = 0
    j = 0
    ans = []
    for i in range(len(X[:,0])):
        for j in range(i,len(X[:,0])):
            if abs(X[i,j])<number:
                ans.append([i+1,j+1])
    return ans

def densite_logistique(x):
    return (2*np.exp(-2*x))/(1+np.exp(-2*x))/(1+np.exp(-2*x))



def question10a():
    x = np.linspace(-3,3,100)
    E = np.random.rand(1000)
    E = 0.5*np.log(E/(1-E))
    plt.hist(E,bins = 80,normed = 1, label = "Stimulation")
    plt.plot(x,densite_logistique(x), label = "densite de la loi logistique")
    
def question10b():
    A = np.matrix([
            [1,2,3],
            [3,5,2],
            [1,4,7],
            [2,3,4],
            [2,4,5],
            [2,9,2],
            [13,2,1]
            ])
    E = np.random.rand(3)
    E = 0.5*np.log(E/(1-E))
    X = np.dot(A,E)       #C'est un echantillon! XD
    return X

def gamma(W,X1):
    a = len(W[:,0])
    b = X1.shape[1]
    ans = np.eye(a,b)
    
    for j in range(a):
        for m in range(b):
            ans[j,m]=0
            for i in range(len(X1[:,0])):
                ans[j,m]+=(   densite_logistique(   np.dot(W,np.transpose(X1[i,:]))[j]  )     )*X1[i,m]
            ans/=len(X1[:,0])
    return ans
        

def gradAscent(W,n,p,X): #W Start n numbers p pace X echantillon
    for i in range(n):
        G = gamma(W,X)
        M = np.dot(G,np.dot(np.transpose(W),W))
        M2 = W+M
        W = W + p * M2
        #print(W)
    #print(W)
    return W



def question17(B):
   # plt.figure()
   # plt.scatter(B[:,0],B[:,1])
    W = np.matrix([
            [-4,-0.5],
            [-2,1],
            ])
    W = gradAscent(W,500,0.005,B[:,0:2])
    #print(W)
    x = np.linspace(-1,5,100)
    A = np.linalg.inv(W)
    #print(A)
   # plt.plot(x,A[0,0]/A[1,0]*x)
   # plt.plot(x,A[0,1]/A[1,1]*x)
    plt.figure()
    plt.scatter(B[:,0],B[:,1])
    plt.plot(x,A[1,0]/A[0,0]*x)
    plt.plot(x,A[1,1]/A[0,1]*x)
# main program starts here
x = read()
List_matrix = text_to_matrix(x)
A = np.matrix(List_matrix[0])
B = np.matrix(List_matrix[1])
X = question2(A)
#print(X)
#valeur_propre,vecteur_propre = question3(X)
#question4(X,valeur_propre,vecteur_propre)
#question5(valeur_propre,vecteur_propre)
#question6(X,valeur_propre,vecteur_propre)
#print("12132132321312312232")
#print(question8(estiCov(X),0.1))



#question10a()
#print(question10b())

B = question2(B)
question17(B)
valeur_prorpre,vecteur_propre = question3(B[:,0:2])
print(vecteur_propre)
x = np.linspace(-1,5,100)
plt.plot(x,vecteur_propre[1,0]/vecteur_propre[0,0]*x)
plt.plot(x,vecteur_propre[1,1]/vecteur_propre[0,1]*x)
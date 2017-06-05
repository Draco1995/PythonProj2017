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
    A = X.copy()
    x = [[],[],[]]
    for i in range(3):
        x[i].append(np.dot(X,np.transpose(vecteur_propre[i])))
    plt.scatter(x[0],x[1])
    flg = plt.figure()
    ax = flg.gca(projection = "3d")
    ax.scatter(x[0],x[1],x[2])
# main program starts here
x = read()
List_matrix = text_to_matrix(x)
A = np.matrix(List_matrix[0])
B = np.matrix(List_matrix[1])
X = question2(A)
valeur_propre,vecteur_propre = question3(X)
#question4(X,valeur_propre,vecteur_propre)
question5(valeur_propre,vecteur_propre)
question6(X,valeur_propre,vecteur_propre)
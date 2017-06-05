# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:12:49 2017

@author: MSI
"""

# Demo file for Spyder Tutorial
# Hans Fangohr, University of Southampton, UK

import numpy as np
import scipy as sp
import matplotlib as mp
import sympy as sy

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

def max_index(a):
    index = 0
    max_valeur = a[0]
    for i in range(len(a)-1):
        if max_valeur<a[i+1]:
            max_valeur = a[i+1]
            index = i+1
    return index
        

def inertieDuNuage(X):
    S = np.transpose(X)*X
    a,b = np.linalg.eig(S)
    '''
    print(a)
    print()
    print(b)
    '''
    index = max_index(a)
    return a[index],b[index]


def question3(X):
    valeur_propre,vecteur_propre = inertieDuNuage(X)
    print(valeur_propre)
    print()
    print(vecteur_propre)


# main program starts here
x = read()
List_matrix = text_to_matrix(x)
A = np.matrix(List_matrix[0])
B = np.matrix(List_matrix[1])
X = question2(A)
X = question3(X)
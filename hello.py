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
    print_matrix(y[0])
    print_matrix(y[1])

def hello():
    """Print "Hello World" and return None"""
    print("Hello World")


"""
I changed something
"""


# main program starts here
x = read()
text_to_matrix(x)
x[1].startswith("%")
x[1][0]
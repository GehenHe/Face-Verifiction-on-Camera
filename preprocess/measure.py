# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:05:44 2015
bigger means similiar
@author: teddy
"""
import numpy as np 

def cos(fea1,fea2):
    """
    every row is a fea
    """
    norm1 = np.linalg.norm(fea1,axis = 1)
    norm2 = np.linalg.norm(fea2,axis = 1)
    return np.sum(fea1*fea2,axis = 1)/norm1/norm2
def compute_dis(fea1,fea2):
    """
    every row is a fea
    """
    return -np.linalg.norm(fea1-fea2,axis = 1)

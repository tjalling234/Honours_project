 # -*- coding: utf-8 -*-
"""
Created on 10-04-2020

Author: Pieke Geraedts

Description: This script contains constraint methods that could be interesting to add when optimizing stationary probabilities. Each type of constraint can be added as contraint or as a penalty value.
    The initial goal is to let these constraints work for the setting of optimising a single 'player's' ranking.
"""
from MC_derivative import *
from Markov_chain.Markov_chain_new import MarkovChain
import numpy as np
import time
from itertools import permutations
from scipy.spatial.distance import euclidean

###CONSTRAINT METHODS###
def cons_diag(P_first, P_current):
    pass

def cons_zeros(P):
    pass

def cons_subset(P_first, P_current, subset, r):
    pass

def cons_delta(P_first, P_current, r):
    pass

###PENALTY METHODS###
def pen_diag(P_first, P_current):
    '''
    Purpose:
        A penalty that ensures that the diagonal entries in P are not changed.
    Input:
        P_current - the first transition matrix P
        P_current - the 'current' transition matrix P
    Output 
        the deviation of diagonal entries in P_current from P_first
    '''
    n,_ = P_first.shape
    return sum(abs(P_first[r,r] - P_current[r,r]) for r in range(n))


def pen_zeros(P_first, P_current):
    '''
    Purpose:
        A penalty that ensure that zero entries in P are not changed.
    Input:
        P_current - the first transition matrix P
        P_current - the 'current' transition matrix P
    Output 
        sum of all zero entries in P
    '''
    #NOTE: checked finding zeros for courtois, zacharys, landofoz
    n,_ = P_first.shape
    #all (i,j)
    lst_full = list(permutations(range(n),2)) + [(i,i) for i in range(n)]
    #transform all (i,j) to list format
    lst_full = list(list(lst_full[i]) for i in range(len(lst_full)))
    #(i,j) with P(i,j) != 0
    np_lst = np.argwhere(P_first)
    #transform (i,j) with P(i,j) != 0  to list format
    lst = list(list(np_lst[i]) for i in range(len(np_lst)))
    #(i,j) with P(i,j) == 0
    zeros = [item for item in lst_full if item not in lst]

    return sum(abs(P_current[ij[0],ij[1]]) for ij in zeros)

def pen_subset(P_first, P_current, subset, r):
    '''
    Purpose:
        A penalty that ensure that certrain P(r,j) entries in P are not changed.
    Input:
        P_current - the first transition matrix P
        P_current - the 'current' transition matrix P
        subset - a list of column indeces
        r - the row r of which the columns in subset are not allowed to change
    Output 
        the deviation P_current[r,j] from P_first[r,j] for j in subset
    '''
    return sum(abs(P_first[r,j] - P_current[r,j]) for j in subset)

def pen_delta(P_first, P_current, delta, r):
    '''
    Purpose:
        A penalty that ensure that P_current deviates at most delta (in norm) from P_first
    Input:
        P_current - the first transition matrix P
        P_current - the 'current' transition matrix P
        delta - real value, the allowed norm perturbation
        r - the row r of which the columns in subset are not allowed to change
    Output 
        the difference between delta and the norm deviation of P_current from P_first
    '''
    #NOTE: For now we take the euclidean distance (2-norm), consider other norms later.
    return abs(euclidean(P_first[r], P_current[r]) - delta)


if __name__ == '__main__':
    MC = MarkovChain('Courtois')
    P = MC.P
    pen_zeros(P, P)
    
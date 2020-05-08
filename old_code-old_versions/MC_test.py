 # -*- coding: utf-8 -*-
"""
Created on 19-03-2020

Author: Pieke Geraedts

Description: This script is used to test the MarkovChain functionality and to log any changes made to it.

ToDo:
-Get pygraphviz module working
"""
from Markov_chain.Markov_chain_new import MarkovChain
'''
Adjustments to MarkovChain package:
-Added '()' to all print statements
-In tools.py changed '\\data\\' to '/data/' 
-In Markov_chain_new 'import util' was used, but I needed (in accordance with sys.path) from Markov_chain import util. Simillarly for tools
'''
import numpy as np
from scipy.spatial.distance import euclidean
import time
import scipy.linalg as la
from Markov_Chain_chris import MarkovChain as MC_chris
import sys
#TODO: Add cache property, so that all values are calculated only once.
#NOTE: All computations have an error. P - toP(toTheta(P)) is not zero but some very small number. Similar for MC.D - D(Theta), etc. (largest error was order e-07)

def toTheta(P):
    '''
    Purpose: 
        Convert the P^0.5 transition matrix using cartesian coordinates to a Theta transition matrix using spherical coordinates
    Input:
        P - a (nxn)-matrix, the transition matrix with probabilities
    Output:
        Theta - a (nxn)-matrix representing each row of the P matrix in spherical coordinates, corresponding to a row in mSph. 
                i.e., each row in Theta respresents a row in P.
    '''
    n, _ = P.shape  
    P = np.sqrt(P)
    Theta = np.zeros((n,n-1))
    for j in range(n):
       # Theta[j,0] = euclidean(P[j,:],np.zeros(n))   #radius
        for i in range(n-1):    #determine angles
            if (np.array_equal(P[j,i:], np.zeros(n-i))):
                break #All angles remain at zero. Using below formulas will give division by 0
            if (i == n-2):  
                if (P[j,i] >= 0): 
                    Theta[j,i] = np.arccos(P[j,i]/euclidean(P[j,i:], np.zeros(n-i)))   
                else:
                    Theta[j,i] = 2*np.pi - np.arccos(P[j,i]/euclidean(P[j,i:], np.zeros(n-i)))   
            else:
                Theta[j,i] = np.arccos(P[j,i]/euclidean(P[j,i:], np.zeros(n-i)))

    return Theta

def toP(Theta):
    '''
    Purpose: 
        Convert the Theta transition matrix using spherical coordinates to a P transition matrix using cartesian coordinates
    Input:
        vSph - a (nxn)-matrix with values represented using spherical coordinates
    Output:
        P - a (nxn)-matrix representing each row of the Theta matrix in cartesian coordinates, corresponding to a row in P. 
                i.e., each row in P respresents a row in Theta.
    '''
    n,_ = Theta.shape   
    P = np.zeros((n,n))
    for j in range(n):
        for i in range(n):  
            if (i == n-1):
                P[j,i] = np.prod([np.sin(Theta[j,:]), np.ones(n-1)])
            else:
                P[j,i] = np.prod([np.sin(Theta[j,:i]), np.ones(i)])*np.cos(Theta[j,i])
        
    return P**2

def two_norm(vec):
    '''
    Purpose:   
        Compute the 2-norm of a vector, i.e., Euclidean distance
    Input:
        vec - compatible for numpy arrays and lists
    Output:
        two_norm - the two norm of vec
    '''
    return np.square(np.sum(vec[i]**2 for i in range(len(vec))))

def diag(A):
    '''
    Purpose:
        Return dg(A), where all off-diagonals are set to zero
    Input:
        A - (n,n)-matrix
    Output:
        dg_A - (n,n)-matrix
    '''
    n,_ = A.shape
    dg_A = np.zeros((n,n))
    np.fill_diagonal(dg_A, np.diag(A))

    return dg_A

def stationaryDist(Theta, mu):
    '''
    Purpose:
        Determine the stationary distribution of the Markov chain described by the transition matrix P(theta)
    Input:
        Theta - a (nx(n-1))-matrix of angels
        mu - a n-dim vector for the Markov chain's 'start values' or None 
    Output:
        pi - a n-dim vector with the (unique) steady state probabilities
    '''
    MC = MarkovChain(toP(Theta))
    if MC.bUniChain:   #P is a uni-chain
        pi = MC.pi[0]
    else:               #P is a multi-chain
        Pi = MC.Pi
        pi = mu@Pi    

    return pi

def ErgodicProjector(Theta):
    '''
    Purpose:
        Determine the ergodic projector of the Markov chain described by the transition matrix P(theta)
    Input:
        Theta - a (nx(n-1))-matrix of angels
    Output:
        Pi - a (nxn)-matrix
    '''
    return MarkovChain(toP(Theta)).Pi

def M(Theta):
    '''
    Purpose:
        Determine the mean first passage time matrix for the Markov chain described by the transition matrix P(theta)
    Input:
        theta - a (nx(n-1))-matrix of angels
    Output:
        M - a (nxn)-matrix. the mean first passage time matrix.
    '''
    #NOTE: this will return an error message if P(Theta) is a multi-chain
    #Q: can we make this work for multi-chains by considering mu? Similar to how mu solves uniqueness in calculating pi -> probably not
    return MarkovChain(toP(Theta)).M

def calcD(Theta):
    '''
    Purpose:
        Determine the deviation matrix for the Markov chain described by the transition matrix P(Theta)
    Input:
        Theta - a (nx(n-1))-matrix of angels
    Output:
        a (nxn)-matrix, the deviation matrix (D = sum^{inf}(P^n(Theta) - PI(Theta))).
    '''
    return MarkovChain(toP(Theta)).D

def derivativeP(Theta, r, c):
    '''
    Purpose:
        Calculate the partial derivative of P(Theta) w.r.t. Theta(r,c)
    Input:
        Theta - a (nx(n-1))-matrix of angels
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output:
        derivP - (nxn)-matrix of the derivative of the P(Theta) matrix 
    '''
    n,_ = Theta.shape
    theta_r= Theta[r]  #only angels of interest
    grad_P_r = np.zeros((n-1,n))
    
    if (n == 2):
        print ('Not yet solved')
        exit()
    
    for j in range(n):
        deriv = np.zeros(n-1)
        if (j == 0):
            deriv[0] = -2*np.sin(theta_r[0])*np.cos(theta_r[0])
        elif (j == n-1):
            for k in range(n-1):
                lst = list(set(range(n-1)) - set([k]))
                deriv[k] = 2*np.sin(theta_r[k])*np.cos(theta_r[k])*np.prod([np.sin(theta_r[l])**2 for l in lst])
        else:
            for k in range(j):
                lst = list(set(range(j)) - set([k]))
                deriv[k] = 2*np.sin(theta_r[k])*np.cos(theta_r[k])*np.cos(theta_r[j])**2 * np.prod([np.sin(theta_r[l])**2 for l in lst])
            deriv[j] = -2*np.sin(theta_r[j])*np.cos(theta_r[j])*np.prod([np.sin(theta_r[l])**2 for l in range(j)])
        grad_P_r[:,j] = deriv

    derivP = np.zeros((n,n))
    derivP[r] = grad_P_r[c]    
    
    return derivP

def derivative_pi(Theta, r, c):
    '''
    Purpose:
        Determine the derivative of the stationary distribution w.r.t. Theta(r,c)
    Input:
        Theta - a (nxn)-matrix of angles
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output:
        deriv_pi - the derivative of pi(Theta) vector
    '''
    n,_ = Theta.shape
    pi = stationaryDist(Theta, c)
    D = calcD(Theta)
    deriv_P = derivativeP(Theta, r, c)    
    #Usage of grad is confusing since it is a derivative. grad is still used for if the function is extended...
    grad_pi = np.zeros(n)
    grad_pi[r] = sum(pi[j]*sum(deriv_P[j,k]*D[k,c] for k in range(n)) for j in range(n))

    return grad_pi

def derivativePi(Theta, r ,c):
    '''
    Purpose:
        Calculate the derivative of the ergodic projector of P(Theta) w.r.t. Theta(r,c)
    Input:
        Theta - a (nx(n-1))-matrix of angles
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output: 
        derivPi - (nxn)-matrix of the derivative of the Pi(Theta) matrix 
    '''
    n,_ = Theta.shape
    MC = MarkovChain(toP(Theta))
    Pi = MC.Pi
    
    if (MC.bUniChain == False):
        print ('The MC is not a unichain, derivative not yet determined')
        return

    #We have uni-chain
    return np.array([derivative_pi(Theta,r,c)]*n)


def derivativeD(Theta, r, c):
    '''
    Purpose:
        Calculate the derivative of the deviation matrix of P(Theta) w.r.t. Theta(r,c)
    Input:
        Theta - a (nx(n-1))-matrix of angles
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output: 
        derivD - (nxn)-matrix of the derivative of the D(Theta) matrix 
    '''
    n,_ = Theta.shape
    D = calcD(Theta)
    P = toP(Theta)
    Pi = ErgodicProjector(Theta)
    derivP = derivativeP(Theta, r, c)
    derivPi = derivativePi(Theta, r, c)
    inv_term = np.linalg.inv((np.identity(n) - P + Pi))
    derivD = -inv_term@(-derivP + derivPi)@inv_term - derivPi      #Column r and row r are (for most) nonzero, does this make sense?
    
    return derivD

def derivativeM(Theta, r, c):
    '''
    Purpose:
        Calculate the derivative of the mean first passage time matrix of P(Theta) w.r.t. Theta(r,c)
    Input:
        Theta - a (nx(n-1))-matrix of angles
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output: 
        derivM - (nxn)-matrix of the derivative of the M(Theta) matrix 
    '''
    n,_ = Theta.shape
    pi = stationaryDist(Theta, None)
    Ones = np.ones((n,n))
    Pi = ErgodicProjector(Theta)
    dg_Pi = diag(Pi)
    D = calcD(Theta)
    dg_D = diag(D)
    derivD = derivativeD(Theta,r,c)
    dg_derivD = diag(derivD)
    #derivative of the inverse of dg_Pi
    deriv_dg_Pi_inv = np.zeros((n,n))
    deriv_dg_Pi_inv[r,r]  = derivative_pi(Theta,r,c)[r]/(pi[r]**2)
    #using derivative of M formula
    derivM = (-derivD + Ones@dg_derivD) @ np.linalg.inv(dg_Pi) + deriv_dg_Pi_inv@(np.identity(n) - D + Ones@dg_D)   #Column r and row r are (for most) nonzero, does this make sense?
    
    return derivM

def J(Theta):
    pass

def main():
    print ("===============================================================")
    start_time = time.time()

    #Testing
    n=6
    P = np.ones((n,n))/n

    #P = np.array([[1.,0.0],[0.0,1.]])
    r = 3   #0,1,..,n-1 The state for which the stationary distribution will be maximized
    c = 3 #None, 0, 1, .., n-2
    
    MC = MarkovChain(P)
    pi = MC.pi[0]
    n, m = P.shape
    Theta = toTheta(P)    
    print (np.round(derivativeM(Theta,r,c)),2)
    
    #Running time
    print ('Running time of the programme:', time.time() - start_time)
    print ("===============================================================")

if __name__ == '__main__':
    main()


MC = MarkovChain('Courtois')

 # -*- coding: utf-8 -*-
"""
Created on 04-04-2020

Author: Pieke Geraedts

Description: This script uses the MarkovChain class functionality to provide derivatives for relevant Markov chain matrices (i.e. P, D, M, Pi, pi) w.r.t. a single element of Theta.

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
import sys
import random
from matplotlib import pyplot as plt
#TODO: Make most functions compatible with multi-chains. This can be done by letting the optimization depend on some initial mu vector, so that pi remains unique.
#NOTE: Maybe instead of giving the P matrix to each function, we can give the MC to each function. This way the MarkovChain function is not calculated again several times. 
    #Does this really improve (quicken) anything?

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

def v_norm(vec, v):
    '''
    Purpose:   
        Compute the v-norm of a vector
    Input:
        vec - numpy array or list
        v - integer
    Output:
        v_norm - the v norm of vec
    '''
    pass

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

def stationaryDist(P, mu):
    '''
    Purpose:
        Determine the stationary distribution of the Markov chain described by the transition matrix P
    Input:
        P - a (nxn)-matrix (numpy array) of transition probabilities
        mu - a (nx1)-vector (numpy array) for the Markov chain's 'start values' or None 
    Output:
        pi - a (nx1)-vector (numpy array) with the (unique) steady state probabilities
    '''
    MC = MarkovChain(P)
    if MC.bUniChain:   #P is a uni-chain
        pi = MC.pi[0]
    else:               #P is a multi-chain
        Pi = MC.Pi
        pi = mu@Pi    

    return pi

def derivativeP(P, r, c):
    '''
    Purpose:
        Calculate the partial derivative of P w.r.t. Theta(r,c)
    Input:
        P - a (nxn)-matrix (numpy array) of transition probabilities
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output:
        derivP - (nxn)-matrix (numpy array) of the derivative of the P matrix 
    '''
    n,_ = P.shape
    Theta = toTheta(P)
    theta_r= Theta[r]  #only angels of interest
    grad_P_r = np.zeros((n-1,n))
    
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

def derivative_pi(P, r, c):
    '''
    Purpose:
        Determine the derivative of the stationary distribution w.r.t. Theta(r,c)
    Input:
        P - a (nxn)-matrix (numpy array) of transition probabilities
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output:
        deriv_pi - (nx1)-vector (numpy array) the derivative of pi
    '''
    n,_ = P.shape
    MC = MarkovChain(P)
    pi = MC.pi[0]
    D = MC.D
    deriv_P = derivativeP(P, r, c)    
    deriv_pi = pi @ deriv_P @ D

    return deriv_pi

def derivativePi(P, r ,c):
    '''
    Purpose:
        Calculate the derivative of the ergodic projector of P(Theta) w.r.t. Theta(r,c)
    Input:
        P - a (nxn)-matrix (numpy array) of transition probabilities
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output: 
        derivPi - (nxn)-matrix (numpy array) of the derivative of the Pi matrix 
    '''
    n,_ = P.shape
    MC = MarkovChain(P)
    Pi = MC.Pi
    
    if (MC.bUniChain == False):
        print ('The MC is not a unichain, derivative not yet determined')
        exit()

    #We have uni-chain
    return np.array([derivative_pi(P,r,c)]*n)

def derivativeD(P, r, c):
    '''
    Purpose:
        Calculate the derivative of the deviation matrix of P w.r.t. Theta(r,c)
    Input:
        P - a (nxn)-matrix (numpy array) of transition probabilities
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output: 
        derivD - (nxn)-matrix (numpy array) of the derivative of the D matrix 
    '''
    n,_ = P.shape
    MC = MarkovChain(P)
    Pi = MC.Pi
    derivP = derivativeP(P, r, c)
    derivPi = derivativePi(P, r, c)
    inv_term = np.linalg.inv((np.identity(n) - P + Pi))
    derivD = -inv_term@(-derivP + derivPi)@inv_term - derivPi      #Column r and row r are (for most) nonzero, does this make sense?
    
    return derivD
#NOTE: the derivative of M does not make compelete sence to me!
def derivativeM(P, r, c):
    '''
    Purpose:
        Calculate the derivative of the mean first passage time matrix of P(Theta) w.r.t. Theta(r,c)
    Input:
        P - a (nxn)-matrix (numpy array) of transition probabilities
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output: 
        derivM - (nxn)-matrix (numpy array) of the derivative of the M matrix 
    '''
    n,_ = P.shape
    Ones = np.ones((n,n))
    MC = MarkovChain(P)
    pi = MC.pi[0]
    Pi = MC.Pi
    D = MC.D
    dg_Pi = diag(Pi)
    dg_D = diag(D)
    derivD = derivativeD(P,r,c)
    dg_derivD = diag(derivD)
    #derivative of the inverse of dg_Pi
    deriv_dg_Pi_inv = np.zeros((n,n))
    deriv_dg_Pi_inv[r,r]  = -1*(derivative_pi(P,r,c)[r]) / (pi[r]**2)
    #using derivative of M formula
    derivM = (-derivD + Ones@dg_derivD) @ np.linalg.inv(dg_Pi) + deriv_dg_Pi_inv@(np.identity(n) - D + Ones@dg_D)   #Column r and row r are (for most) nonzero, does this make sense?
    
    return derivM
    
def objJ(P, r, gamma):
    '''
    Purpose:
        Compute the value of the objective function. The objective is to maximize pi[r] under the condition that the Markov chain remains irreducible,
             for this the (relevant) values of M are used as penalty in the objective function.
    Input:
        P - a (nxn)-matrix (numpy array) of transition probabilities
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        gamma - real number, the penalty factor for M
    Output:
        J - real number, the objective value
    '''
    MC = MarkovChain(P)
    pi = MC.pi[0]
    M = MC.M

    return pi[r] #+ gamma*(sum(M[r]) + sum(M[:,r]))

def deriv_objJ(P, r, c, gamma):
    '''
    Purpose:
        Compute the derivative of the objective function
    Input:
        P - a (nxn)-matrix (numpy array) of transition probabilities
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
        gamma - real number, the penalty factor for M
    Output:
        derivJ - real number, the derivative of the objective value
    '''
    deriv_pi = derivative_pi(P, r, c)
    derivM = derivativeM(P,r,c)
    
    return deriv_pi[r] #+ gamma*(sum(derivM[r]) + sum(derivM[:,r]))

def simulation(P, r, epsilon, gamma, N):
    '''
    Purpose:
        perform a simulation to optimize pi[r]
    '''
    Theta = toTheta(P)
    n,_ = P.shape
    mTheta_changes = np.zeros((N,n-1))
    mStatDist_changes = np.zeros((N,n))
    for step_nr in range(N):
        v_derivJ = np.zeros(n-1)    #store derivJ for all Theta[r,.] angles
        for c in range(n-1):    #c represents angle index
            derivJ = deriv_objJ(P,r, c, gamma)
            v_derivJ[c] = derivJ
        #find angle with highest effect on the objective value
        c_max = np.argmax(abs(v_derivJ))
        #c_change = random.randint(0,n-2)
        #Update angle Theta[r,c_max], we are maximizing
        Theta[r,c_max] = Theta[r,c_max] + epsilon*v_derivJ[c_max]       
        mTheta_changes[step_nr] = Theta[r]
        #New P matrix
        P = toP(Theta)
        MC = MarkovChain(P)
        epsilon = epsilon
        #stop condition for reducibility
        if (MC.bUniChain == False):
            print ('The Markov chain described by P is no longer uni-chain! This should not happen.')
            exit()
        
        #print current progress
        #print ('r=', r)
        #print ('gradient obj=', v_derivJ)
        #print ('P leaving state r=', P[r])
        #print ('P entering  state r=', P[:,r])
        #print ('pi=', MC.pi[0])
        mStatDist_changes[step_nr] = MC.pi[0]

    #print Final progress (again)
    print ('\n\n\n')
    print ('Final stationary probabilities:', MC.pi[0])
    print ('Final Objective value:', objJ(P,r,gamma))
    print ('Final transition probabilities:\n', np.round(P,3))
    plotThetaprocess(mTheta_changes, mStatDist_changes)

def plotThetaprocess(mTheta_changes, mStatDist_changes):
    N, n = mStatDist_changes.shape
    for c in range(n-1):
        plt.plot(mTheta_changes[:,c], label=c)
        plt.legend()
    plt.show()

    for i in range(n):
        plt.plot(mStatDist_changes[:,i], label=i)
        plt.legend()
    plt.show()

    pass    

def main():
    print ("===============================================================")
    start_time = time.time()

    ####simulation####
    #Initialize
    n=4
    P = np.ones((n,n))/n
    MC = MarkovChain('Courtois')
    P = MC.P
    r = 2   #0,1,..,n-1 The state for which the stationary distribution will be maximized
    c = 1 #0, 1, .., n-2
    Theta = toTheta(P)
    epsilon = 0.2
    gamma = 0.001    #at M = 10/gamma (=100) the effect of M to the objective function will now start to become noticable 
    N = 500    #Number of iterations

    #Start
    print (np.round(P,3))
    print (np.round(MC.pi,3))
    simulation(P,r,epsilon,gamma,N)
    
    #Running time
    print ('Running time of the programme:', time.time() - start_time)
    print ("===============================================================")

if __name__ == '__main__':
    main()




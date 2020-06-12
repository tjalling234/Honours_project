# -*- coding: utf-8 -*-
"""
Created on 05-04-2020

Author: Pieke Geraedts

Description: 
Use update on p vector
"""
#NOTE: The simulation is VERY SLOW for large n. 
from MC_derivative import *
from MC_constraints import *
from Markov_chain.Markov_chain_new import MarkovChain
import numpy as np
import time
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt
import math
import copy

def plotThetaprocess(mTheta_changes, mStatDist_changes, mObj_changes, r, step_nr, epsilon):
   # power = -1*int(math.floor(math.log10(epsilon))) 
    N, n = mStatDist_changes.shape

    for c in range(n-1):
        plt.plot(mTheta_changes[:step_nr+1,c], label=c)
        plt.title('Progess of the Theta[r] row for r=%d and epsilon=%.3f' % (r, epsilon))
        plt.legend()
    plt.savefig('Graphs_Results/simulation_ranking/fig_theta_%d' % (r))
    plt.show()
    
    for i in range(n):
        plt.plot(mStatDist_changes[:step_nr+1,i], label=i)
        plt.title('Progess of the stationary probabilities for r=%d and epsilon=%.3f' % (r, epsilon))
        plt.legend()
    plt.savefig('Graphs_Results/simulation_ranking/fig_pi_%d' % (r))
    plt.show() 

    plt.plot(mObj_changes[:step_nr+1], label='J')
    plt.title('Progess of the objective function for r=%d and epsilon=%.3f' % (r, epsilon))
    plt.legend()
    plt.savefig('Graphs_Results/simulation_ranking/fig_J_%d' % (r))
    plt.show()

def stop_condition1(pi_old, pi_new, r):
    '''
    True if ranking of state r has improved, False otherwise.
    '''
    ranking_old = np.argsort(pi_old)
    ranking_new = np.argsort(pi_new)

    if np.where(ranking_new == r)[0] > np.where(ranking_old == r)[0]:
        return True
    else:
        return False

def objJ(Theta, r, gamma):
    '''
    Purpose:
        Compute the value of the objective function. The objective is to maximize pi[r] under the condition that the Markov chain remains irreducible,
             for this the (relevant) values of M are used as penalty in the objective function.
    Input:
        Theta - a (nxn)-matrix (numpy array) of transition probabilities
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        gamma - real number, the penalty factor for M
    Output:
        J - real number, the objective value
    '''
    MC = MarkovChain(toP(Theta))
    pi = MC.pi[0]

    return pi[r]

def deriv_objJ(Theta, r, c, gamma):
    '''
    Purpose:
        Compute the derivative of the objective function
    Input:
        Theta - a (nxn)-matrix (numpy array) of transition probabilities
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
        gamma - real number, the penalty factor for M
    Output:
        derivJ - real number, the derivative of the objective value
    '''
    deriv_pi = derivative_pi(Theta, r, c)

    return deriv_pi[r] 

def updateP(P, v_gradJ, epsilon, r, c):
    #Update entire row, then normalize
    #NOTE: Allowing for negative probabilities could be interesting in some applications (e.g. Lending money).
    #Every prob that become negative is set to zero
    n,_ = P.shape
    P[r] = P[r] + epsilon*v_gradJ
    for i in range(n):
        if P[r,i] < 0:
            P[r,i] = 0
    P[r] = P[r]/euclidean(P[r],np.zeros(n))
    
    return P

def simulation(Theta, r, epsilon, gamma, N):
    '''
    Purpose:
        perform a simulation to optimize pi[r]
    '''
    P = toP(Theta)
    P_old = P
    n,_ = P.shape
    pi_old = stationaryDist(Theta, None)

    for step_nr in range(N):
        v_gradJ = np.zeros(n)   
        for c in range(n):    
            derivJ = deriv_objJ(Theta,r, c, gamma)
            v_gradJ[c] = derivJ
        P = updateP(P, v_gradJ, epsilon, r, c)
        Theta = toTheta(P)
        pi = stationaryDist(Theta, None)
        
        print ('==========')
        print ('epsilon=', epsilon)
        print ('iteration=', step_nr)
        print ('pi_new', np.round(pi,4))
        print ('P[r]=', np.round(P[r],3))
        print ('Obj=', objJ(Theta, r, gamma))
        print ('==========')

def start():
    print ("===============================================================")
    start_time = time.time()
    ####simulation####
    #Initialize
    n=8
    MC = MarkovChain('Courtois')
    P = MC.P
    #P = np.ones((n,n))/n
    #MC = MarkovChain(P)
    r = 1   #0,1,..,n-1 The state for which the stationary distribution will be maximized
    #c = 5 #0, 1, .., n-2
    Theta = toTheta(P)
    gamma = 0.01
    epsilon = 0.1
    N = 5000    #Number of iterations
    
    #Start
    simulation(Theta,r,epsilon,gamma,N)
 
    #Running time
    print ('Running time of the programme:', time.time() - start_time)
    print ("===============================================================")

if __name__ == '__main__':
    start()
    

    
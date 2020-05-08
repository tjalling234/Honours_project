# -*- coding: utf-8 -*-
"""
Created on 05-04-2020

Author: Pieke Geraedts

Description: This script is a simulation for optimizing the ranking (i.e., stat. prob.) of a single player.
    Updating the whole theta row. Using the subset penalty. It converges very slowly for the diagonal.

"""
from MC_derivative import *
from MC_constraints import *
from Markov_chain.Markov_chain_new import MarkovChain
import numpy as np
import time
import copy
from matplotlib import pyplot as plt
import random

def plotThetaprocess(mTheta_changes, mStatDist_changes, mObj_changes, vPr_changes, r):
    N, n = mStatDist_changes.shape
    for c in range(n-1):
        plt.title('Progess of theta')
        plt.plot(mTheta_changes[:,c], label=c)
        plt.legend()
#        plt.savefig('Graphs_Results/tmpfig_theta')
    plt.show()

    for i in range(n):
        plt.plot(mStatDist_changes[:,i], label=i)
        plt.title('Progess of the stationary probabilities')
        plt.legend()
#        plt.savefig('Graphs_Results/tmpfig_pi')
    plt.show() 

    plt.plot(mObj_changes, label='J')
    plt.title('Progess of the objective function')
    plt.legend()
#    plt.savefig('Graphs_Results/tmpfig_J')
    plt.show() 

    plt.plot(vPr_changes, label='P[r,r]')
    plt.title('Progess of self probability for r=%d' % (r))
    plt.legend()
    #plt.savefig('Graphs_Results/fig_P_%d' % (r))
    plt.show()

def objJ(Theta, r, subset, gamma, psi, Theta_ini):
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
    #M = MC.M
    
    P_first = toP(Theta_ini)

    return pi[r] - psi * pen_subset(P_first, MC.P, subset, r)

def deriv_objJ(Theta, r, c, subset, gamma, psi, Theta_ini):
    '''
    Purpose:
        Compute the derivative of the objective function
    Input:
        Theta - a (nxn)-matrix (numpy array) of transition probabilities
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
        gamma - real number, the penalty factor for M
        psi - real number, the penalty factor for S
    Output:
        derivJ - real number, the derivative of the objective value
    '''
    deriv_pi = derivative_pi(Theta, r, c)
    #derivM = derivativeM(Theta,r,c)
    
    P_first = toP(Theta_ini)
    P_current = toP(Theta)
    if pen_subset(P_first, P_current, subset, r) == 0:
        deriv_pen_subset = 0
    else:
        deriv_pen_subset = derivative_pen_subset(Theta,r,c,subset,Theta_ini)
        
    return deriv_pi[r] - psi*deriv_pen_subset

def simulation(Theta, r, epsilon, subset, gamma, psi, N):
    '''
    Purpose:
        perform a simulation to optimize pi[r]
    '''
    Theta_ini = copy.deepcopy(Theta)
    P = toP(Theta)
    P_old = P
    n,_ = P.shape
    mTheta_changes = np.zeros((N,n-1))
    mStatDist_changes = np.zeros((N,n))
    vObj_changes = np.zeros(N)
    vPr_changes = np.zeros(N)
    pi_old = stationaryDist(Theta, None)
    pi = pi_old
    theta_1 = np.zeros(n-1) #theta r at iteration n-1
    theta_2 = np.zeros(n-1) #theta r at iteration n-2

    for step_nr in range(N):
        v_gradJ = np.zeros(n-1)    #store derivJ for all Theta[r,.] angles
        for c in range(n-1):    #c represents angle index
            derivJ = deriv_objJ(Theta, r, c, subset, gamma, psi, Theta_ini)
            v_gradJ[c] = derivJ
        #Theta[r] = Theta[r] + epsilon*v_gradJ
        c_max = np.argmax(abs(v_gradJ))
        Theta[r,c_max] = Theta[r,c_max] + epsilon*v_gradJ[c_max]
        #Update MC measures
        P = toP(Theta)
        pi = stationaryDist(Theta, None)
        MC = MarkovChain(P)
        #Track changes
        mStatDist_changes[step_nr] = pi
        vObj_changes[step_nr] = objJ(Theta, r, subset, gamma, psi, Theta_ini)
        mTheta_changes[step_nr] = Theta[r]
        vPr_changes[step_nr] = P[r,r]


        '''
        if (v_gradJ[c_max] < 0): 
            momentum[step_nr % M] = -1
        else: 
            momentum[step_nr % M] = 1
        #Update rule, based on momentum
        #if (sum(momentum) == M or sum(momentum) == -M): #M times the same direction: SPEED UP
        #    epsilon = epsilon*1.1
        #    momentum = np.zeros(M)
        if (momentum[step_nr % M]*momentum[(step_nr-1) % M] == -1): #consecutive directions are different: SLOW DOWN
            epsilon = epsilon*0.9
            #momentum = np.zeros(M)
        '''

        print ('==========')
        print ('epsilon=', epsilon)
        print ('iteration=', step_nr)
        print ('momentum nr=', step_nr % 5)
        print ('pi_new', np.round(pi,4))
        print ('P[r]=', np.round(P[r],3))
        print ('Obj=', np.round(vObj_changes[step_nr],3))
        print ('momentum', momentum)
        print ('Maximum value in gradient', v_gradJ[c_max])
        print ('==========')
        
    print ('Final Objective value:', round(objJ(Theta, r, subset, gamma, psi, Theta_ini),3))
    print ('Final transition probabilities:\n', np.round(MC.P[r],3))
    print ('Final stationary probabilities:\n', np.round(stationaryDist(Theta, np.ones(n)/n),3))
    #print ('Final M value:\n', np.round(MC.M*gamma,2))
    plotThetaprocess(mTheta_changes, mStatDist_changes, vObj_changes, vPr_changes, r)

def start():
    print ("===============================================================")
    start_time = time.time()
    ####simulation####
    #Initialize
    n=4
    MC = MarkovChain('Courtois')
    P = MC.P
    #P = np.ones((n,n))/n
    #MC = MarkovChain(P)
    r = 1   #0,1,..,n-1 The state for which the stationary distribution will be maximized
    #c = 5 #0, 1, .., n-2
    Theta = toTheta(P)
    epsilon = 0.1  #Epsilon = 0.005 and update (>200) epsilon decreasing factor with factor 10 worked for all r in example Courtois
    gamma = 10**-6
    N = 500    #Number of iterations
    
    subset = [2]
    psi = 2

    simulation(Theta, r, epsilon, subset, gamma, psi, N)
    
    #Running time
    print ('Running time of the programme:', time.time() - start_time)
    print ("===============================================================")

if __name__ == '__main__':
    start()
    
    
    




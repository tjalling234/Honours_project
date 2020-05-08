# -*- coding: utf-8 -*-
"""
Created on 05-04-2020

Author: Pieke Geraedts

Description: 
This script is a simulation for optimizing the ranking (i.e., stat. prob.) of a single player with a constraint to bound M.
We try both entire row update and single element update. Also, we try different epsilon choices.
 -> Updating the whole theta row Then using log(M) seems to work pretty well but still the simulation bounces back 
    (after most likely reaching optimal) to a suboptimal (local opt) solution and remains there. Using log(M) instead of M or sqrt(M) ensures that the 
    bounce back (i.e., suboptimal solution) is much closer to the optimal solution. 
 -> Single theta element update:
"""

from MC_derivative import *
from MC_constraints import *
from Markov_chain.Markov_chain_new import MarkovChain
import numpy as np
import time
from matplotlib import pyplot as plt
import random

def plotThetaprocess(mTheta_changes, mStatDist_changes, vObj_changes, vPr_changes, step_nr, r):
    N, n = mStatDist_changes.shape

    for c in range(n-1):
        plt.plot(mTheta_changes[:step_nr,c], label=c)
        plt.title('Progess of the Theta[r] row for r=%d' % (r))
        plt.legend()
    plt.savefig('Graphs_Results/simulation_M/fig_theta_%d' % (r))
    plt.show()

    for i in range(n):
        plt.plot(mStatDist_changes[:step_nr,i], label=i)
        plt.title('Progess of the stationary probabilities for r=%d' % (r))
        plt.legend()
    plt.savefig('Graphs_Results/simulation_M/fig_pi_%d' % (r))
    plt.show() 

    plt.plot(vObj_changes[:step_nr], label='J')
    plt.title('Progess of the objective function for r=%d' % (r))
    plt.legend()
    plt.savefig('Graphs_Results/simulation_M/fig_J_%d' % (r))
    plt.show()

    plt.plot(vPr_changes[:step_nr], label='P[r,r]')
    plt.title('Progess of self probability for r=%d' % (r))
    plt.legend()
    plt.savefig('Graphs_Results/simulation_M/fig_P_%d' % (r))
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
    M = MC.M

    return pi[r] - gamma*np.log((max(np.max(M[r]), np.max((M[:,r])), 1)))

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
    derivM = derivativeM(Theta,r,c)

    return deriv_pi[r] - gamma*np.log((max(np.max(derivM[r]), np.max((derivM[:,r])), 1)))       

def simulation(Theta, r, epsilon, gamma, N):
    '''
    Purpose:
        perform a simulation to optimize pi[r]
    '''
    P = toP(Theta)
    P_old = P
    n,_ = P.shape
    mTheta_changes = np.zeros((N,n-1))
    mStatDist_changes = np.zeros((N,n))
    vObj_changes = np.zeros(N)
    vPr_changes = np.zeros(N)
    pi_old = stationaryDist(Theta, None)
    pi_new = pi_old

    for step_nr in range(N):
        if step_nr > 150: epsilon = 1/(step_nr)
        if step_nr > 300: epsilon = 1/(2*step_nr)
        if step_nr > 450: epsilon = 1/(4*step_nr)
        if step_nr > 600: epsilon = 1/(8*step_nr)
        if step_nr > 750: epsilon = 1/(16*step_nr)
        if step_nr > 900: epsilon = 1/(32*step_nr)
        v_gradJ = np.zeros(n-1)    #store derivJ for all Theta[r,.] angles
        for c in range(n-1):    #c represents angle index
            derivJ = deriv_objJ(Theta,r, c, gamma)
            v_gradJ[c] = derivJ
        #Update row Theta[r] 
        #Theta[r] = Theta[r] + epsilon*v_gradJ
        #Update singel Theta[r,c_max]
        c_max = np.argmax(abs(v_gradJ))
        Theta[r,c_max] = Theta[r,c_max] + epsilon*v_gradJ[c_max]
        #Update MC measures
        P = toP(Theta)
        pi = stationaryDist(Theta, None)
        MC = MarkovChain(P)
        #Track changes
        mStatDist_changes[step_nr] = pi
        vObj_changes[step_nr] = objJ(Theta, r, gamma)
        mTheta_changes[step_nr] = Theta[r]
        vPr_changes[step_nr] = P[r,r]
        
        print ('==========')
        print ('theta[r]=', np.round(Theta,3)[r])
        print ('Gradient Obj=', np.round(v_gradJ,3))
        print ('Gradient Obj x Epsilon=', np.round(v_gradJ*epsilon,3))
        print ('epsilon=', epsilon)
        print ('iteration=', step_nr)
        print ('pi_new', np.round(pi_new,4))
        print ('P[r]=', np.round(P[r],3))
        print ('Obj=', np.round(vObj_changes[step_nr],3))
        print ('==========')
    
    print ('Maximum M value=', np.max(MC.M))
    print ('Final self probability=', round(P[r,r],3))
    plotThetaprocess(mTheta_changes, mStatDist_changes, vObj_changes, vPr_changes, step_nr, r)

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
    epsilon = 0.1 
    gamma = 0.001
    N = 1000    #Number of iterations

    #Start
    #print ('###########')
    #print ('First stationary probabilities:\n', MC.pi[0])
    #print ('First Objective value:', objJ(Theta,r,gamma))
    #print ('First transition probabilities:\n', np.round(MC.P,3))
    #print ('First M value:\n', np.round(MC.M*gamma,2))
    #for r in range(8):
    simulation(Theta, r, epsilon, gamma, N)
    
    #Running time
    print ('Running time of the programme:', time.time() - start_time)
    print ("===============================================================")

if __name__ == '__main__':
    start()

    

# -*- coding: utf-8 -*-
"""
Created on 05-04-2020

Author: Pieke Geraedts

Description: 
This script is a simulation for optimizing the ranking (i.e., stat. prob.) of a single player, 
where we add an optional stopping condition for when player r has improved its ranking.
We try both entire row update and single element update. Also, we try different epsilon choices.
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
import sys

def plotThetaprocess(vTheta_size, mTheta_changes, mStatDist_changes, mObj_changes, vDerivJ_size, r, step_nr, epsilon):
   # power = -1*int(math.floor(math.log10(epsilon))) 
   
    N, n = mStatDist_changes.shape
    '''
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
    plt.savefig('Graphs_Results/simulation_pi/fig_pi_%d' % (r))
    plt.show() 
    '''

    plt.plot(mObj_changes[:step_nr+1], label='J')
    plt.title('Progess of the objective function for r=%d and epsilon=%.3f' % (r, epsilon))
    plt.legend()
    plt.savefig('Graphs_Results/simulation_pi/fig_J_%d' % (r))
    plt.show()

    plt.figure(1)
    plt.plot(vDerivJ_size[:step_nr+1], label=r'||$\nabla J$||')
    plt.title('Progress of derivative norm')
    plt.legend()
    plt.savefig('Graphs_Results/simulation_pi/fig_derivJ_%d' % (r))
    plt.show()

    plt.figure(2)
    plt.plot(vTheta_size[:step_nr+1], label=r'||$\theta$||')
    plt.title('Progress of theta row')
    plt.legend()
    plt.savefig('Graphs_Results/simulation_pi/fig_theta_%d' % (r))
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

def objJ(Theta, r):
    '''
    Purpose:
        Compute the value of the objective function. The objective is to maximize pi[r] under the condition that the Markov chain remains irreducible,
             for this the (relevant) values of M are used as penalty in the objective function.
    Input:
        Theta - a (nxn)-matrix (numpy array) of transition probabilities
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
    Output:
        J - real number, the objective value
    '''
    MC = MarkovChain(toP(Theta))
    pi = MC.pi[0]

    return pi[r]

def deriv_objJ(Theta, r, c):
    '''
    Purpose:
        Compute the derivative of the objective function
    Input:
        Theta - a (nxn)-matrix (numpy array) of transition probabilities
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output:
        derivJ - real number, the derivative of the objective value
    '''
    deriv_pi = derivative_pi(Theta, r, c)

    return deriv_pi[r] 

def simulation(Theta, r, epsilon, N):
    '''
    Purpose:
        perform a simulation to optimize pi[r]
    '''
    P = toP(Theta)
    P_old = P
    n,_ = P.shape
    mTheta_changes = np.zeros((N,n-1))
    vTheta_size = np.zeros((N,1))
    mStatDist_changes = np.zeros((N,n))
    vDerivJ_size = np.zeros((N,1))
    vObj_changes = np.zeros(N)
    vPr_changes = np.zeros(N)
    pi_old = stationaryDist(Theta, None)
    pi = pi_old
    theta_1 = np.zeros(n-1) #theta r at iteration n-1
    theta_2 = np.zeros(n-1) #theta r at iteration n-2

    for step_nr in range(N):
        v_gradJ = np.zeros(n-1)    #store derivJ for all Theta[r,.] angles
        for c in range(n-1):    #c represents angle index
            derivJ = deriv_objJ(Theta,r, c)
            v_gradJ[c] = derivJ
        Theta[r] = Theta[r] + epsilon*v_gradJ
        #Update gain sequence
        '''
        if ((Theta[r] - theta_1)@(theta_1 - theta_2) < 0):
            epsilon = epsilon*0.8
        if (step_nr == 0):
            theta_1 = copy.deepcopy(Theta[r])
        else:
            theta_2 = copy.deepcopy(theta_1)
            theta_1 = copy.deepcopy(Theta[r])
        '''
        
        #Update MC measures
        P = toP(Theta)
        pi = stationaryDist(Theta, None)
        MC = MarkovChain(P)
        delta = euclidean(P_old[r], P[r])
        #Track changes
        mStatDist_changes[step_nr] = pi
        vObj_changes[step_nr] = objJ(Theta, r)
        mTheta_changes[step_nr] = Theta[r]
        vTheta_size[step_nr] = euclidean(Theta[r], np.zeros(n-1))
        vDerivJ_size[step_nr] = euclidean(v_gradJ, np.zeros(n-1))
        vPr_changes[step_nr] = P[r,r]
        
        #if (stop_condition1(pi_old, pi_new, r) == True): break

        print ('==========')
        print ('epsilon=', epsilon)
        print ('iteration=', step_nr)
        print ('pi_new', np.round(pi,3))
        print ('P[r]=', np.round(P[r],3))
        print ('Obj=', np.round(vObj_changes[step_nr],3))
        print ('Gradient=', np.round(v_gradJ,3))
        print ('==========')
    plotThetaprocess(vTheta_size, mTheta_changes, mStatDist_changes, vObj_changes, vDerivJ_size, r, step_nr, epsilon)

    return pi

def start(r):
    #print ("===============================================================")
    start_time = time.time()
    ####simulation####
    #Initialize
    n=8
    MC = MarkovChain('Courtois')
    P = MC.P
    #P = np.ones((n,n))/n
    #MC = MarkovChain(P)
    #r = 2   #0,1,..,n-1 The state for which the stationary distribution will be maximized
    #c = 5 #0, 1, .., n-2
    Theta = toTheta(P)
    epsilon = 0.0005
    N = 1500    #Number of iterations
    

    #Start
    pi = simulation(Theta,r,epsilon,N)
    print ('Final stationary distribution for r=%d:\n' % r, np.round(pi,3))
    #Running time
    print ('Running time of the programme:', time.time() - start_time)
    print ("===============================================================")

if __name__ == '__main__':
    if (len (sys.argv) != 2):
        print ('File requires single input: r')
        exit()    
    start(int(sys.argv[1]))

    
    

    
# -*- coding: utf-8 -*-
"""
Created on 05-04-2020

Author: Pieke Geraedts

Description: This script is a simulation for optimizing the ranking (i.e., stat. prob.) of a single player.
    Updating theta is done for a sinlge element and by taking the element with maximum absolute value in the gradient

"""
from MC_derivative import *
from Markov_chain.Markov_chain_new import MarkovChain
import numpy as np
import time
from matplotlib import pyplot as plt
#TODO: Parallelizeren -> Make sure that the simulation method returns the final process (of its input). Then, let this simulation module work for N iterations. 
    #Use a different py file to call this modules, which should allow for easier parallelisation. If these steps are done for all simulation files, then
    #all simulations can easily be called from a central py file. Preventing all files from requiring a parallelisation procedure. 
    #OR/PERHAPS: using yield could also allow for easier parallelisation.
import random

def plotThetaprocess(mTheta_changes, mStatDist_changes, mObj_changes):
    N, n = mStatDist_changes.shape
    for c in range(n-1):
        plt.plot(mTheta_changes[:,c], label=c)
        plt.legend()
        plt.savefig('Graphs/tmpfig_theta')
    plt.show()

    for i in range(n):
        plt.plot(mStatDist_changes[:,i], label=i)
        plt.title('Progess of the stationary probabilities')
        plt.legend()
        plt.savefig('Graphs/tmpfig_pi')
    plt.show() 

    plt.plot(mObj_changes, label='J')
    plt.title('Progess of the objective function')
    plt.legend()
    plt.savefig('Graphs/tmpfig_J')
    plt.show() 

def stop_condition1(pi_old, pi_new):
    '''
    True if ranking has changed, False otherwise.
    '''
    if np.all(np.argsort(pi_old) == np.argsort(pi_new)):
        return False
    else:
        return True

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

    return pi[r] #- gamma*(max(np.max(M[r]), np.max((M[:,r]))))

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
    
    return deriv_pi[r] #- gamma*(max(np.max(derivM[r]), np.max((derivM[:,r]))))

def simulation(Theta, r, epsilon, gamma, N):
    '''
    Purpose:
        perform a simulation to optimize pi[r]
    '''
    P = toP(Theta)
    n,_ = P.shape
    mTheta_changes = np.zeros((N,n-1))
    mStatDist_changes = np.zeros((N,n))
    mObj_changes = np.zeros(N)

    for step_nr in range(N):
        if (step_nr > 100): epsilon = 1/(step_nr)
        v_derivJ = np.zeros(n-1)    #store derivJ for all Theta[r,.] angles
        for c in range(n-1):    #c represents angle index
            derivJ = deriv_objJ(Theta,r, c, gamma)
            v_derivJ[c] = derivJ
        #find angle with highest effect on the objective value
        c_max = np.argmax(abs(v_derivJ))
        #c_max = np.random.randint(n-1)
        #Update angle Theta[r,c_max], we are maximizing
        Theta[r,c_max] = Theta[r,c_max] + epsilon*v_derivJ[c_max]
        mTheta_changes[step_nr] = Theta[r]
        #New P matrix
        P = toP(Theta)
        MC = MarkovChain(P)
        mStatDist_changes[step_nr] = MC.pi[0]
        mObj_changes[step_nr] = objJ(Theta, r, gamma)
        
        
        print ('==========')
        print ('iteration=', step_nr)
        '''
        print ('epsilon=', epsilon)
        print ('P[r]=', np.round(P[r],3))
        print ('theta[r]=', np.round(mTheta_changes[step_nr],3))
        print ('pi=', np.round(mStatDist_changes[step_nr],3))
        print ('Derivative Obj=', np.round(v_derivJ,3))
        print ('c max=', c_max)
        print ('Obj=', np.round(mObj_changes[step_nr],3))
        print ('==========')
        '''
    #print and plot final results
    print ('Final Objective value:', objJ(Theta,r,gamma))
    print ('Final transition probabilities:\n', np.round(MC.P,3))
    print ('Final stationary probabilities:\n', MC.pi[0])
    #print ('Final M value:\n', np.round(MC.M*gamma,2))
    plotThetaprocess(mTheta_changes, mStatDist_changes, mObj_changes)

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
    r = 0   #0,1,..,n-1 The state for which the stationary distribution will be maximized
    #c = 5 #0, 1, .., n-2
    Theta = toTheta(P)
    epsilon = 0.1
    gamma = 0.0005
    N = 300    #Number of iterations

    #Start
    print ('###########')
    print ('First stationary probabilities:\n', MC.pi[0])
    print ('First Objective value:', objJ(Theta,r,gamma))
    print ('First transition probabilities:\n', np.round(MC.P,3))
    
    #print ('First M value:\n', np.round(MC.M*gamma,2))
    simulation(Theta, r, epsilon, gamma, N)
    
    #Running time
    print ('Running time of the programme:', time.time() - start_time)
    print ("===============================================================")

def plot():
    n=500
    Theta = toTheta(MarkovChain('Courtois').P)
    theta_rnd = np.array([0] + list(np.sort(np.random.rand(n-2)*np.pi/2)) + [np.pi/2])
    mObj = np.zeros((n,2))
    vP = np.zeros(n)
    gamma = 0.0000005
    r=7
    '''
    for i in range(n):
        Theta_tmp1 = Theta
        Theta_tmp1[r,6] = theta_rnd[i]
        mObj[i,0] = objJ(Theta_tmp1,r,gamma)
        vP[i] = toP(Theta_tmp1)[r,r]
        #Theta_tmp2 = Theta
        #Theta_tmp2[r,5] = theta_rnd[i]
        #Theta_tmp3 = Theta
        #Theta_tmp3[r,4] = theta_rnd[i]
        #Theta_tmp4 = Theta
        #Theta_tmp4[r,3] = theta_rnd[i]
        #Theta_tmp5 = Theta
        #Theta_tmp5[r,2] = theta_rnd[i]
        #Theta_tmp6 = Theta
        #Theta_tmp6[r,1] = theta_rnd[i]
        #Theta_tmp7 = Theta
        #Theta_tmp7[r,0] = theta_rnd[i]
        # mObj[i,1] = objJ(Theta_tmp2,r,gamma)
    plt.plot(theta_rnd, vP, label ='P[r,r]')
    plt.legend()
    plt.show()
    plt.plot(theta_rnd, mObj[:,0], label='J1')
    plt.legend()
    plt.show()
    '''
    mDerivJ = np.zeros((n,7))
    for i in range(n):
        Theta_tmp1 = Theta
        Theta_tmp1[r,0] = theta_rnd[i]    
        for c in range(7):
            mDerivJ[i,c] = deriv_objJ(Theta_tmp1,r,c,gamma)

    for c in range(7):
        plt.plot(theta_rnd, mDerivJ[:,c], label=c)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #plot()
    start()
    
    




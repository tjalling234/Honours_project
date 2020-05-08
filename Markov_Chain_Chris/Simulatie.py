# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:16:00 2020

@author: cfran
"""
#NOTE: Interesting (not unexpected) observation: the maximum a player can reach by changing its own angles can be lower than the value it reaches when other players optimize their own value.
#NOTE: Soms zijn de stappen nog negatief, i.e., 'slecht' voor de speler die will maximaliseren. Komt dit door te grote stappen van epsilon?
from Markov_Chain import MarkovChain
import numpy as np
import sys
import matplotlib.pyplot as plt


''' Iniitialize Markov chain '''

sclar1 = (1/3)
sclar2 = (1/9)
sclar3 = (1/9)
mTheta = np.array([[sclar3*np.pi, sclar1*np.pi, sclar1*np.pi],
                   [sclar3*np.pi, sclar1*np.pi, sclar2*np.pi],
                   [sclar3*np.pi, sclar2*np.pi, sclar2*np.pi],
                   [sclar3*np.pi, sclar2*np.pi, sclar2*np.pi]])

# =============================================================================
# P = np.array([[0.25, 0.25, 0.5],
#               [0.25, 0.25, 0.5],
#               [0.25, 0.5, 0.25]])
# =============================================================================
    
# =============================================================================
# mTheta = np.array([[(1/3)*np.pi, (1/2)*np.pi],
#                    [(1/3)*np.pi, (1/2)*np.pi],
#                    [(1/3)*np.pi, (1/2)*np.pi]])
# =============================================================================


''' Simulation parameters '''

M = 5000                                                                        # Number of update iterations 
epsilon = 0.1
player = 0                                                                      # Player to optimize


''' Functions '''

def create_mVar(player, i, n):
    '''
    Returns mVar for player with angle i to differentiate
    - i: angle index
    - n: number of states
    '''
    mVar = np.zeros( (n, n))
    I = np.eye(n)
    mVar[player] = I[i]

    return mVar[:,:-1]


''' Main '''

MC = MarkovChain(inpt=mTheta)

print('Optimizing player: ', player, '(Python idx)')
print('')

# Print first stats
print('Start P:')
print(MC.P.round(2), '\n')

print('Start stationary distribution of P: ')
print(MC.pi())
print('')

vIncrements = np.zeros(M-1)
vArg = np.zeros(M-1)
for step_nr in range(M-1):
    
    MC = MarkovChain(mTheta)
    Pi = MC.Pi()
    pi = MC.pi()
    D = MC.D()
    
    m_pi = np.zeros( (MC.n-2,MC.n) ) # Store all stat. distribution for adjusting different angles
    max_change_arg = -1
    max_change = 0.0
    for angle_idx in range(1,MC.n-1):
        
        mVar = create_mVar(player, angle_idx, MC.n)
        
        stat_derivative = (pi @ (MC.P_derivative(player, mVar) @ D))
        m_pi[angle_idx-1] = stat_derivative
        
        if stat_derivative.T[player] > max_change:
            max_change_arg = angle_idx
            max_change = stat_derivative.T[player]
                    
    # Update angle
    old_pi = MarkovChain(mTheta).pi()[0]
    mTheta[player,max_change_arg] = mTheta[player,max_change_arg] + epsilon * m_pi[max_change_arg-1,player]
       
    # Store max change (arg)
    vIncrements[step_nr] = m_pi[max_change_arg-1,player]
    vArg[step_nr] = max_change_arg

# Print final MC
print('Final P:')
print(MC.P.round(2), '\n')

print('Final Stationary distribution of P: ')
print(MC.pi())
print('')

# Plot increments
plt.figure()
plt.plot(vIncrements)
plt.xlabel('Step #')
plt.ylabel('Increment')
plt.show()
    


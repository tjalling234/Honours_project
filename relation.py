'''
Created:
    2 June 2020
 Purpose:
    Determine the relation between the p_ij's and theta_ij's.
    -> Plot the values in a row of P against a single theta. Do this for Courtois and P = (1/n)^{nxn}s
'''

import numpy as np
from MC_simulation_pi import *
from Markov_chain.Markov_chain_new import MarkovChain
from matplotlib import pyplot

MC = MarkovChain('Courtois')
P = MC.P

N = 500
n, _ = P.shape
#P = np.ones((n,n))/n
r = 4
c = 0

for c in range(n-1):
    Theta = toTheta(P)
    theta_rnd = np.sort(np.random.rand(N)*np.pi/2)
    p_vec = np.zeros((N,n))

    for i in range(N):
        Theta[r,c] = theta_rnd[i]
        p_vec[i] = toP(Theta)[r]

    lb_lst = range(n)+np.ones(n)
    for i in range(n):
        plt.plot (theta_rnd, p_vec[:,i], label=r'$p_%d$' % (i+1))
        plt.title(r'Behaviour of $p=(p_1,\dots,p_%d)$ w.r.t. $\theta_%d$' % (n,c+1))
        plt.legend()
        plt.show()

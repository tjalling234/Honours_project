import MC_simulation_M as simM
import MC_simulation_pi as simpi
from MC_derivative import *
from multiprocessing import Pool
import sys
import os
from Markov_chain.Markov_chain_new import MarkovChain
import numpy as np
import matplotlib.pyplot as plt

def run(r):
    os.system('Python3 MC_simulation_pi.py %d' % r)
    

if __name__ == '__main__':
#    p = Pool(7)
#    p.map(run, range(8))

    MC = MarkovChain('Courtois')
    P = MC.P
    n,_ = P.shape
    r=1
    Theta = toTheta(MC.P)
    Theta[r] = np.array([1.571, 0.023, 0.,    1.571, 0.321, 0.785, 1.571])
    gamma = 10**-7
    epsilon = 0.05
 
    N= 100
    theta_rnd = np.sort(np.random.rand(N)*np.pi/2)

    lst = [] 
    for c in range(n-1):
        for i in range(N):
            Theta[r,c] = theta_rnd[i]
            lst.append(simM.objJ(Theta, r, gamma))



        plt.plot(theta_rnd, lst[c*N:(c+1)*N])
        plt.show()
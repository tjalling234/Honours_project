import numpy as np
from Markov_chain.Markov_chain_new import MarkovChain
from MC_derivative import *
from MC_simulation_pi import *

MC = MarkovChain('Courtois')
P = MC.P
n,_ = P.shape
r = 6
gamma = 10*-6

#Not sure what this does. Seems like it checked specific cases.


#CASE 1
print ('============CASE 1==============')
P[r]= np.array([3.76281107e-12, 3.74939946e-33, 6.77637548e-10, 7.76037695e-09,
 3.74939943e-33, 9.89209654e-02, 8.11954322e-01, 8.91247046e-02])
Theta = toTheta(P)
print ('Theta[r]:', np.round(Theta[r],3))
print ('---')
for c in range(n-1):
    print (deriv_objJ(Theta, r, c, gamma))
print ('---')
print ('==========================')

#CASE 2
print ('============CASE 2==============')
P[r]= np.array([3.76281107e-12, 3.74939946e-33, 6.77637548e-10, 7.76037695e-09,
 3.74939943e-33, 9.89209654e-02, 8.11858645e-01, 8.92203815e-02])
P[r]= np.array([0, 0, 0, 0,
 0, 9.89209654e-02, 8.11858645e-01, 8.92203815e-02])
Theta = toTheta(P)
print ('Theta[r]:', np.round(Theta[r],3))
print ('---')
for c in range(n-1):
    print (deriv_objJ(Theta, r, c, gamma))
print ('---')
print ('==========================')

#OPTIMAL
print ('============OPTIMAL==============')
P[r]= np.array([0,0,0,0,0,0,1,0])
Theta = toTheta(P)
print ('Theta[r]:', np.round(Theta[r],3))
print ('---')
for c in range(n-1):
    print (deriv_objJ(Theta, r, c, gamma))
print ('---')
print ('==========================')
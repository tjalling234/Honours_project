import numpy as np
import time
from matplotlib import pyplot
from MC_derivative import *
from MC_simulation_ranking import *

'''
GOAL: 
    Analyse division of computation time for the simulation.
Method:
    Run most methods for multiple Theta instances and average computation time of the function.
'''
'''
def grad_pi_r(Theta, r):
    n, _ = Theta.shape
    grad_pi = []
    for c in range(n-1):
        grad_pi.append(derivative_pi(Theta,r,c)[r])

    return np.array(grad_pi)

start_time = time.time()
N = 200
n = 6
r = 3
P = np.ones((n,n))/n
Theta = toTheta(P)
theta_rc = np.sort(np.random.uniform(low=0.0,high=np.pi/2,size=N))
vDeriv_pi = np.zeros((N,n))

for c in range(n-1):
    for i in range(N):
        Theta[r,c] = theta_rc[i]
        Theta[r,0] = np.pi/2
        vDeriv_pi[i] = derivativeP(Theta, r, c)[r]
        min_loc = np.min(vDeriv_pi)

    plt.plot(theta_rc, vDeriv_pi, label=c)
    plt.legend()
    plt.show()

Theta[r,0] = np.pi/2
print (np.round(derivativeP(Theta, r, 0)[r],3))
print (np.round(derivativeP(Theta, r, 1)[r],3))
print (np.round(derivativeP(Theta, r, 2)[r],3))


print ('Elapsed time:', round(time.time() - start_time,4))
'''

P_first = MarkovChain('Courtois').P
Theta_first = toTheta(P_first)
P = P_first
Theta = Theta_first
n,_ = P.shape
N=1000
theta_rnd = np.sort(np.random.uniform(low=0.0,high=np.pi/2,size=N))
rnd_r = 4
rnd_c = 3

#TIME M DERIVATIVE
start_time = time.time()
num = 0
for theta_rc in theta_rnd:
    Theta[rnd_r,rnd_c] = theta_rc
    MC = MarkovChain(toP(Theta))
    for r in range(n):
        for c in range(n-1):
            MC.M
            #derivativeM(Theta, r, c)
            num += 1
print ('Duration of computing M for all r and c in Courtois:', time.time()-start_time)
avg_M = (time.time()-start_time)/num

#TIME P DERIVATIVE
start_time = time.time()
for theta_rc in theta_rnd:
    Theta[rnd_r,rnd_c] = theta_rc
    MC = MarkovChain(toP(Theta))
    for r in range(n):
        for c in range(n-1):
            MC.P
            #derivativeP(Theta,r,c)
print ('Duration of computing P for all r and c in Courtois:', time.time()-start_time)
avg_P = (time.time()-start_time)/num

#TIME D DERIVATIVE
start_time = time.time()
for theta_rc in theta_rnd:
    Theta[rnd_r,rnd_c] = theta_rc
    MC = MarkovChain(toP(Theta))
    for r in range(n):
        for c in range(n-1):
            MC.D
            #derivativeD(Theta,r,c)
print ('Duration of computing D for all r and c in Courtois:', time.time()-start_time)
avg_D = (time.time()-start_time)/num

#TIME pi DERIVATIVE
start_time = time.time()
for theta_rc in theta_rnd:
    Theta[rnd_r,rnd_c] = theta_rc
    MC = MarkovChain(toP(Theta))
    for r in range(n):
        for c in range(n-1):
            stationaryDist(Theta, None)
            #MC.pi
            #derivative_pi(Theta,r,c)
print ('Duration of computing pi for all r and c in Courtois:', time.time()-start_time)
avg_pi = (time.time()-start_time)/num

#TIME subset DERIVATIVE
start_time = time.time()
for theta_rc in theta_rnd:
    Theta[rnd_r,rnd_c] = theta_rc
    MC = MarkovChain(toP(Theta))
    for r in range(n):
        for c in range(n-1):
            pen_subset(P_first,MC.P, [],r)
            #derivative_pen_subset(Theta, r, c, [], Theta_first)
            
print ('Duration of computing subset for all r and c in Courtois:', time.time()-start_time)
avg_subset = (time.time()-start_time)/num



print ('Average computation time M with %d tries = %.6f' % (num, avg_M))
print ('Average computation time P with %d tries = %.6f' % (num, avg_P))
print ('Average computation time D with %d tries = %.6f' % (num, avg_D))
print ('Average computation time pi with %d tries = %.6f' % (num, avg_pi))
print ('Average computation time subset with %d tries = %.6f' % (num, avg_subset))

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:12:10 2020

@author: cfran
"""

''' imports '''

import numpy as np
import scipy.linalg as la
import numpy.linalg as npla
np.seterr(divide='raise')


''' Classes '''

class MarkovChain(object):
    '''
    Initialize Markov Chain object by either passing transition matrix P or mTheta.
    Other must be passed as None
    '''
    
    def __init__(self, inpt, verbose=False):
        
        self.verbose = verbose
        
        if inpt.shape[0] == inpt.shape[1]: # P was passed
            self.n = int(inpt.shape[0])
            self.P = inpt
            self.Theta = self.P_to_mTheta()
        elif inpt.shape[0] == inpt.shape[1]+1 : # mTheta was passed
            self.n = int(inpt.shape[0])
            self.Theta = inpt
            self.P = self.mTheta_to_P()
        else:
            raise ValueError('MarkovChain incorrectly initialized')
            
        self.ec, self.tc = self.MC_structure()
        self.bMultiChain = len(self.ec) > 1
        self.bUniChain = not self.bMultiChain
        self.bTransientStates = len(self.tc) > 0
        
    def I(self):
        '''
        Identity matrix. (from Markov_chain_new.py)
        '''
        return np.eye(self.n)
        
    def P_to_mTheta(self):
        ''' 
        Returns mTheta with all angles for a given P. 
        - Based on a transition matrix with single angles on diagonal
        '''   
        mTheta = np.zeros( (self.n, self.n-1) )
        
        # All first entries of mTheta first (needed for solving the system of equations)
        for i in range(self.n):            
            mTheta[i,0] = np.arccos( np.sqrt(self.P[i,i]) )
        
        # Rest of mTheta
        for i in range(self.n): # Rows
            for j in range(1,i): # Solve for all elements until i
                if i == j: # Already solved first entry of mTheta
                    continue
                else:
                    mTheta[i,j] = np.arccos( np.sqrt( self.P[i,j] / np.prod([np.sin(mTheta[i,k])**2 for k in range(j)]) ) )
            
            # Solve for diagonal element of mTheta
            if i <= self.n-2: # mTheta has (n x n-1) dimension
                mTheta[i,i] = np.arccos( np.sqrt( self.P[i,0] / np.prod([np.sin(mTheta[i,k])**2 for k in range(i)]) ) )
            
            # Continue solving for elemenets after diagonal elements of theta
            for j in range(i+1,self.n-1):
                mTheta[i,j] = np.arccos( np.sqrt( self.P[i,j] / np.prod([np.sin(mTheta[i,k])**2 for k in range(j)]) ) )
                          
        return mTheta
        
    def mTheta_to_P(self):
        '''
        Returns transition matrix P using spherical coordinates,
        - Transition matrix according self.Theta, where Cos(theta) is placed along diagonal,
          diagonal elements are placed at first entry of each row.
        '''
        P = np.zeros((self.n, self.n))
        for i in range(self.n): # rows
            for j in range(self.n): # columns
                if j == 0: # all elements for first column of P
                    if i == 0:
                        P[0,0] = np.cos(self.Theta[0,0])**2
                    else: #i >= 0, j == 0
                        if i <= self.n-2 : # Insert expression with cos
                            P[i,0] = np.prod( [np.cos(self.Theta[i,i])**2, np.prod([np.sin(self.Theta[i,k])**2 for k in range(i)])] )
                        else:
                            P[i,0] = np.prod( [np.sin(self.Theta[i,k])**2 for k in range(i)] )
                else: # j >= 0 (all other columns of P)   
                    if j == i: # (here cos(theta) swaps to first entry of matrix P)
                        P[i,j] = np.cos(self.Theta[i,0])**2
                    else: #i < j, i > j
                        if j <= self.n-2:
                            P[i,j] = np.prod( [np.cos(self.Theta[i,j])**2, np.prod([np.sin(self.Theta[i,k])**2 for k in range(j)])] )
                        else:
                            P[i,j] = np.prod( [np.sin(self.Theta[i,k])**2 for k in range(j)] )
        return P
    
    def P_derivative(self, i, mVar):
        #NOTE: Deze i wordt niet gebruikt?
        '''
        Returns the partial derivative of P (transition matrix)
        - i: player/state to differentiate
        - mVar: matrix shape equivalent to self.Theta, denotes the variable to which to differentiate
        '''
        # Issue warnings if intialisation went wrong
        if mVar.shape[0] != self.n or mVar.shape[1] != self.n-1:
            raise ValueError('Differentiation matrix not of same shape as P.')
        elif len(np.sum(mVar, axis=1)[np.sum(mVar, axis=1) > 1]) > 0 :
            raise ValueError('Can only differentiate to one varialbe per row of P.')
        else:
            pass
        
        P_der = np.zeros( (self.n, self.n) )
        for i in range(self.n): # rows
            
            # Skip row if no derivative is asked
            if np.sum(mVar[i,:]) == 0 :
                continue
                
            for j in range(self.n): # columns
                # all elements for first column of P
                if j == 0: # all elements for first column of P
                    if i == 0:
                        if mVar[0,0] == 1:
                            P_der[0,0] = -2*np.sin(self.Theta[0,0])*np.cos(self.Theta[0,0])
                        else:
                            P_der[0,0] = 0
                    else: # i > 0, j == 0
                        if i <= self.n-2 : # Insert expression with cos
                            if mVar[i,i] == 0: # sine term to differentiate
                                P_der[i,0] = np.prod( [np.cos(self.Theta[i,i])**2, \
                                np.prod([np.sin(self.Theta[i,k])**2 if mVar[i,k] == 0 \
                                         else 2*np.sin(self.Theta[i,k])*np.cos(self.Theta[i,k]) for k in range(i)])] )
                            else : # cosine term to differentiate
                                P_der[i,0] = np.prod( [-2*np.sin(self.Theta[i,i])*np.cos(self.Theta[i,i]), \
                                np.prod([np.sin(self.Theta[i,k])**2 for k in range(i)])] )
                        else:
                            P_der[i,0] = np.prod( [np.sin(self.Theta[i,k])**2 if mVar[i,k] == 0 \
                                 else 2*np.sin(self.Theta[i,k])*np.cos(self.Theta[i,k]) for k in range(i)] )
                
                # (all other columns of P)
                else: # j > 0
                    if j == i: # (here cos(theta) swaps to first entry of matrix P)
                        if mVar[i,0] == 1:
                            P_der[i,j] = -2*np.sin(self.Theta[i,0])*np.cos(self.Theta[i,0])
                        else:
                            P_der[i,j] = 0
                    else: #i < j, i > j
                        
                        # Skip row if no derivative is asked
                        if np.sum(mVar[i,:j+1]) == 0 :
                            continue
                        
                        if j <= self.n-2:                            
                            if mVar[i,j] == 0: # sine term to differentiate
                                P_der[i,j] = np.prod( [np.cos(self.Theta[i,j])**2, \
                                np.prod([np.sin(self.Theta[i,k])**2 if mVar[i,k] == 0 \
                                         else 2*np.sin(self.Theta[i,k])*np.cos(self.Theta[i,k]) for k in range(j)])] )
                            else : # cosine term to differentiate
                                P_der[i,j] = np.prod( [-2*np.sin(self.Theta[i,j])*np.cos(self.Theta[i,j]), \
                                    np.prod([np.sin(self.Theta[i,k])**2 for k in range(j)])] )
                        else:
                            P_der[i,j] = np.prod([np.sin(self.Theta[i,k])**2 if mVar[i,k] == 0 \
                                 else 2*np.sin(self.Theta[i,k])*np.cos(self.Theta[i,k]) for k in range(j)])
        return P_der
    
    def MC_structure(self):
        '''
        Determine the ergodic classes (ec) and transient classes (tc). (form Markov_chain_new.py)
        '''
        
        from tarjan import tarjan
        from util import create_graph_dict # util.py from package
        
        sccs = tarjan(create_graph_dict(self.P))  # strongly connected components (sccs)
        ec = []  # a list to keep track of the ergodic classes
        tc = []  # a list to keep track of transient states
        prec = 10**(-10)  # precision used

        for scc in sccs:
            if abs(np.sum(self.P[np.ix_(scc, scc)]) - len(scc)) < prec:
                # no 'outgoing' probabilities: scc is an ergodic class
                ec.append(scc)
            else:  # scc is a transient connected component
                tc.append(scc)

        if self.verbose:
            if len(ec) > 1:
                print('P describes a Markov multi-chain.')
            else:
                print('P describes a Markov uni-chain.')

        return ec, tc
    
    def Pi(self):
        '''
        Ergodic projector. (from Markov_chain_new.py)
        '''
        if self.bUniChain:
            # a unique stationary distribution
            return np.dot(np.ones((self.n, 1)), self.pi())

        # Markov multi-chain
        Pi = np.zeros((self.n, self.n))

        # transient (tr) states (st) preparation
        trStIdxs = [t for tc_sub in self.tc for t in tc_sub]
        nTrSt = len(trStIdxs)  # number of transient states
        ITrSt = np.eye(nTrSt)
        if self.bTransientStates:
            PTrSt = self.P[np.ix_(trStIdxs, trStIdxs)]  # transient part of transition matrix
            ITrSt = np.eye(nTrSt)  # identity matrix of transient size
            ProbTrtoErg = npla.inv(ITrSt - PTrSt)  # erg. prob. from tr. st. to erg. st.

        # fill Pi
        for e in self.ec:

            # ergodic classes
            idxsEc = np.ix_(e, e)
            Pi[idxsEc] = MarkovChain(self.P[idxsEc]).Pi  # note a uni-chain

            if self.bTransientStates:
                # transient states to ergodic classes
                trStToEcIdxs = np.ix_(trStIdxs, e)  # indexes from tr. to erg.
                PTtStToEc = self.P[trStToEcIdxs]  # transient part to ergodic class e
                Pi[trStToEcIdxs] = ProbTrtoErg.dot(PTtStToEc).dot(Pi[idxsEc])

        return Pi
    
    def pi(self):
        ''' 
        Stationary distribution (if it exists). (from Markov_chain_new.py)
        '''

        if self.bUniChain:
            # stationary distribution exists
            Z = self.P - np.eye(self.n)
            Z[:, [0]] = np.ones((self.n, 1))
            # TODO: instead of inverse, solve a system of linear equations
            return la.inv(Z)[[0], :]
        else:
            raise Warning('Stationary distribution does not exist.')
    
    def M(self):
        '''
        Mean first passage matrix. (from Markov_chain_new.py)
        '''
    
        # TODO: M is non-existent for multi-chains, maybe option to calculate per ergodic class?

        if len(self.ec) > 1 or len(self.tc) > 1:
            raise Exception('Mean first passage matrix does not exist for multi-chains.')

        dgMatrixD = np.diag(np.diag(self.D))
        dgMatrixPiInv = la.inv(np.diag(np.diag(self.Pi())))

        return (self.I - self.D + self.ones.dot(dgMatrixD)).dot(dgMatrixPiInv)
    
    def Z(self):
        '''
        Fundamental matrix. (from Markov_chain_new.py)
        '''
        return la.inv(self.I() - self.P + self.Pi())

    def D(self):
        '''
        Deviation matrix. (from Markov_chain_new.py)
        '''
        return self.Z() - self.Pi()

if __name__ == '__main__':
    n=6
    r=3
    c=3
    P = np.ones((n,n))/n
    MC = MarkovChain(P)
    
    for c in range(5):
        mVar = np.zeros((n,n-1))
        mVar[r,c] = 1
        print (MC.P_derivative(2,mVar))
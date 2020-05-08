# -*- coding: utf-8 -*-
"""
Created on Wed May  6 20:50:21 2020

@author: cfran
"""

def toTheta(P):
    '''
    Purpose: 
        Convert the P^0.5 transition matrix using cartesian coordinates to a Theta transition matrix using spherical coordinates
    Input:
        P - a (nxn)-matrix with transition probabilities
    Output:
        Theta - a (nxn)-matrix representing each row of the P matrix in spherical coordinates, corresponding to a row in mSph. 
                i.e., each row in Theta respresents a row in P. (Based on a transition matrix with single angles on diagonal)
    '''
    n, _ = P.shape
    P = np.sqrt(P)
    mTheta = np.zeros( (n,n-1) )
    # All first entries of mTheta first
    for i in range(n):            
        mTheta[i,0] = np.arccos( P[i,i] )

    # Rest of mTheta
    for i in range(n): # Rows
        for j in range(1,n-1): # Columns
            if j<i:
                vProb = np.concatenate( ([P[i,0]], P[i,j:i], P[i,i+1:]) )
                if (np.array_equal(vProb, np.zeros(len(vProb)))):
                    break #All angles remain at zero. Using below formulas will give division by 0
                mTheta[i,j] = np.arccos( P[i,j] / euclidean(vProb, np.zeros(len(vProb))) )
            elif j==i:
                vProb = np.concatenate( ([P[i,0]], P[i,i+1:]) )
                if (np.array_equal(vProb, np.zeros(len(vProb)))):
                    break #All angles remain at zero. Using below formulas will give division by 0
                mTheta[i,j] = np.arccos( P[i,0] / euclidean(vProb, np.zeros(len(vProb))) )
            else: #j>i
                vProb = P[i,j:]
                if (np.array_equal(vProb, np.zeros(len(vProb))) ):
                    break #All angles remain at zero. Using below formulas will give division by 0
                mTheta[i,j] = np.arccos( P[i,j] / euclidean(vProb, np.zeros(len(vProb))) )
    return mTheta
        
def toP(Theta):
    '''
    Purpose: 
        Convert the Theta transition matrix using spherical coordinates to a P transition matrix using cartesian coordinates
    Input:
        Theta - a (nx(n-1))-matrix (numpy array) with angles
    Output:
        P - a (nxn)-matrix representing each row of the Theta matrix in cartesian coordinates, corresponding to a row in P. 
                i.e., each row in P respresents a row in Theta. (Based on a transition matrix with single angles on diagonal)
    '''
    n, _ = Theta.shape 
    P = np.zeros((n, n))
    for i in range(n): # rows
        for j in range(n): # columns
            if j == 0: # all elements for first column of P
                if i == 0:
                    P[0,0] = np.cos(Theta[0,0])
                else: #i >= 0, j == 0
                    if i <= n-2 : # Insert expression with cos
                        P[i,0] = np.prod( [np.cos(Theta[i,i]), np.prod([np.sin(Theta[i,k]) for k in range(i)])] )
                    else:
                        P[i,0] = np.prod( [np.sin(Theta[i,k]) for k in range(i)] )
            else: # j >= 0 (all other columns of P)   
                if j == i: # (here cos(theta) swaps to first entry of matrix P)
                    P[i,j] = np.cos(Theta[i,0])
                else: #i < j, i > j
                    if j <= n-2:
                        P[i,j] = np.prod( [np.cos(Theta[i,j]), np.prod([np.sin(Theta[i,k]) for k in range(j)])] )
                    else:
                        P[i,j] = np.prod( [np.sin(Theta[i,k]) for k in range(j)] )
    return P**2

def derivativeP(Theta, r, c):
    '''
    Purpose:
        Calculate the partial derivative of P w.r.t. Theta(r,c)
    Input:
        Theta - a (nx(n-1))-matrix (numpy array) with angles
        r - indicates the row of the theta element we differentiate to, i.e., for Theta(r,c).
        c - indicates the column of the theta element we differentiate to, i.e., for Theta(r,c).
    Output:
        derivP - (nxn)-matrix (numpy array) of the derivative of the P matrix
        (Based on a transition matrix with single angles on diagonal)
    '''
    P = toP(Theta)
    n,_ = P.shape
    P_der = np.zeros( (n, n) )
    for j in range(n): # columns
        if j==r:
            if c==0: # derivative is zero
                # Insert first element of gradient (swapped with diagonal)
                P_der[r,j] = -2 * np.sin(Theta[r,0]) * np.cos(Theta[r,0])
            else:
                continue
        else:
            # Insert first element of gradient (swapped with diagonal)   
            if j==0:
                if r<c: # derivative is zero
                    continue
                elif r==c: # differentiate cosine
                    P_der[r,0] = np.prod( [-2*np.sin(Theta[r,r])*np.cos(Theta[r,r]), \
                                 np.prod([np.sin(Theta[r,k])**2 for k in range(r)])] )
                else: #r>c  
                    if r < n-1:    # differentiate sine                               
                        P_der[r,0] = np.prod( [np.cos(Theta[r,r])**2, \
                                     np.prod([np.sin(Theta[r,k])**2 if k!=c \
                                         else 2*np.sin(Theta[r,k])*np.cos(Theta[r,k]) for k in range(r)])] )
                    elif r == n-1 :
                        P_der[r,0] = np.prod( [np.sin(Theta[r,k])**2 if k!=c \
                                         else 2*np.sin(Theta[r,k])*np.cos(Theta[r,k]) for k in range(r)] )
            else: #j!=0
                if j<c: # derivative is zero
                    continue
                else: # j>=c
                    if j==n-1: # last column    
                        P_der[r,j] = np.prod( [np.sin(Theta[r,k])**2 if k!=c \
                                         else 2*np.sin(Theta[r,k])*np.cos(Theta[r,k]) for k in range(j)])
                    else:
                        P_der[r,j] = np.prod( [np.cos(Theta[r,j])**2 if j!=c else -2*np.sin(Theta[r,j])*np.cos(Theta[r,j]), \
                                     np.prod( [np.sin(Theta[r,k])**2 if k!=c \
                                         else 2*np.sin(Theta[r,k])*np.cos(Theta[r,k]) for k in range(j)])] )
    return P_der
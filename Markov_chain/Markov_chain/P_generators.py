# -*- coding: utf-8 -*-
"""
Created on 12/22/2018

Author: Joost Berkhout (CWI, email: j.berkhout@cwi.nl)

Description: Contains generators of some standard Markov chain probability
transition matrices.
"""

from __future__ import division
import numpy as np
import tools
import scipy.linalg as la


def cockroach_graph(k):
    """Normalized cockroach adjacency matrix with 4*k nodes.

    Cockroach graph originates from Guattery and Miller (1998). The top row is
    labeled as 1, 2, ..., 2k while the second row is labeled from
    4k, 4k+1, ..., 2k+1.

    Parameters
    ----------
    k : int
      Equal to the number of nodes divided by 4.

    Returns
    -------
    P : np.array
      The transition probability matrix of normalized cockroach adjacency matrix
    """

    # init
    n = 4*k  # number of nodes
    P = np.zeros((n,n))  # transition matrix

    # connect nodes from left to right in top row, then connect 2k with 2k+1
    # and then connect second row from right to left all the way to 4k.
    for i in range(n-1):
        P[i, i+1] = 1
        P[i+1, i] = 1

    # add extra connections between the two rows
    for j in range(k-1):
        P[k+j, 3*k-1-j] = 1
        P[3*k-1-j, k+j] = 1

    return tools.normalize_rows(P)


def random_multi_chain(structure, a):
    """Random multi-chain of a given structure.

    Parameters
    ----------
    structure : list
        Gives the ergodic classes sizes and number of transient states, resp..
    a : float/int
        Inflation number with which the random numbers will be multiplied.

    Returns
    -------
    P : np.array
        The transition probability matrix of a random Markov multi-chain.
    """

    # fill block diagonals
    mA = []
    for i in structure:
        mA.append(a*np.random.rand(i, i))
    mA = la.block_diag(*mA)

    # add transient part
    n = sum(structure)
    nTr = structure[-1]
    strtIdxTr = n - nTr
    mA[strtIdxTr:, :strtIdxTr] = a*np.random.rand(nTr, n-nTr)

    return tools.normalize_rows(mA)


if __name__ == "__main__":
    from Markov_chain_new import MarkovChain

    MC = MarkovChain(cockroach_graph(4))
    # print(cockroach_graph(4))
    # print(random_multi_chain([2,2,1], 10))

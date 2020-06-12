# -*- coding: utf-8 -*-
"""
Created on 12/22/2018

Author: Joost Berkhout (CWI, email: j.berkhout@cwi.nl)

Description: Implementation of the Kemeny Decomposition Algorithm (KDA).
"""

from __future__ import division
import matplotlib.pyplot as plt
from Markov_chain_new import MarkovChain
import copy
import numpy as np
import tools
import Markov_chain as MCOld


class KDA(object):
    """Kemeny decomposition algorithm.

    The notation of the corresponding OR article is used.

    Parameters
    ----------
    MC : MarkovChain
        Original Markov chain object which will be cut.
    CO_A : str
        Condition of how often the Kemeny constant derivatives are being
        recaculated (outer while loop in KDA). The options are:
            'CO_A_1(i)' = Number of times performed < i
            'CO_A_2(E)' = Number of ergodic classes in cutted MC is < E
            'CO_A_3(C)' = Number of strongly connected components in cutted
                          MC is < C
    CO_B : str
        Condition of how many edges are being cut per iteration (inner while
        loop in KDA). The options are:
            'CO_B_1(e)' = Number of edges cut is < e
            'CO_B_2(E)' = Number of ergodic classes in MC is < E
            'CO_B_3(q)' = Not all edges with MC.KDer < q are cut
    SC : bool
        Determines whether the edges will be symmetrically cut, i.e., when True
        if edge (i, j) is cut, then also edge (j, i) will be cut.
    verbose : bool or None
        Decides whether information will be printed or not. If None, it is
        based on MC.verbose.
    cutLog : dict
        Dictionary which logs the cutting of the Markov chain.

    """

    def __init__(self, MC, CO_A='CO_A_1(1)', CO_B='CO_B_3(0)', SC=False,
                 verbose=None):
        self.MC = copy.deepcopy(MC)
        self.CO_A = CO_A
        self.CO_B = CO_B
        self.SC = SC
        self.verbose = MC.verbose if verbose is None else verbose
        self.cutLog = {'MCs': [], 'edgesCut': []}
        self.apply_KDA()

    def condition_A(self):
        """Returns bool whether condition A is True or False. """

        nrInCO_A = int(self.get_nr_in_brackets(self.CO_A))

        if self.CO_A[:6] == 'CO_A_1':
            return self.iterCounter < nrInCO_A
        elif self.CO_A[:6] == 'CO_A_2':
            return self.MC.nEc < nrInCO_A
        elif self.CO_A[:6] == 'CO_A_3':
            return self.MC.nSCC < nrInCO_A
        else:
            raise Exception('Unknown condition A chosen (CO_A).')

    def condition_B(self):
        """Returns bool whether condition B is True or False. """

        if self.CO_B[:6] == 'CO_B_2':
            return len(self.MC.ec) < self.get_nr_in_brackets(self.CO_B)
        else:
            raise Exception('Unknown condition B chosen (CO_B).')

    def apply_KDA(self):
        """Applies KDA with conditions A and B. """

        self.iterCounter = 0  # of the outer while loop of KDA

        while self.condition_A():
            # start a new cutting "iteration"

            if self.CO_B[:6] == 'CO_B_1':

                # find edges to cut
                e = self.get_nr_in_brackets(self.CO_B)
                cutEdgesInIter = self.MC.get_most_connecting_edges(e)

                # cut and log the edges
                self.log_cut_edges(self.cut_edges(cutEdgesInIter))

            elif self.CO_B[:6] == 'CO_B_3':

                # find edges to cut
                q = self.get_nr_in_brackets(self.CO_B)
                cutEdgesInIter = self.MC.get_edges_below_threshold(q)

                # cut and log the edges
                self.log_cut_edges(self.cut_edges(cutEdgesInIter))

            else:

                cutEdgesInIter = (np.array([], dtype=int),
                                  np.array([], dtype=int))

                while self.condition_B():  # inner while loop
                    cutEdges = self.MC.get_most_connecting_edges(1)
                    cuttedEdge = self.cut_edges(cutEdges)
                    cutEdgesInIter = self.add_edges(cutEdgesInIter, cuttedEdge)

                # log the edges cut
                self.log_cut_edges(cutEdgesInIter)

            self.iterCounter += 1

    def cut_edges(self, edges):
        """Cut all given edges in the Markov chain (in place).

        When self.SC = True, also the edges in reversed direction are being cut.

        Parameters
        ----------
        edges : tuple or np.array
            Contains the edges to cut represented as a tuple of arrays or a
            boolean np.array.

        Returns
        -------
        edges : tuple
            The edges that are cut captured as a tuple of arrays.
        """

        if not isinstance(edges, tuple):
            # boolean np.array, transform to tuple
            edges = np.nonzero(edges)

        if self.SC:
            # also cut edges in the other direction
            edges = self.add_reversed_edges(edges)

        self.MC.P[edges] = 0  # cut edges

        self.MC = MarkovChain(self.MC.P, verbose=self.verbose)

        return edges

    def add_reversed_edges(self, edges):
        """Adds reversed edges in the edge tuple of arrays. """
        return self.add_edges(edges, tuple(x for x in edges[::-1]))

    def add_edges(self, edges, edgesToAdd):
        """Add edgesToAdd to edge and return copy. Both should be
        tuple of arrays """
        return tuple(np.append(x, y) for x, y in zip(edges, edgesToAdd))

    def log_cut_edges(self, cutEdges):
        """Log the edges from cutEdges. """

        if not isinstance(cutEdges, tuple):
            cutEdges = np.nonzero(cutEdges)

        self.cutLog['MCs'].append(copy.deepcopy(self.MC))
        self.cutLog['edgesCut'].append(zip(*cutEdges))

    @staticmethod
    def get_nr_in_brackets(s):
        """Gets the number given between brackets in string s.

        Example:
            s = "faksdfkjha(382)jflaksjdfkj"
            get_nr_in_brackets(s) == 382

        """
        return int(s[s.find("(")+1:s.find(")")])

    @staticmethod
    def lexico_order(vOld, vNew):
        """Reorders vNew so that it is in lexico order regarding vOld.

        Purpose: For two strongly connected components lists of two MCs.
        """

        lexicoV = []
        for eOld in vOld:
            for eNew in vNew:
                if set(eNew).issubset(set(eOld)):
                    lexicoV.append(eNew)
        return lexicoV

    @staticmethod
    def lexico_reorder_complete(listOfLists):
        """Lexicographically reorder lists inside given list l.

        Purpose: reorder strongly connected components history list in a
        lexico way.
        """

        lexicoL = [listOfLists[0]]

        for idx, l in enumerate(listOfLists[1:]):
            lexicoL.append(KDA.lexico_order(lexicoL[idx], l))

        return lexicoL

    def plot(self):
        """Plots the progress of the clustering based upon strongly
        connected components. Method under construction for Bernd. """

        # user init (dist = distance)
        distInCluster = 1
        distInterCluster = 2
        extendDistFactor = .05

        # determine the Markov chains with different nSCC to plot
        logMCs = self.cutLog['MCs']
        if len(logMCs) == 0:
            print('\nWarning: No edges were cut by KDA. Please consider the '
                  'original Markov chain and the KDA conditions. Nothing '
                  'will be plotted.\n')
            return
        nSCCsMCs = [MC.nSCC for MC in logMCs]
        _, idxs = np.unique(nSCCsMCs, return_index=True)
        plotMCs = [logMCs[i] for i in idxs]
        SCCLog = [MC.SCC for MC in plotMCs]

        # determine ordering indexes
        heightsIndexes = np.empty(self.MC.n,)
        finalClusters = KDA.lexico_reorder_complete(SCCLog)[-1]
        curHeight = 0
        for c in finalClusters:
            heightsCluster = curHeight + np.arange(0, distInCluster*len(c), distInCluster)
            heightsIndexes[c] = heightsCluster
            curHeight = heightsCluster[-1] + distInterCluster

        # plot clustering result

        # init figure
        fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1)

        # set labels
        ax.set_xlabel('Level of decomposition $\longrightarrow$')
        ax.set_ylabel('Agents')

        # set ticks
        ax.set_xticks(np.arange(plotMCs[-1].nSCC))
        plt.yticks(heightsIndexes, range(self.MC.n))
        # plt.grid(which='major', axis='x')
        ax.tick_params(axis=u'both', which=u'both',length=0)  # remove tick lines

        # set axes limits
        maxY = max(heightsIndexes)
        maxX = len(plotMCs)
        extendDistYAxis = maxY*extendDistFactor
        extendDistXAxis = maxX*extendDistFactor*maxX/maxY  # scale for equal sizes
        xVals = [-extendDistXAxis, len(plotMCs) - 1 + extendDistXAxis]
        yVals = [min(heightsIndexes) - extendDistYAxis,
                 max(heightsIndexes) + extendDistYAxis]
        plt.xlim(xVals)
        plt.ylim(yVals)

        # remove outer box
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # plot horizontal lines
        for h in heightsIndexes:
            ax.plot(xVals, [h, h], color='black')

        # plot vertical cluster lines for all Markov chains
        GP = self.MC.ranking.Google_PageRank_scores[0]
        for idx, MC in enumerate(plotMCs):
            for c in MC.SCC:

                # plot vertical 'cluster' line
                yValsCluster = [min(heightsIndexes[c]) - extendDistYAxis/4,
                                max(heightsIndexes[c]) + extendDistYAxis/4]
                ax.plot([idx, idx], yValsCluster, color='black')

                # plot leader
                idxLeader = c[np.argsort(GP[c])[-1]]
                ax.scatter(idx, heightsIndexes[idxLeader], color='blue',
                           zorder=100)


if __name__ == "__main__":
    MC = MarkovChain('ZacharysKarateClub', verbose=True)

    # KDA1 = KDA(MC)
    #
    KDA2 = KDA(MC, CO_A='CO_A_3(3)', CO_B='CO_B_1(3)', SC=True)
    print (KDA2.MC)

    KDA3 = KDA(MC, CO_A='CO_A_1(1)', CO_B='CO_B_3(0)', SC=True)
    print (KDA3.MC)

    KDA3a = KDA(MC, CO_A='CO_A_2(2)', CO_B='CO_B_1(1)', SC=True)
    print (KDA3a.MC)

    KDA3b = KDA(MC, CO_A='CO_A_2(3)', CO_B='CO_B_1(1)', SC=True)
    print (KDA3b.MC)

    MCCourtois = MarkovChain('Courtois', verbose=True)
    KDA4 = KDA(MCCourtois, CO_A='CO_A_2(3)', CO_B='CO_B_1(1)')
    print (KDA4.MC)
    print (KDA4.cutLog['edgesCut'])

    KDA5 = KDA(MCCourtois, CO_A='CO_A_1(1)', CO_B='CO_B_3(0)')
    print (KDA5.MC)
    print (KDA5.MC.P - KDA4.MC.P)

    KDA6 = KDA(MCCourtois, CO_A='CO_A_1(2)', CO_B='CO_B_1(2)')
    KDA7 = KDA(MCCourtois, CO_A='CO_A_3(3)', CO_B='CO_B_1(1)')
    KDA8 = KDA(MC, CO_A='CO_A_3(5)', CO_B='CO_B_1(1)')
    # KDA.MC.draw()

    KDAORArticleFig4 = KDA(MC, CO_A='CO_A_1(1)', CO_B='CO_B_3(0)', SC=False)
    print ('Figure 4 of OR article correponds to:')
    print (KDAORArticleFig4.MC)
    print (KDAORArticleFig4.MC.ec + KDAORArticleFig4.MC.tc)
	
	
    # Test with old implementation
    import scipy.linalg as la

    MC = MarkovChain('Courtois')
    P = np.array(MC.P)
    KDerOld = MCOld.calc_Kemeny_derivatives_exact(P)
    PCutOld = MCOld.Kemeny_cutting(P,
                                   KDerOld,
                                   np.sum((KDerOld*(P>0)) < 0),
                                   False)
    KDA1 = KDA(MC, CO_A='CO_A_1(1)', CO_B='CO_B_3(0)', SC=False, verbose=False)
    print (la.norm(PCutOld - KDA1.MC.P))

    PCutOld = MCOld.Kemeny_cutting(P,
                                   KDerOld,
                                   np.sum((KDerOld*(P>0)) < 0),
                                   True)
    KDA1 = KDA(MC, CO_A='CO_A_1(1)', CO_B='CO_B_3(0)', SC=True, verbose=False)
    print (la.norm(PCutOld - KDA1.MC.P))

    N = 15  # nr. of edges to cut
    PCutOld = MCOld.Kemeny_cutting(P,
                                   KDerOld,
                                   N,
                                   False)
    KDA1 = KDA(MC, CO_A='CO_A_1(1)', CO_B='CO_B_1({})'.format(N), SC=False, verbose=False)
    print (la.norm(PCutOld - KDA1.MC.P))

    KDA7.plot()
    KDA8.plot()

    print('LOTR example')
    MC = MarkovChain('lotr_test2')
    KDA_MC = KDA(MC, CO_A='CO_A_3(3)', CO_B='CO_B_1(1)', SC=False)
    KDA_MC.plot()
    print(KDA_MC.cutLog['edgesCut'])


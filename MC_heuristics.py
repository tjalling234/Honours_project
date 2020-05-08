# -*- coding: utf-8 -*-
"""
Created on 06-05-2020

Author: Pieke Geraedts

Description: 
    This py file contains heuristics for epsilon. These heuristics work on the basis of MarkovChain input and can be called from the other files with the correct input.
    At first these heuristics will work for the setting of changing a single theta element.
    The heuristics:
        -Momentum: Take into accoount information on the derivative of the objective in previous iterations. 
            Using only the direction of the derivative.
        -ImprovedMomentum: Take into accoount information on the derivative of the objective in previous iterations. 
            Using both the direction and the magnitude of the derivative.
        -Hessian: Calculate the hessian and use this to update epsilon.
        
"""
#TODO: make the methods for these 3 heuristics.

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:13:12 2019

@author: ryans
"""

import networkx as nx
import numpy as np
from scipy import sparse

def similarity(Graph,i = 2,t = 0.1):
    
    """
    Inputs:
    G: A networkx undirected unweighted Graph
    i: A chosen vertex
    t: Free parameter
    
    Outputs: ordered list of vertices by similarity score.
    """
    
    #Set up variables
    N = nx.number_of_nodes(Graph)  
    A = nx.to_numpy_matrix(Graph)
    L = nx.laplacian_matrix(Graph)
    D = L + A
    I = np.identity(N)
    ones = np.zeros(N) + 1
    C = np.block([[A,I-D],[I,np.zeros([N,N])]])
    lam = np.amax(np.linalg.eigvals(C))
    
    #Generate Warning near pole:
    if t > (1/lam)*0.9:
        print('WARNING t = %f is close or larger than 1/lambda = %f' %(t, 1/lam))
    
    #Calculate generating function
    M = I - (A * t) + (D - I)*(t**2)
    M_inv = np.linalg.inv(M)
    NBRW = (1 - t**2)*M_inv
    
    #Calculate similarities for all vertices which are not neighbours of i
    similarities = {}
    for j in Graph:
        if j not in Graph.neighbors(i) and j!=i:
            s_ij = NBRW[i,j] + NBRW[j,i]
            similarities[(i,j)] = s_ij
    
    similarities_sorted_by_value = sorted(similarities.items(), key=lambda k: k[1])
    
    
    #Convert into ordered list of vertices:
    sorted_vertices = []
    for i in similarities_sorted_by_value:
        sorted_vertices.insert(0,i[0])
        
    return similarities
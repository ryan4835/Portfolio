# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:27:25 2019

@author: ryans
"""
import networkx as nx
import numpy as np
from scipy import sparse
from NBTRsimilarityforedgepredictsparse import similarity_for_predict_sparse



def edge_predictor_sparse(Graph, t=0.1):
    
    """Inputs: 
    G: Network X graph
    t: free paramater
    
    Outputs: Dictionary of possible new edges with similarities
    """
    Scores = {}
    for i in Graph:
        update = similarity_for_predict_sparse(Graph,i,t)
        Scores.update(update)
    
    return Scores
        
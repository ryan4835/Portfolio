# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:27:25 2019

@author: ryans
"""
import networkx as nx
import numpy as np
from scipy import sparse
from NBTRsimilarityforedgepredict import similarity_for_predict



def edge_predictor(Graph, t=0.1):
    
    """Inputs: 
    G: Network X graph
    t: scalar free paramater
    
    Outputs: Dictionary of possible new edges with similarities
    """
    Scores = {}
    
    for i in Graph:
        update = similarity_for_predict(Graph,i,t)
        Scores.update(update)
    
    return Scores
        
# -*- coding: utf-8 -*-
"""
Creating a class of basic Graph Convolutional Network (GCN) models, with methods to compute the underlying spectral gap and mixing times. 
"""
import networkx as nx
import numpy as np
from scipy import linalg
import pandas as pd

class GCN:
    def __init__(self, G, c1,c2):
        """
        G: a NetworkX graph with n nodes
        c1: list of nodes labelling a cluster
        c2: list labelling a different cluster.
        
        #Assign
        cmap: Colour map for the two clusters. 
        A: n x n adjacency matrix
        L: n x n graph laplacian
        D: m x n diagonal degree matrix
        signal: an intial randomised signal for the graph, with different locations dependent on cluster. 
        
        """
        self.c1 = c1
        self.c2 = c2
        self.SB = G
        self.cmap = cmap = np.array([1 if i in c1 else 0 for i in list(self.SB)])
        self.size = int(self.SB.number_of_nodes())
        self.A = nx.to_numpy_matrix(self.SB)
        self.L = nx.laplacian_matrix(self.SB)
        self.D = self.L + self.A 
        self.Atilde = self.A + np.identity(self.SB.number_of_nodes())
        self.Dtilde = self.D + np.identity(self.SB.number_of_nodes())
        self.Dtilde_inv_sqrt = np.diag(np.diag(self.Dtilde)**(-1/2))
        self.new_laplacian = self.Dtilde_inv_sqrt*self.A*self.Dtilde_inv_sqrt
        self.signal = np.array([[np.random.normal(loc= -0.2),np.random.normal(loc=-0.25)]
                                if i in self.c1
                                else [np.random.normal(loc=0.25),np.random.normal(loc=0.25)]
                                for i in list(self.SB)])
        
        
    def vector_average(self,H):
        total = np.array([0,0]).astype(float)
        for row in H:
            total += row
        
        return total/H.shape[0]    
    
    
    
    def SpectralGap(self):
        
        sparse_L = nx.normalized_laplacian_matrix(self.SB)
        numpy_L = sparse_L.todense()
        eigenvals = sorted(np.linalg.eig(numpy_L)[0])
        spec_gap = eigenvals[1] - eigenvals[0]
        return spec_gap
    
    
    def ForwardPass(self, H):
        H_new = np.matmul(self.new_laplacian, H)
        
        return H_new
    
    
    def Mixing_time(self):
        n = 0
        H = self.signal
        while np.linalg.norm(self.vector_average(H[self.cmap == 0, :])
                             - self.vector_average(H[self.cmap == 1, :])) > self.size/2000 and (n < 50):
           H = np.array(self.ForwardPass(H))
           n +=1
          
        return n
    
    
    
    def Mixing_time_average(self, n_iter = 5):
        total = 0
        for i in range(n_iter):
            total += self.Mixing_time()
            
        return total/n_iter

    
    def Mixing_time_draw(self):
        
        picture = [0,5,10,15,20,25]
        data = {}
        n = 0
        k = 0
        H = self.signal
        
        while (n < 50):
            H = np.array(self.ForwardPass(H))
            if n in picture:
                data[2*k] = np.ravel(H[:,0])
                data[2*k + 1] = np.ravel(H[:,1])
                k+=1
            n +=1
            
        return data
    
    

#!/usr/bin/env python
# coding: utf-8




import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from NBTRsimilarity import similarity




from sklearn import metrics
from sklearn import model_selection
from AUC import AUC_score
from edge_predictor_sparse import edge_predictor_sparse
from katz_edge_predictor_sparse import katz_edge_predictor_sparse
from Katzindexsim import katz_similarity
import collections




#Parse Spanish Train Graph Data
train_path = 'C:\\Users\\ryans\Documents/HT/Networks/Datasets/dimacs10-netscience/out.dimacs10-netscience'

train_file = open(train_path)
train_lines = train_file.readlines()
train_edges = []
for i in train_lines:
    train_edges.append([int(s) for s in i.split() if s.isdigit()][:2])

train_edges = train_edges[1:]
train_file.close()
#print(train_edges)




G = nx.Graph(train_edges)

nx.draw(G, with_labels = False, node_color = 'g', node_size = 10)

print(G.number_of_nodes())
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.6, color='b')


plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d+0.4  for d in deg])
ax.set_xticklabels(deg)
ax.axes.get_xaxis().set_ticks([])




def split_to_test(E, index):
    E_P = []
    for i in index:
        E_P.append(E[i])
    return E_P



n_iter = 5
t_size = 0.1
att = 0.04
G = nx.Graph(train_edges)
G = nx.convert_node_labels_to_integers(G)
E = []
G_complete = nx.complete_graph(G.number_of_nodes())
E_C = [] 
for i in G_complete.edges:
    if i not in G.edges:
        E_C.append(i)
for i in G.edges:
    E.append(i)
five_folds = model_selection.KFold(n_splits=n_iter,shuffle=True, random_state=0)




AUC_sum = 0
katz_sum = 0
ra_sum = 0
ja_sum = 0 
pa_sum = 0 
aa_sum = 0
for i in five_folds.split(E):
    G = nx.Graph(train_edges)
    G = nx.convert_node_labels_to_integers(G)
    E_P = split_to_test(E, i[1])
    G.remove_edges_from(E_P)

    d = edge_predictor_sparse(G, t = att)
    d_k = katz_edge_predictor_sparse(G, t = att)
    ra_iter = nx.resource_allocation_index(G)
    d_ra = {}
    ja_iter = nx.jaccard_coefficient(G)
    d_ja = {}
    pa_iter = nx.preferential_attachment(G)
    d_pa = {}
    aa_iter = nx.adamic_adar_index(G)
    d_aa = {}
    for i in ra_iter:
        d_ra[i[:2]] = i[-1]
    for i in ja_iter:
        d_ja[i[:2]] = i[-1]
    for i in pa_iter:
        d_pa[i[:2]] = i[-1]
    for i in aa_iter:
        d_aa[i[:2]] = i[-1]
    
    katz_sum += AUC_score(d_k, E_C, E_P)
    AUC_sum += AUC_score(d, E_C, E_P)
    ra_sum += AUC_score(d_ra, E_C, E_P)
    ja_sum += AUC_score(d_ja, E_C, E_P)
    pa_sum += AUC_score(d_pa, E_C, E_P) 
    aa_sum += AUC_score(d_aa, E_C, E_P) 

print('Katz', katz_sum/n_iter)    
print('NB',AUC_sum/n_iter)
print('RA', ra_sum/n_iter)
print('JA', ja_sum/n_iter)
print('PA', pa_sum/n_iter)
print('AA', aa_sum/n_iter)
    
    
    

from scipy import sparse
G = nx.Graph(train_edges)

Graph = nx.convert_node_labels_to_integers(G)

N = nx.number_of_nodes(Graph)  
A = nx.to_numpy_matrix(Graph)
L = nx.laplacian_matrix(Graph)
D = L + A
I = np.identity(N)
ones = np.zeros(N) + 1
C = np.block([[A,I-D],[I,np.zeros([N,N])]])
lam = np.amax(np.linalg.eigvals(C))
print(1/lam)





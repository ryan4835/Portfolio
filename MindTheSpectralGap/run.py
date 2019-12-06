import numpy as np 
import networkx as nx
from GCN import GCN
from GCN1 import GCN1
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



kc_path = 'C:\\Users\\ryans\Documents/HT/Networks/Datasets/ucidata-zachary/out.ucidata-zachary'
kc_file = open(kc_path)
kc_lines = kc_file.readlines()
kc_edges = []

for i in kc_lines:
    kc_edges.append([int(s) for s in i.split() if s.isdigit()][:])




kc_edges = kc_edges[2:]
kc_file.close()



G = nx.Graph(kc_edges)
test = nx.algorithms.community.centrality.girvan_newman(G)
bipartition = tuple(sorted(c) for c in next(test))
component_1 = bipartition[0]
component_2 = bipartition[1]
cmap = [1 if i in component_1 else 0 for i in list(G)]
draw_graph = nx.draw(G, node_color = cmap)



network = GCN(G = G, c1 = component_1, c2 = component_2)
network.Mixing_time_average(n_iter = 100)
draw_data = network.Mixing_time_draw()



draw_dataframe = pd.DataFrame(draw_data)
draw_dataframe['Community'] = cmap
print(draw_dataframe.head())



fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(2, 2, 1)
sns.scatterplot(draw_dataframe[0],
                draw_dataframe[1], 
                hue = draw_dataframe['Community'],
                ax=ax, 
                palette='dark',
                legend = False, 
                alpha=0.8)

ax.set_ylabel('')
ax.set_xlabel('Initial signals')
ax.set_xticklabels([])
ax.set_yticklabels([])

ax = fig.add_subplot(2, 2, 2)
sns.scatterplot(draw_dataframe[2],
                draw_dataframe[3], 
                hue = draw_dataframe['Community'],
                ax=ax, 
                palette='dark',
                legend = False, 
                alpha=0.8)

ax.set_ylabel('')
ax.set_xlabel('5 layers')
ax.set_xticklabels([])
ax.set_yticklabels([])

ax = fig.add_subplot(2, 2, 3)
sns.scatterplot(draw_dataframe[4],
                draw_dataframe[5],
                hue = draw_dataframe['Community'],
                ax=ax, palette='dark',
                legend = False, 
                alpha=0.8)

ax.set_ylabel('')
ax.set_xlabel('10 layers')
ax.set_xticklabels([])
ax.set_yticklabels([])

ax = fig.add_subplot(2, 2, 4)
sns.scatterplot(draw_dataframe[6],
                draw_dataframe[7], 
                hue = draw_dataframe['Community'],
                ax=ax, palette='dark',
                legend = False, 
                alpha=0.8)

ax.set_ylabel('')
ax.set_xlabel('15 layers')
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.savefig('KCmixingsignals.png')
plt.show()




q = np.arange(0,1,0.01)
x = []
y = []

for q0 in q:
    test = GCN1(p=1, q = q0)
    x.append(test.SpectralGap())
    y.append(test.Mixing_time_average(n_iter = 100))
    
q = np.arange(0,1,0.01)
x2 = []
y2 = []

for q0 in q:
    test = GCN1(p=0.5, q = q0)
    x2.append(test.SpectralGap())
    y2.append(test.Mixing_time_average(n_iter = 100))
    
q = np.arange(0,1,0.01)
x3 = []
y3 = []

for q0 in q:
    test = GCN1(size = 17, p=0.8, q = q0)
    x3.append(test.SpectralGap())
    y3.append(test.Mixing_time_average(n_iter = 100))
    
plt.plot(x, y, x2, y2, x3, y3)
plt.scatter(0.13227232922951662, 16, s = 100)

#karate club
plt.scatter(0.1217678493851207, 26)

#highschool
plt.scatter(0.039580065508459085, 19)

#spanish train
plt.scatter(0.06280067896164994, 25)

#infectious
plt.scatter(0.017955918982715816, 44)

plt.xlabel('Spectral Gap')
plt.ylabel('Mixing Time')
plt.legend(['Paramaters1','Parameters2','Parameters3', 'Karate Club',
            'Highschool', 'Madrid Train', 'Infectious'])
    
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize


r_c_arr = np.array([2.5])

gap = 25
T_min = 100
T_max = 900+gap
sep = np.floor((T_max - T_min) / gap).astype(np.int16)
T = np.zeros(sep, dtype = "float64")
T[0] = T_min
T[1] = T_min + gap
for i in range(2, sep):
    T[i] = T[i-1] + gap
T = T.astype(np.float32)
t_arr = T

gap = 0.01
T_min = 0.51
T_max = 1+gap
sep = np.floor((T_max - T_min) / gap).astype(np.int16)
T = np.zeros(sep, dtype = "float64")
T[0] = T_min
T[1] = T_min + gap
for i in range(2, sep):
    T[i] = T[i-1] + gap
T = T.astype(np.float32)
f_c_arr = T

def hindex(g, n):
    sorted_neighbor_degrees = sorted((g.degree(v) for v in g.neighbors(n)), reverse=True)
    h = 0
    for i in range(1, len(sorted_neighbor_degrees)+1):
        if sorted_neighbor_degrees[i-1] < i:
            break
        h = i
    return h

P = np.array([0])

for p in P:
    for r_c in r_c_arr:
        for f_c in f_c_arr:
            for t in t_arr:
                M = np.load("./graph/D_" + str(int(p)) + "/T_" + str(int(t)) + "_r_" + str(r_c) + "_f_" + str(f_c) + ".npy", allow_pickle=True)
                G = nx.from_numpy_array(M)

                # Degree Centrality
                dc = np.zeros((len(G)), dtype = "float32")
                degree_centrality = nx.degree_centrality(G)
                for i in range(len(G)):
                    dc[i] = degree_centrality[i]

                # H-index Centrality
                h_index = np.zeros((len(G)), dtype = "float32")
                for i in range(len(G)):
                    h_index[i] = hindex(G, i)

                # Closeness Centrality
                cc = np.zeros((len(G)), dtype = "float32")
                closeness_centrality = nx.closeness_centrality(G)
                for i in range(len(G)):
                    cc[i] = closeness_centrality[i]
                    
                # Betweenness Centrality
                bc = np.zeros((len(G)), dtype = "float32")
                betweenness_centrality = nx.betweenness_centrality(G)
                for i in range(len(G)):
                    bc[i] = betweenness_centrality[i]

                # Eigenvector Centrality
                ec = np.zeros((len(G)), dtype = "float32")
                eigenvector_centrality = nx.eigenvector_centrality(G, max_iter = 10000, tol = 1.0e-5)
                for i in range(len(G)):
                    ec[i] = eigenvector_centrality[i]

                # K-shell Centrality
                kn = np.zeros((len(G)), dtype = "float32")
                core_number = nx.core_number(G)
                for i in range(len(G)):
                    kn[i] = core_number[i]

                # Clustering Coeffcient
                clc = np.zeros((len(G)), dtype = "float32")
                clustering = nx.clustering(G)
                for i in range(len(G)):
                    clc[i] = clustering[i]
                
                # Subgraph Centrality
                sc= np.zeros((len(G)), dtype = "float32")
                subgraph_centrality = nx.subgraph_centrality(G)
                for i in range(len(G)):
                    sc[i] = subgraph_centrality[i]
                
                # Load Centrality
                lc = np.zeros((len(G)), dtype = "float32")
                load_centrality = nx.load_centrality(G)
                for i in range(len(G)):
                    lc[i] = load_centrality[i]

                # Harmonic Centrality
                hc = np.zeros((len(G)), dtype = "float32")
                harmonic_centrality = nx.harmonic_centrality(G)
                for i in range(len(G)):
                    hc[i] = harmonic_centrality[i]
                
                X_data = np.column_stack((dc, h_index, cc, bc, ec, kn, clc, sc, lc, hc))
                X_data = normalize(X_data, axis=0, norm='max')
                
                np.save("data/D_" + str(int(p)) + "/X_" + str(int(t)) + "_r_" + str(r_c) + "_f_" + str(f_c) + ".npy", X_data, allow_pickle=True)

                print(str(p), "_", str(f_c), "_" , str(int(t)))

                
                

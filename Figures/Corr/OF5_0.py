#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
import MDAnalysis as md
import time
from sklearn.preprocessing import normalize

def get_neighbors(g, node, depth=1):
    output = {}
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1,depth+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x,[]))
        nodes = output[i]
    return output

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
fs = T

r_c = 2.5
ts = np.array([100,200,300,400,500,600,700,800,900])
fts = 5
p = 0

for t in ts:
    for f_c in fs:
        M = np.load("./graph/D_" + str(int(p)) + "/T_" + str(int(t)) + "_r_" + str(r_c) + "_f_" + str(f_c) + ".npy", allow_pickle=True)
        G = nx.from_numpy_array(M)
        D = np.load("distance_table/D_" + str(int(p)) + "/D_" + str(int(t)) + ".npy")

        source = "traj/T_" + str(int(p)) + "/"
        traj_file = source + "traj.dcd"
        u = md.lib.formats.libdcd.DCDFile(traj_file)
        N_frame = u.n_frames
        n_atom = u.header['natoms']
        traj = u.readframes()[0]
        conf = traj[t]

        cost = np.zeros((864, fts))

        for ft in range(fts):
            for i in range(864):
                lst = get_neighbors(G,i,fts)
                node = lst[ft+1]
                r_i = conf[i]
                s = 0
                for j in node:
                    r_j = conf[j]
                    r_ij = r_j - r_i
                    R_ij = D[i,j]
                    for k in node:
                        if (i != j) and (j != k) and (i != k):
                            r_k = conf[k]
                            r_ik = r_k - r_i
                            R_ik = D[i,k]
                            dot_ij_ik = r_ij[0]*r_ik[0]+r_ij[1]*r_ik[1]+r_ij[2]*r_ik[2]
                            cos_theta = dot_ij_ik / (R_ij * R_ik)
                            s_ = (1 + cos_theta) * np.exp(-(ft+1))
                            s = s + s_
                cost[i,ft] = s   
        np.save("ang_origin/D_" + str(int(p)) + "/F5_" + str(int(t)) + "_r_" + str(r_c) + "_f_" + str(f_c) + ".npy", cost, allow_pickle=True)
        
        print(p,"_",t,"_",f_c)


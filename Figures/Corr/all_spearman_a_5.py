#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

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
f_c = T

cr0 = np.zeros((sep), dtype = "float32")
cr1 = np.zeros((sep), dtype = "float32")
cr2 = np.zeros((sep), dtype = "float32")
cr3 = np.zeros((sep), dtype = "float32")
cr4 = np.zeros((sep), dtype = "float32")

num_set = np.array([0,1,2,3,4,5,6,7,8,9])
time_set = np.array([100,200,300,400,500,600,700,800,900])

counter = 0
for f in f_c:
    x_0 = np.zeros((0), dtype = "float32")
    x_1 = np.zeros((0), dtype = "float32")
    x_2 = np.zeros((0), dtype = "float32")
    x_3 = np.zeros((0), dtype = "float32")
    x_4 = np.zeros((0), dtype = "float32")
    y_ = np.zeros((0), dtype = "float32")
    
    for time in time_set:    
        for num in num_set:
            file = "ang_origin/D_" + str(int(num)) + "/F5_" + str(int(time)) + "_r_2.5_f_" + str(f_c[counter]) + ".npy"
            p_hop = np.load("data/D_" + str(int(num)) + "/P_hop_" + str(int(time)) + ".npy", allow_pickle=True)
            X = np.load(file, allow_pickle=True)
            sh = np.load("data/D_" + str(int(num)) + "/Y_" + str(int(time)) + ".npy", allow_pickle=True)
            x0 = X[:, 0]
            x1 = X[:, 1]
            x2 = X[:, 2]
            x3 = X[:, 3]
            x4 = X[:, 4]
            y = p_hop
            x_0 = np.concatenate((x_0, x0), axis=0)
            x_1 = np.concatenate((x_1, x1), axis=0)
            x_2 = np.concatenate((x_2, x2), axis=0)
            x_3 = np.concatenate((x_3, x3), axis=0)
            x_4 = np.concatenate((x_4, x4), axis=0)
            y_ = np.concatenate((y_, y), axis=0)
            
    corr_0, p_value_0 = spearmanr(x_0, y_)
    corr_1, p_value_1 = spearmanr(x_1, y_)
    corr_2, p_value_2 = spearmanr(x_2, y_)
    corr_3, p_value_3 = spearmanr(x_3, y_)
    corr_4, p_value_4 = spearmanr(x_4, y_)
    
    cr0[counter] = corr_0
    cr1[counter] = corr_1
    cr2[counter] = corr_2
    cr3[counter] = corr_3
    cr4[counter] = corr_4
    
    print(f_c[counter])
    counter = counter + 1

plt.figure(figsize=(8,6))

plt.plot(f_c,cr0,label="1-order neighbor")
plt.plot(f_c,cr1,label="2-order neighbor")
plt.plot(f_c,cr2,label="3-order neighbor")
plt.plot(f_c,cr3,label="4-order neighbor")
plt.plot(f_c,cr4,label="5-order neighbor")

plt.xlabel("Value of $\mathcal{A} \in (0.5,1]$")
plt.ylabel("Spearman's coefficient between $p_{hop,i}(t)$ and angle function on $\mathcal{G}$")
plt.legend()

plt.savefig("/home/chem/msrgxt/Desktop/a.png", dpi = 300)

plt.show()



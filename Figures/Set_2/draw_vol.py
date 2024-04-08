#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import linecache as lc

N_frame = 1001;

N = np.zeros((N_frame), dtype = "int")
for i in range(N_frame):
    N[i] = i + 1
    
num = 10

parameter = np.zeros((num), dtype = "object")
f = np.zeros((num), dtype = "object")

# plt.figure(figsize=[10,8])

plt.figure(figsize=[5,10])

for i in range(num):
    name = "T_" + str(i) + "/nuc.log"
    v_max_n = np.zeros((N_frame), dtype="float32")
    for j in range(N_frame):
        txt = str.split(lc.getline(name, 311 + j))
        num_str = txt[8];
        if num_str == "-1e+20":
            v_max_n[j] = 1
        else:
            v_max_n[j] = float(num_str)
    parameter[i] = v_max_n
    f[i] ,= plt.plot(N, parameter[i])


plt.xlabel("Time $100*(0.2*(m*\sigma^2/\epsilon)^{1/2})$")
plt.ylabel("Volume")

plt.xlim([100,900])

plt.legend(handles=[f[6],f[7],f[8],f[9]], labels = ["Traj$(T^{(2)}_{7})$","Traj$(T^{(2)}_{8})$","Traj$(T^{(2)}_{9})$","Traj$(T^{(2)}_{10})$"], loc="lower left", ncol=2)

plt.title("Volume")
plt.savefig("/home/chem/msrgxt/Desktop/b_2.png", dpi = 300)
plt.show()

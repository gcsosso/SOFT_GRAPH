#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

set_ = 2
select = 5
num = 10

parameter = np.zeros((num), dtype = "object")
f = np.zeros((num), dtype = "object")


gap = 10
T_min = 100
T_max = 900+gap
sep = np.floor((T_max - T_min) / gap).astype(np.int16)
T = np.zeros(sep, dtype = "float64")
T[0] = T_min
T[1] = T_min + gap
for i in range(2, sep):
    T[i] = T[i-1] + gap
T = T.astype(np.float32)
x = T

# plt.figure(figsize=[10,7.5])

plt.figure(figsize=[5.2,8])

for i in range(num):
    file = "softness_" + str(int(set_)) + "/ans_" + str(int(select)) + "_t_" + str(int(i)) + ".csv"
    y = pd.read_csv(file, header=None)
    y = np.transpose(y.values)
        
    f[i] ,= plt.plot(x,y)

plt.axhline(y=0, color='r', linestyle='--')

# plt.legend(handles=[f[0],f[1],f[2],f[3],f[4],f[5],f[6],f[7],f[8],f[9]], labels = ["Traj$(T^{(2)}_{1})$","Traj$(T^{(2)}_{2})$","Traj$(T^{(2)}_{3})$","Traj$(T^{(2)}_{4})$","Traj$(T^{(2)}_{5})$","Traj$(T^{(2)}_{6})$","Traj$(T^{(2)}_{7})$","Traj$(T^{(2)}_{8})$","Traj$(T^{(2)}_{9})$","Traj$(T^{(2)}_{10})$"], loc="best", ncol=3)
plt.title("Average Softness")
plt.xlabel("Time $100*(0.2*(m*\sigma^2/\epsilon)^{1/2})$")
plt.ylabel("Average Softness, $\mathfrak{S}(t)$")
plt.savefig("/home/chem/msrgxt/Desktop/softness.png", dpi = 300)
plt.show()


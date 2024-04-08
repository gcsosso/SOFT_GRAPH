#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([100,200,300,400,500,600,700,800,900])

y_ = np.zeros((9), dtype = "int")
y = pd.read_csv("set_2/ans_5_t_5.csv", header=None)
y = np.transpose(y.values)

for i in range(1,len(y_)):
    y_[i] = y[i * 10]

plt.figure(figsize=[12,10])
plt.plot(x,y_)




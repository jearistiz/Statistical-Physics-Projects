# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from time import time



def ising_microstates(L=2):
    N = L * L
    config = np.array([[0] * N] * 2**N)
    for i in range(N):
        index_factor = 2**N  / 2**(i+1)
        for j in range(2**i):
            config[j*index_factor:(j+1)*index_factor][i] = (-1)**j
    return config

print(ising_microstates())
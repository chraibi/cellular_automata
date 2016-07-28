#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Name:       flow_eval.py
# Created:    26.07.16
__author__ = 'Carlos Esparza'
#-------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

class Config:
    # settings ------------------------------------------------------------------
    ks = 10
    kd = 50
    shuffle = False
    reverse = False
    decay = 0.2
    diffusion = 0.2
    width = 8.0
    height = 8.0

    clean = False
    plotP = False
    plotS = False
    plotD = False
    plotAvgD = False

    def __init__(self, npeds,nruns = 1):
        self.numPeds = npeds
        self.nruns = nruns


import cellular_automaton as ca


nruns = 2

ca.box = [1, 20, 1, 12]

max_peds = 160
step = 10

X = np.reshape([[x] * nruns for x in range(step, max_peds + 1, step)],
               max_peds//step * nruns)

Y = np.zeros(max_peds//step * nruns)

for i in range(max_peds//step):
    conf = Config(step * (i+1), nruns)
    tt = ca.main(conf)
    Y[i*nruns : (i+1)*nruns] = tt
    print(i)

    if conf.plotD or conf.plotAvgD:
        os.system('rm -rf dff-{}'.format(i))
        os.system('mv dff dff-{}'.format(i))

    if conf.plotP:
        os.system('rm -rf peds-{}'.format(i))
        os.system('mv peds peds-{}'.format(i))


plt.scatter(X, Y)
A = np.vstack([X, np.ones(len(X))]).T
m, t = np.linalg.lstsq(A, Y)[0]
plt.plot(X, m*X + t)
plt.xlabel("npeds")
plt.ylabel("t / s")
plt.title('t ~ {:.2f}s * N + {:.2f}s'.format(m, t))

plt.savefig('evac_times.png')


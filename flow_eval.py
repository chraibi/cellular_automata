#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Name:       flow_eval.py
# Created:    26.07.16
__author__ = 'Carlos Esparza'
#-------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import cellular_automaton as ca
import pickle

class Config:
    # settings ------------------------------------------------------------------
    ks = 5
    kd = 0
    shuffle = False
    reverse = False
    decay = 0.3 # delta
    diffusion = 0.1 # alpha
    width = 8
    height = 8
    numPeds = 10

    clean = False
    plotP = False
    plotS = False
    plotD = False
    plotAvgD = True

    def __init__(self, nruns = 1, **kwargs):
        self.nruns = nruns
        self.__dict__.update(kwargs)


def time_var(var, min, max, step, nruns):
    Y = np.zeros((max - min)//step * nruns)

    for i, x in enumerate(range(min, max, step)):
        conf = Config(nruns, **{var: x})
        tt = ca.main(conf)
        Y[i * nruns: (i + 1) * nruns] = tt

        if conf.plotD or conf.plotAvgD:
            os.system('rm -rf dff-{}'.format(i))
            os.system('mv dff dff-{}'.format(i))

        if conf.plotP:
            os.system('rm -rf peds-{}'.format(i))
            os.system('mv peds peds-{}'.format(i))

    X = np.reshape([[x] * nruns for x in range(i + 1)],
                   (i + 1) * nruns)

    plt.scatter(X, Y)
    A = np.vstack([X, np.ones(len(X))]).T
    m, t = np.linalg.lstsq(A, Y)[0]
    plt.plot(X, m * X + t)
    plt.xlabel(var)
    plt.ylabel("t / s")
    plt.title('t ~ {:.2f}s * N + {:.2f}s'.format(m, t))

    conf_str = '-{}.{}x{}'.format(max, step, nruns) + \
               '-{0.width}x{0.height}-ks{0.ks}-kd{0.kd}'.format(conf) + \
               '-a{0.decay}-d{0.diffusion}'.format(conf) * bool(conf.kd)

    plt.savefig('evac_times{}.png'.format(conf_str))

    with open('data-{}.p'.format(conf_str), 'wb') as f:
        pickle.dump((X, Y), f)


ca.box = [1, 20, 1, 12]

var = 'ks'

min = 2
max = 30
step = 2

nruns = 4

time_var(var, min, max, step, nruns)


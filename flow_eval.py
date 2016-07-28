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
from multiprocessing import Pool


CONF_STR = '_{width}x{height}_N{numPeds}_ks{ks}_kd{kd}'
CONF_STR_OPT = '_d{decay}_a{diffusion}'

PRETTY_NAMES = {'ks': '\kappa_S',
                'kd': '\kappa_D',
                'decay': '\delta',
                'diffusion': '\alpha'}


def pretty(var):
    if var in PRETTY_NAMES:
        return PRETTY_NAMES[var]
    else:
        return var

class Config:
    # settings ------------------------------------------------------------------
    ks = 2
    kd = 0
    shuffle = False
    reverse = False
    decay = 0.3 # delta
    diffusion = 0.1 # alpha
    width = 25.2
    height = 25.2
    numPeds = 1116

    clean = False
    plotP = False
    plotS = False
    plotD = False
    plotAvgD = False

    def __init__(self, nruns = 1, **kwargs):
        self.nruns = nruns
        self.__dict__.update(kwargs)


def time_var(var, values, nruns):
    Y = np.zeros(len(values) * nruns)

    for i, x in enumerate(values):
        nproc = min(nruns, 8)

        #njobs = [nruns // nproc] * nproc
        #for i in range(nruns % nproc):
        #    njobs[i] += 1

        confs = [Config(nruns = 1, **{var: x}) for _ in range(nruns)]
        with Pool(nproc) as pool:
            tt = list(map(ca.main, confs))
            print(tt)
        Y[i * nruns: (i + 1) * nruns] = tt

        if conf.plotD or conf.plotAvgD:
            os.system('rm -rf dff-{}'.format(i))
            os.system('mv dff dff-{}'.format(i))

        if conf.plotP:
            os.system('rm -rf peds-{}'.format(i))
            os.system('mv peds peds-{}'.format(i))

    X = np.reshape([[x] * nruns for x in values], (i + 1) * nruns)

    plt.scatter(X, Y, marker='.')
    A = np.vstack([X, np.ones(len(X))]).T
    (m, t), r = np.linalg.lstsq(A, Y)[:2]

    plt.xlabel('$' + pretty(var) + '$', size=20)
    plt.ylabel(r'$t / s$', size=20)

    if r / len(values) < 100:
        plt.plot(X, m * X + t)
        plt.title(r't ~ {:.2f}s * N + {:.2f}s  |  Error: {:.2f}s'
                  .format(m, t, r[0] / len(values)))
    else:
        print('\n' + '-'*60)
        print('no apparent linear correlation between variables')
        avgs = [sum(Y[i * nruns : (i + 1) * nruns]) / nruns for i in range(len(values))]
        plt.plot(values, avgs)

    conf_dic = Config.__dict__.copy()
    conf_dic.update(conf.__dict__)
    conf_dic[var] = 'X'

    conf_str = CONF_STR.format(**conf_dic) + \
               CONF_STR_OPT.format(**conf_dic) * (bool(conf.kd) or var == 'kd')

    plt.savefig('figs/evac_times{}.png'.format(conf_str))

    with open('data/data{}.p'.format(conf_str), 'wb') as f:
        pickle.dump((X, Y), f) # use protocol = 2 for Python 2 compatibility

if __name__ == '__main__':
    ca.box = [1, 20, 1, 12]

    var = 'kd'

    values = np.linspace(0, 3, 10)

    nruns = 3

    time_var(var, values, nruns)


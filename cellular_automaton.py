#!/usr/bin/env python3

from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
import itertools  # for cartesian product
import time
import random
import os
import logging
import argparse

#######################################################
MAX_STEPS = 2048
steps = range(MAX_STEPS)

cellSize = 0.4  # m
vmax = 1.2
dt = cellSize / vmax  # time step

from_x, to_x = 1, 20  # todo parse this too
from_y, to_y = 1, 12  # todo parse this too
box = [from_x, to_x, from_y, to_y]
del from_x, to_x, from_y, to_y


# DFF = np.ones( (dim_x, dim_y) ) # dynamic floor field
#######################################################

logfile = 'log.dat'
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_parser_args():
    parser = argparse.ArgumentParser(
        description='Cellular Automaton. Floor Field Model [Burstedde2001] Simulation of pedestrian'
                    'dynamics using a two-dimensional cellular automaton Physica A, 2001, 295, 507-525')
    parser.add_argument('-s', '--ks', type=int, default=10,
                        help='sensitivity parameter for the  Static Floor Field (default 10)')
    parser.add_argument('-d', '--kd', type=int, default=10,
                        help='sensitivity parameter for the  Dynamic Floor Field (default 10)')
    parser.add_argument('-n', '--numPeds', type=int, default=10, help='Number of agents (default 10)')
    parser.add_argument('-p', '--plotS', action='store_const', const=True, default=False,
                        help='plot Static Floor Field')
    parser.add_argument('--plotD', action='store_const', const=True, default=False,
                        help='plot Dynamic Floor Field')
    parser.add_argument('--plotAvgD', action='store_const', const = True, default=False,
                        help='plot average Dynamic Floor Field')
    parser.add_argument('-P', '--plotP', action='store_const', const=True, default=False,
                        help='plot Pedestrians')
    parser.add_argument('-r', '--shuffle', action='store_const', const=True, default=False,
                        help='random shuffle')
    parser.add_argument('-v', '--reverse', action='store_const', const=True, default=False,
                        help='reverse sequential update')
    parser.add_argument('-l', '--log', type=argparse.FileType('w'), default='log.dat',
                        help='log file (default log.dat)')
    parser.add_argument('--decay', type=float, default=0.3,
                        help='the decay probability of the Dynamic Floor Field (default 0.2')
    parser.add_argument('--diffusion', type=float, default=0.1,
                        help='the diffusion probability of the Dynamic Floor Field (default 0.2)')
    parser.add_argument('-W', '--width', type=float, default=4.0,
                        help='the width of the simulated are in meters, excluding walls')
    parser.add_argument('-H', '--height', type=float, default=4.0,
                        help='the height of the are in meters, excluding walls')

    parser.add_argument('-c', '--clean', action='store_const', const=True, default=False,
                        help='remove files from dff/ and peds/')

    parser.add_argument('-N', '--nruns', type=int, default=1,
                        help='repeat the simulation N times')
    args = parser.parse_args()
    return args


def init_obstacles():
    return np.ones((dim_x, dim_y), int)  # obstacles/walls/boundaries


def init_walls(exit_cells, ):
    """
    define where are the walls. Consider the exits
    """
    OBST = init_obstacles()

    OBST[0, :] = OBST[-1, :] = OBST[:, -1] = OBST[:, 0] = np.Inf
    for e in exit_cells:
        OBST[e] = 1
    # OBST[dim_x//2, dim_y//2] = OBST[0, 0]
    # OBST[dim_x//2 + 1, dim_y//2] = OBST[0, 0]
    # print "walls"
    # print OBST
    return OBST


def check_N_pedestrians(box, N_pedestrians):
    """
    check if <N_pedestrian> is too big. if so change it to fit in <box>
    """
    # holding box, where to distribute pedestrians
    # ---------------------------------------------------
    from_x, to_x = box[0], box[1]
    from_y, to_y = box[2], box[3]
    # ---------------------------------------------------
    nx = to_x - from_x + 1
    ny = to_y - from_y + 1
    if N_pedestrians > nx * ny:
        logging.warning("N_pedestrians (%d) is too large (max. %d). Set to max." % (N_pedestrians, nx * ny))
        N_pedestrians = nx * ny

    return N_pedestrians


def init_peds(N, box):
    """
    distribute N pedestrians in box
    """
    from_x, to_x = box[:2]
    from_y, to_y = box[2:]
    nx = to_x - from_x + 1
    ny = to_y - from_y + 1
    PEDS = np.ones(N, int)  # pedestrians
    EMPTY_CELLS_in_BOX = np.zeros(nx * ny - N, int)  # the rest of cells in the box
    PEDS = np.hstack((PEDS, EMPTY_CELLS_in_BOX))  # put 0s and 1s together
    np.random.shuffle(PEDS)  # shuffle them
    PEDS = PEDS.reshape((nx, ny))  # reshape to a box
    EMPTY_CELLS = np.zeros((dim_x, dim_y), int)  # this is the simulation space
    EMPTY_CELLS[from_x:to_x + 1, from_y:to_y + 1] = PEDS  # put in the box
    return EMPTY_CELLS


def plot_sff(SFF, walls):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.cla()
    cmap = plt.get_cmap()
    cmap.set_bad(color='k', alpha=0.8)
    vect = SFF * walls
    vect[vect < -10] = np.Inf
    # print vect
    plt.imshow(vect, cmap=cmap, interpolation='nearest')  # lanczos nearest
    plt.colorbar()
    plt.savefig("SFF.png")
    # print "figure: SFF.png"

def plot_dff(dff, walls, name="DFF", max_value=None, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.cla()
    cmap = plt.get_cmap()
    cmap.set_bad(color='k', alpha=0.8)
    vect =  dff.copy()
    vect[walls < -10] = -np.Inf
    plt.imshow(vect, cmap=cmap, interpolation='nearest', vmin=0, vmax=max_value)  # lanczos nearest
    cbar = plt.colorbar()
    if title:
        plt.title(title)

    plt.savefig("dff/{}.png".format(name))
    plt.close()
    #print("figure: {}.png".format(name))


def plot_peds(peds, walls, i):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.cla()
    cmap = plt.get_cmap("gray")
    cmap.set_bad(color='b', alpha=0.8)
    N = np.sum(peds)
    # print N, type(N)
    # print peds+walls
    ax.imshow(peds + walls, cmap=cmap, vmin=-1, vmax=2,
              interpolation='nearest')  # 1-peds because I want the peds to be black
    S = 't: %3.3d  |  N: %3.3d ' % (i, N)
    plt.title("%8s" % S)
    figure_name = os.path.join('peds', 'peds%.5d.png' % i)
    plt.savefig(figure_name)
    plt.close()


def init_DFF():
    """
    """
    return np.zeros((dim_x, dim_y))


def update_DFF(dff, diff):
    for cell in diff:
        assert walls[cell] > -10
        dff[cell] += 1

    for i, j in itertools.product(range(dim_x), range(dim_y)):
        for _ in range(int(dff[i, j])):
            if np.random.rand() < delta: # decay
                dff[i, j] -= 1
            elif np.random.rand() < alpha: # diffusion
                dff[i, j] -= 1
                dff[random.choice(get_neighbors((i, j)))] += 1
        assert walls[i, j] > -10 or dff[i, j] == 0, (dff, i, j)
    # dff[:] = np.ones((dim_x, dim_y))

@lru_cache(1024**2)
def init_SFF(exit_cells, dim_x, dim_y):
    # start with exit's cells
    SFF = np.empty((dim_x, dim_y))  # static floor field
    SFF[:] = np.Inf

    cells_initialised = []
    for e in exit_cells:
        cells_initialised.append(e)
        SFF[e] = 0

    while cells_initialised:
        cell = cells_initialised.pop(0)
        neighbor_cells = get_neighbors(cell)
        for neighbor in neighbor_cells:
            # print "cell",cell, "neighbor",neighbor
            if SFF[cell] + 1 < SFF[neighbor]:
                SFF[neighbor] = SFF[cell] + 1
                cells_initialised.append(neighbor)

    for e in exit_cells:
        SFF[e] = -10
    return SFF


def get_neighbors(cell):
    """
     von Neumann neighborhood
    """
    neighbors = []
    i, j = cell

    if i < dim_y - 1:
        neighbors.append((i + 1, j))

    if i >= 1:
        neighbors.append((i - 1, j))

    if j < dim_x - 1:
        neighbors.append((i, j + 1))

    if j >= 1:
        neighbors.append((i, j - 1))

    return filter_walls(neighbors)


def filter_walls(cells):
    return [c for c in cells if walls[c] > -10]

def seq_update_cells(peds, sff, dff, prob_walls, kappaD, kappaS, shuffle, reverse):
    """
    sequential update
    input
       - peds:
       - sff:
       - dff:
       - prob_walls:
       - kappaD:
       - kappaS:
       - rand: random shuffle
    return
       - new peds
    """

    tmp_peds = np.empty_like(peds)  # temporary cells
    np.copyto(tmp_peds, peds)
    grid = list(itertools.product(range(1, dim_x), range(1, dim_y)))
    if shuffle:  # sequential random update
        random.shuffle(grid)
    elif reverse:  # reversed sequential update
        grid.reverse()

    dff_diff = []

    for (i, j) in grid:  # walk through all cells in geometry
        if peds[i, j] == 0:
            continue

        if (i, j) in exit_cells:
            tmp_peds[i, j] = 0
            dff_diff.append((i, j))
            continue

        p = 0
        probs = {}
        cell = (i, j)
        for neighbor in get_neighbors(cell):  # get the sum of probabilities
            probability = np.exp(-kappaS * sff[neighbor]) * np.exp(kappaD * dff[neighbor]) * (1 - tmp_peds[neighbor]) * \
                          prob_walls[neighbor]
            p += probability
            probs[neighbor] = probability

        if p == 0:  # pedestrian in cell can not move
            continue

        r = np.random.rand() * p
        # print ("start update")
        for neighbor in get_neighbors(cell): #TODO: shuffle?
            r -= probs[neighbor]  # todo this is calculated twice
            if r <= 0:  # move to neighbor cell
                tmp_peds[neighbor] = 1
                tmp_peds[i, j] = 0
                dff_diff.append((i, j))
                break

    return tmp_peds, dff_diff


def print_logs(N_pedestrians, width, height, t, dt, nruns, Dt):
    """
    print some infos to the screen
    """
    print ("Simulation of %d pedestrians" % N_pedestrians)
    print ("Simulation space (%.2f x %.2f) m^2" % (width, height))
    print ("SFF:  %.2f | DFF: %.2f" % (kappaS, kappaD))
    print ("Mean Evacuation time: %.2f s, runs: %d" % ((t * dt) / nruns, nruns))
    print ("Total Run time: %.2f s" % Dt)
    print ("Factor: x%.2f" % (dt * t / Dt))


def setup_dir(dir, clean):
    import os
    if os.path.exists(dir) and clean: os.system('rm -rf %s' % dir)
    os.makedirs(dir, exist_ok=True)


def simulate(n, N_pedestrians, sff, prob_walls, shuffle, reverse, drawP, drawD_avg):
    peds = init_peds(N_pedestrians, box)
    dff = init_DFF()

    for t in steps:  # simulation loop
        if drawP:
            plot_peds(peds, walls, t)
            print('\tn: %3d ----  t: %3d |  N: %3d' % (n, t, int(np.sum(peds))))

        peds, dff_diff = seq_update_cells(peds, sff, dff, prob_walls, kappaD, kappaS,
                                          shuffle, reverse)

        update_DFF(dff, dff_diff)

        if not peds.any():  # is everybody out?
            break
    else:
        raise TimeoutError("simulation taking to long")
    if drawD_avg:
        return t, dff.copy()
    else:
        return t


def main(args):
    global kappaS, kappaD, dim_y, dim_x, exit_cells, SFF, alpha, delta, walls
    # init parameters
    drawS = args.plotS  # plot or not
    drawP = args.plotP  # plot or not
    kappaS = args.ks
    kappaD = args.kd
    N_pedestrians = args.numPeds
    shuffle = args.shuffle
    reverse = args.reverse
    drawD = args.plotD
    drawD_avg = args.plotAvgD
    clean_dirs = args.clean
    width = args.width  # in meters
    height = args.height  # in meters
    dim_y = int(width / cellSize + 2 + 0.00000001)  # number of columns, add ghost cells
    dim_x = int(height / cellSize + 2 + 0.00000001)  # number of rows, add ghost cells

    nruns = args.nruns

    exit_cells = frozenset(((dim_x // 2, dim_y - 1), (dim_x // 2 + 1, dim_y - 1)))


    delta = args.decay
    alpha = args.diffusion

    N_pedestrians = check_N_pedestrians(box, N_pedestrians)

    walls = init_walls(exit_cells)

    sff = init_SFF(exit_cells, dim_x, dim_y)
    init_obstacles()
    if drawS:
        plot_sff(sff, walls)
    # to calculate probabilities change values of walls
    prob_walls = np.empty_like(walls)  # for calculating probabilities
    plot_walls = np.empty_like(walls)  # for ploting

    prob_walls[walls != 1] = 0  # not accessible
    prob_walls[walls == 1] = 1  # accessible
    plot_walls[walls != 1] = -10  # not accessible
    plot_walls[walls == 1] = 0  # accessible

    t1 = time.time()
    tsim = 0

    if drawP: setup_dir('peds', clean_dirs)
    if drawS: setup_dir('figs', clean_dirs)
    if drawD or drawD_avg: setup_dir('dff', clean_dirs)

    times = []
    old_dffs = []

    for n in range(nruns):

        if drawD_avg:
            t, sff = simulate(n, N_pedestrians, sff, prob_walls, shuffle, reverse, drawP, drawD_avg)
        else:
            t = simulate(n, N_pedestrians, sff, prob_walls, shuffle, reverse, drawP, drawD_avg)

        tsim += t
        times.append(t * dt)



    t2 = time.time()

    if drawD:
        max_dff = max(field.max() for _, field in old_dffs)
        for tm, dff in old_dffs:
            plot_dff(dff, walls, "DFF-{}".format(tm), max_dff, "t: %3.3d" % tm)
    if drawD_avg:
        plot_dff(sum(x[1] for x in old_dffs) / t, walls, "DFF-avg",
                 title="average over {:.2f} seconds in {} runs".format(sum(times), nruns))

    print_logs(N_pedestrians, width, height, tsim, dt, nruns, t2 - t1)
    return times

if __name__ == "__main__":
    args = get_parser_args()
    main(args)

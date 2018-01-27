#!/usr/bin/env python3

from functools import lru_cache
from multiprocessing.pool import Pool
import itertools as it # for cartesian product
import time
import random
import os
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

#######################################################
MAX_STEPS = 500
steps = range(MAX_STEPS)

cellSize = 0.4  # m
vmax = 1.2
dt = cellSize / vmax  # time step

from_x, to_x = 1, 63  # todo parse this too
from_y, to_y = 1, 63  # todo parse this too
DEFAULT_BOX = [from_x, to_x, from_y, to_y]
del from_x, to_x, from_y, to_y



# DFF = np.ones( (dim_x, dim_y) ) # dynamic floor field
#######################################################

logfile = 'log.dat'
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_parser_args():
    parser = argparse.ArgumentParser(
        description='Cellular Automaton. Floor Field Model [Burstedde2001] Simulation of pedestrian'
                    'dynamics using a two-dimensional cellular automaton Physica A, 2001, 295, 507-525')
    parser.add_argument('-s', '--ks', type=float, default=2,
                        help='sensitivity parameter for the  Static Floor Field (default 2)')
    parser.add_argument('-d', '--kd', type=float, default=1,
                        help='sensitivity parameter for the  Dynamic Floor Field (default 1)')
    parser.add_argument('-n', '--numPeds', type=int, default=10, help='Number of agents (default 10)')
    parser.add_argument('-p', '--plotS', action='store_const', const=True, default=False,
                        help='plot Static Floor Field')
    parser.add_argument('--plotD', action='store_const', const=True, default=False,
                        help='plot Dynamic Floor Field')
    parser.add_argument('--plotAvgD', action='store_const', const = True, default=False,
                        help='plot average Dynamic Floor Field')
    parser.add_argument('-P', '--plotP', action='store_const', const=True, default=False,
                        help='plot Pedestrians')
    parser.add_argument('-r', '--shuffle', action='store_const', const=True, default=True,
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

    parser.add_argument('--parallel', action='store_const', const=True, default=False,
                        help='use multithreading')

    parser.add_argument('--box', type=int, nargs=4, default=DEFAULT_BOX,
                        help='are which is populated with pedestrians: (from_x, to_x, from_y, to_y')

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
    logging.info("Init peds finished. Box: x: [%.2f, %.2f]. y: [%.2f, %.2f]",
                 from_x, to_x, from_y, to_y)
    return EMPTY_CELLS

def plot_sff2(SFF, walls, i):
    """
    plots a numbered image. Useful for making movies
    """
    print(i)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.cla()
    plt.set_cmap('jet')
    cmap = plt.get_cmap()
    cmap.set_bad(color='k', alpha=0.8)
    vect = SFF * walls
    vect[vect < -200] = np.Inf
#    print (vect)
    max_value = np.max(SFF)
    min_value = np.min(SFF)
    plt.imshow(vect, cmap=cmap, interpolation='nearest', vmin=min_value, vmax=max_value, extent=[0, dim_x, 0, dim_y])  # lanczos nearest
    plt.colorbar()
 #   print(i)
    plt.title("%.6d"%i)
    plt.savefig("sff/%.6d.png"%i)

def plot_sff(SFF, walls):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.cla()
    plt.set_cmap('jet')
    cmap = plt.get_cmap()
    cmap.set_bad(color='k', alpha=0.8)
    vect = SFF * walls
    vect[vect < -200] = -np.Inf
    # print vect
    max_value = np.max(SFF)
    min_value = np.min(SFF)
    plt.imshow(vect, cmap=cmap, interpolation='nearest', vmin=min_value, vmax=max_value, extent=[0, dim_x, 0, dim_y])  # lanczos nearest
    plt.colorbar()
    plt.savefig("sff/SFF.pdf")
    plt.savefig("sff/SFF.png", dpi=600)
    print ("figure: sff/SFF.pdf")

def plot_dff(dff, walls, name="DFF", max_value=None, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.cla()
    plt.set_cmap('jet')    
    cmap = plt.get_cmap()
    cmap.set_bad(color='k', alpha=0.8)
    vect =  dff.copy()
    vect[walls < -10] = -np.Inf
    im = ax.imshow(vect, cmap=cmap, interpolation='nearest', vmin=0, vmax=max_value, extent=[0, dim_x, 0, dim_y])  # lanczos nearest
    plt.colorbar(im, format='%.1f')
    #cbar = plt.colorbar()
    if title:
        plt.title(title)

    plt.savefig("dff/{}.png".format(name), dpi=600)
    plt.close()
    logging.info("plot dff. figure: {}.png".format(name))


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
    figure_name = os.path.join('peds', 'peds%.6d.png' % i)
    plt.savefig(figure_name)
    plt.close()


def init_DFF():
    """
    """
    return np.zeros((dim_x, dim_y))


def update_DFF(dff, diff):
    #for cell in diff:
    #    assert walls[cell] > -10
    #     dff[cell] += 1

    dff += diff

    for i, j in it.chain(it.product(range(1, dim_x - 1), range(1, dim_y - 1)), exit_cells):
        for _ in range(int(dff[i, j])):
            if np.random.rand() < delta: # decay
                dff[i, j] -= 1
            elif np.random.rand() < alpha: # diffusion
                dff[i, j] -= 1
                dff[random.choice(get_neighbors((i, j)))] += 1
        assert walls[i, j] > -10 or dff[i, j] == 0, (dff, i, j)
    # dff[:] = np.ones((dim_x, dim_y))

@lru_cache(1)
def init_SFF(exit_cells, dim_x, dim_y, drawS):
    # start with exit's cells
    SFF = np.empty((dim_x, dim_y))  # static floor field
    SFF[:] = np.sqrt(dim_x ** 2 + dim_y ** 2)

    make_videos = 0
    if make_videos and drawS:
        plot_sff2(SFF, walls, 1)

    cells_initialised = []
    for e in exit_cells:
        cells_initialised.append(e)
        SFF[e] = 0

    if make_videos and drawS:
        plot_sff2(SFF, walls, 2)
        i = 3
    while cells_initialised:
        cell = cells_initialised.pop(0)
        neighbor_cells = get_neighbors(cell)
        for neighbor in neighbor_cells:
            # print ("cell",cell, "neighbor",neighbor)
            if SFF[cell] + 1 < SFF[neighbor]:
                SFF[neighbor] = SFF[cell] + 1
                cells_initialised.append(neighbor)
                # print(SFF)
        # print(cells_initialised)
        if make_videos and drawS:
            plot_sff2(SFF, walls, i)
            i += 1

    return SFF

@lru_cache(16*1024)
def get_neighbors(cell):
    """
     von Neumann neighborhood
    """
    neighbors = []
    i, j = cell

    if i < dim_y - 1 and walls[(i + 1, j)] > -10:
        neighbors.append((i + 1, j))
    if i >= 1 and walls[(i - 1, j)] > -10:
        neighbors.append((i - 1, j))
    if j < dim_x - 1 and walls[(i, j + 1)] > -10:
        neighbors.append((i, j + 1))
    if j >= 1 and walls[(i, j - 1)] > -10:
        neighbors.append((i, j - 1))

    # not shuffling singnificantly alters the simulation...
    random.shuffle(neighbors)
    return neighbors


def seq_update_cells(peds, sff, dff, kappaD, kappaS, shuffle, reverse):
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

    dff_diff = np.zeros((dim_x, dim_y))



    grid = list(it.product(range(1, dim_x - 1), range(1, dim_y - 1))) + list(exit_cells)
    if shuffle:  # sequential random update
        random.shuffle(grid)
    elif reverse:  # reversed sequential update
        grid.reverse()

    for (i, j) in grid:  # walk through all cells in geometry
        if peds[i, j] == 0:
            continue

        if (i, j) in exit_cells:
            tmp_peds[i, j] = 0
            dff_diff[i, j] += 1
            continue

        p = 0
        probs = {}
        cell = (i, j)
        for neighbor in get_neighbors(cell):  # get the sum of probabilities
            # original code:
            # probability = np.exp(-kappaS * sff[neighbor]) * np.exp(kappaD * dff[neighbor]) * \
            # (1 - tmp_peds[neighbor])
            # the absolute value of the exponents can get very large yielding 0 or
            # inifite probability.
            # to prevent this we multiply every probability with exp(kappaS * sff[cell) and
            # exp(-kappaD * dff[cell]).
            # since the probabilities are normalized this doesn't have any effect on the model

            probability = np.exp(kappaS * (sff[cell] - sff[neighbor])) * \
                          np.exp(kappaD * (dff[neighbor] - dff[cell])) * \
                          (1 - tmp_peds[neighbor])

            p += probability
            probs[neighbor] = probability

        if p == 0:  # pedestrian in cell can not move
            continue

        r = np.random.rand() * p
        # print ("start update")
        for neighbor in get_neighbors(cell): #TODO: shuffle?
            r -= probs[neighbor]
            if r <= 0:  # move to neighbor cell
                tmp_peds[neighbor] = 1
                tmp_peds[i, j] = 0
                dff_diff[i, j] += 1
                break

    return tmp_peds, dff_diff


def print_logs(N_pedestrians, width, height, t, dt, nruns, Dt):
    """
    print some infos to the screen
    """
    print ("Simulation of %d pedestrians" % N_pedestrians)
    print ("Simulation space (%.2f x %.2f) m^2" % (width, height))
    print ("SFF:  %.2f | DFF: %.2f" % (kappaS, kappaD))
    print ("Mean Evacuation time: %.2f s, runs: %d" % (t * dt / nruns, nruns))
    print ("Total Run time: %.2f s" % Dt)
    print ("Factor: x%.2f" % (dt * t / Dt))


def setup_dir(dir, clean):
    print("make ", dir)
    if os.path.exists(dir) and clean:
        os.system('rm -rf %s' % dir)
    os.makedirs(dir, exist_ok=True)


def simulate(args):

    n, npeds, box, sff, shuffle, reverse, drawP, giveD = args
    peds = init_peds(npeds, box)
    dff = init_DFF()

    old_dffs = []
    for t in steps:  # simulation loop
        print('\tn: %3d ----  t: %3d |  N: %3d' % (n, t, int(np.sum(peds))))
        if drawP:
            plot_peds(peds, walls, t)

        peds, dff_diff = seq_update_cells(peds, sff, dff, kappaD, kappaS,
                                          shuffle, reverse)

        update_DFF(dff, dff_diff)
        if giveD:
            old_dffs.append((t, dff.copy()))

        if not peds.any() or int(np.sum(peds)) == 1:  # is everybody out?
            print("Quite simulation")
            break
    # else:
    #     raise TimeoutError("simulation taking too long")

    if giveD:
        return t, old_dffs
    else:
        return t





def main(args):
    global kappaS, kappaD, dim_y, dim_x, exit_cells, SFF, alpha, delta, walls, parallel, box
    # init parameters
    drawS = args.plotS  # plot or not
    drawP = args.plotP  # plot or not
    kappaS = args.ks
    kappaD = args.kd
    npeds = args.numPeds
    shuffle = args.shuffle
    reverse = args.reverse
    drawD = args.plotD
    drawD_avg = args.plotAvgD
    clean_dirs = args.clean
    width = args.width  # in meters
    height = args.height  # in meters
    parallel = args.parallel
    box = args.box
    # check if no box is specified

    if parallel and drawP :
        raise NotImplementedError("cannot plot pedestrians when multiprocessing")



    dim_y = int(width / cellSize + 2 + 0.00000001)  # number of columns, add ghost cells
    dim_x = int(height / cellSize + 2 + 0.00000001)  # number of rows, add ghost cells
    print("cellsize: ", cellSize, " dim_x: ", dim_x, " dim_y: ", dim_y)
    if box == DEFAULT_BOX:
        print("box == room")
        box = [2, dim_x - 1, 2, dim_y - 1]


    nruns = args.nruns

    exit_cells = frozenset(((dim_x // 2, dim_y - 1), (dim_x // 2 + 1, dim_y - 1)))


    delta = args.decay
    alpha = args.diffusion

    npeds = check_N_pedestrians(box, npeds)

    walls = init_walls(exit_cells)

    sff = init_SFF(exit_cells, dim_x, dim_y, drawS)
    init_obstacles()
    if drawS:
        setup_dir('sff', clean_dirs)
        plot_sff(sff, walls)


    # to calculate probabilities change values of walls
    # prob_walls = np.empty_like(walls)  # for calculating probabilities
    # plot_walls = np.empty_like(walls)  # for ploting

    # prob_walls[walls != 1] = 0  # not accessible
    # prob_walls[walls == 1] = 1  # accessible
    # plot_walls[walls != 1] = -10  # not accessible
    # plot_walls[walls == 1] = 0  # accessible

    t1 = time.time()
    tsim = 0

    if drawP: setup_dir('peds', clean_dirs)
    if drawD or drawD_avg: setup_dir('dff', clean_dirs)

    times = []
    old_dffs = []

    if not parallel:
        for n in range(nruns):
            print("n= ", n, " nruns=", nruns)
            if drawD_avg or drawD:
                t, dffs = simulate((n, npeds, box, sff, shuffle, reverse,
                                    drawP, drawD_avg or drawD))
                old_dffs += dffs
            else:
                t = simulate((n, npeds, box, sff, shuffle, reverse, drawP,
                              drawD_avg or drawD))
            tsim += t
            print("time ", tsim)
            times.append(t * dt)
    else:

        nproc = min(nruns, 8)
        print('using {} processes'.format(nproc))
        jobs = [(n, npeds, box, sff, shuffle, reverse, drawP, drawD_avg or drawD)
                for n in range(nruns)]

        with Pool(nproc) as pool:
            results = pool.map(simulate, jobs)


        if drawD_avg or drawD:
            ts, chunked_dffs = zip(*results)
            times = [t * dt for t in ts]
            tsim = sum(ts)
            old_dffs = sum(chunked_dffs, [])
        else:
            times = [t * dt for t in results]
            tsim = sum(results)




    t2 = time.time()
    print_logs(npeds, width, height, tsim, dt, nruns, t2 - t1)
    if drawD_avg:
        print('plotting average DFF')
        plot_dff(sum(x[1] for x in old_dffs) / tsim, walls, "DFF-avg_runs_%d_N%d_S%.2f_D%.2f"%(nruns,npeds,kappaS,kappaD))
# title=r"$t = {:.2f}$ s, N={}, #runs = {}, $\kappa_S={}\;, \kappa_D={}$".format(sum(times), npeds, nruns, kappaS, kappaD)
    if drawD:
        print('plotting DFFs...')
        max_dff = max(field.max() for _, field in old_dffs)
        for tm, dff in old_dffs:
            print("t: %3.4d" % tm)
            plot_dff(dff, walls, "DFF-%3.4d"%tm, max_dff, "t: %3.4d" % tm)

    return times

if __name__ == "__main__":
    args = get_parser_args()
    main(args)

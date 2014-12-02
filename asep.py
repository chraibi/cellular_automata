# Implementation of the Asymmetric Simple Exclusion Process (ASEP)
#     Copyright (C) 2014-2015  Mohcine Chraibi
#
#     This program is free software; you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation; either version 2 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License along
#     with this program; if not, write to the Free Software Foundation, Inc.,
#     51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# contact: m.chraibi@gmail.com
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import logging
import argparse
import os

logfile = 'log.dat'
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_parser_args():
    parser = argparse.ArgumentParser(description='ASEP - TASEP')
    parser.add_argument('-n', '--np', type=int, default=10, help='number of agents (default 10)')
    parser.add_argument('-N', '--nr', type=int, default=1, help='number of runs (default 1)')
    parser.add_argument('-m', '--ms', type=int, default=100, help='max simulation steps (default 100)')
    parser.add_argument('-w', '--width', type=int, default=50, help='max simulation steps (default 50)')
    parser.add_argument('-p', '--plotP', action='store_const', const=1, default=0, help='plot Pedestrians')
    parser.add_argument('-r', '--shuffle', action='store_const', const=1, default=0, help='random shuffle')
    parser.add_argument('-v', '--reverse', action='store_const', const=1, default=0, help='reverse sequential update')
    parser.add_argument('-l', '--log', type=argparse.FileType('w'), default='log.dat',
                        help='log file (default log.dat)')
    args = parser.parse_args()
    return args


def init_cells(num_peds, num_cells):
    """
    distribute N pedestrians in box 
    """
    if num_peds > num_cells:
        num_peds = num_cells

    cells = np.ones(num_peds, int)  # pedestrians
    zero_cells = np.zeros(num_cells - num_peds, int)  # the rest of cells in the box
    cells = np.hstack((cells, zero_cells))  # put 0s and 1s together
    np.random.shuffle(cells)  # shuffle them
    return cells


def plot_cells(state_cells, walls_inf, i):
    """
    plot the actual state of the cells. we need to make 'bad' walls to better visualize the cells
    :param state_cells: state of the cells
    :param walls_inf: walls for visualisation purposes
    :param i: index for figures
    """
    walls_inf = walls_inf * np.Inf
    tmp_cells = np.vstack((walls_inf, state_cells))
    tmp_cells = np.vstack((tmp_cells, walls_inf))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.cla()
    cmap = plt.get_cmap('gray')
    cmap.set_bad(color='k', alpha=0.8)
    im = ax.imshow(tmp_cells, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='1%', pad=0.1)
    plt.colorbar(im, cax=cax, ticks=[0, 1])
    ax.set_axis_off()
    N = sum(state_cells)
    text = "t: %3.3d | n: %d\n" % (i, N)
    plt.title("%20s" % text, rotation=0, fontsize=10, verticalalignment='bottom')
    figure_name = os.path.join('pngs', u'peds{0:.5d}.png'.format(i))
    plt.savefig(figure_name, dpi=100, facecolor='lightgray')


def print_logs(num_pedestrians, system_width, simulation_steps, evac_time, total_runtime, nruns, vel, d):
    """
    print some information to the screen
    :rtype : none
    """
    print('\n')
    print ('Simulation space (%.2f x 1) m^2' % system_width)
    print ('Mean Evacuation time: %.2f s, runs: %d' % ((evac_time * dt) / nruns, nruns))
    print ('max simulation steps %d' % simulation_steps)
    print ('Total Run time: %.2f s' % total_runtime)
    print ('Factor: %.2f s' % (dt * evac_time / total_runtime))
    print('--------------------------')
    print ('N %d   mean_velocity  %.2f [m/s]   density  %.2f [1/m]' % (num_pedestrians, vel, d))
    print('--------------------------')


# http://stackoverflow.com/questions/27239173/numpy-vectorize-a-parallel-update
def boundary(boundary_cells):
    """enforce boundary conditions
    :rtype : np.ndarray
    """
    boundary_cells = np.concatenate([[0], boundary_cells, [0]])  # add padding cells
    boundary_cells[0] = boundary_cells[-2]
    boundary_cells[-1] = boundary_cells[1]
    return boundary_cells


def asep_parallel(cells):
    """
    update of cells
    :parameter: actual state of cells
    :rtype : cells with new state and number of moves
    """
    assert isinstance(cells, np.ndarray)
    cells = boundary(cells)
    center = cells[1:-1]
    left = cells[0:-2]
    right = cells[2:]
    ones = (center == 1)
    zeros = (center == 0)
    result = np.copy(center)
    result[zeros] = left[zeros]
    result[ones] = right[ones]
    nmoves = sum(np.logical_xor(center, result)) / 2
    return result, nmoves


def asep_parallel2(actual_cells):
    """ equivalent asep_parallel(), but without vectorisation od numpy. Less elegant, but maybe easier to understand ..
    :param actual_cells:
    :return: new cells and number of moves
    """
    """
    :param actual_cells:
    :return:
    """
    neighbors = np.roll(actual_cells, -1)
    assert isinstance(actual_cells, np.ndarray)
    dim = len(actual_cells)
    num_move = 0
    tmp_cells = np.zeros(dim)
    for i, j in enumerate(actual_cells):
        if j and not neighbors[i]:
            tmp_cells[i], tmp_cells[(i + 1) % dim] = 0, 1
            num_move += 1
        elif j:
            tmp_cells[i] = 1
    return tmp_cells, num_move


if __name__ == "__main__":
    args = get_parser_args()  # get arguments
    # init parameters
    N_pedestrians = args.np
    shuffle = args.shuffle
    reverse = args.reverse
    drawP = args.plotP
    #######################################################
    max_steps = args.ms  # simulation time
    num_runs = args.nr  # repeat simulations, for TASEP
    steps = range(max_steps)
    cellSize = 0.4  # m
    max_velocity = 1.2  # m/s
    dt = cellSize / max_velocity  # time step
    width = args.width  # in m
    n_cells = int(width / cellSize)  # number of cells
    if N_pedestrians >= n_cells:
        N_pedestrians = n_cells - 1
        print('warning: maximum of %d cells are allowed' % N_pedestrians)
    else:
        print('info: n_pedestrians=%d (max_pedestrians=%d)' % (N_pedestrians, n_cells))
    #######################################################
    t1 = time.time()
    simulation_time = 0
    density = float(N_pedestrians) / width
    walls = np.ones(n_cells)
    velocities = []  # velocities over all runs
    for n in range(num_runs):
        actual_cells = init_cells(N_pedestrians, n_cells)
        velocity = 0
        for step in steps:  # simulation loop
            if drawP:
                plot_cells(actual_cells, walls, step)

            actual_cells, num_moves = asep_parallel(actual_cells)
            v = num_moves * max_velocity / float(N_pedestrians)
            velocity += v

        velocity /= max_steps
        velocities.append(velocity)
        simulation_time += max_steps
        print ('\t run: %3d ----  vel: %.2f |  den: %.2f' % (n, velocity, density))
    t2 = time.time()
    mean_velocity = np.mean(velocities)
    print_logs(N_pedestrians, width, max_steps, simulation_time, t2 - t1, num_runs, mean_velocity, density)

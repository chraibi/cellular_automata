from functools import lru_cache
import itertools as it # for cartesian product
import random
import os
import logging
# import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class automaton:
    """
    init class defining system
    """
    def __init__(self, args):
        self.npeds = args.numPeds
        self.peds = None
        self.sff = None # static floor field
        self.dff = None # dynamic floor field
        self.mean_dff = None # dynamic floor field
        # drawing parameter
        self.drawS = args.plotS # plot ssf
        self.drawP = args.plotP # plot peds
        self.drawD = args.plotD # plot dff
        self.drawD_avg = args.plotAvgD # plot average dff
        self.image = None
        self.fig = None
        self.title = ''
        # model parameter
        self.kappaS = args.ks
        self.kappaD = args.kd
        self.delta = args.decay
        self.alpha = args.diffusion
        self.moore = args.moore
        # Update parameter
        self.shuffle = args.shuffle
        self.reverse = args.reverse
        self.parallel = args.parallel
        # 2D space parameter
        self.width = args.width  # in meters
        self.height = args.height  # in meters
        self.CELL_SIZE = 0.4 # in m
        # number of columns, add ghost cells
        self.ncols = int(self.width / self.CELL_SIZE + 2 + 0.00000001)
        # number of rows, add ghost cells
        self.nrows = int(self.height / self.CELL_SIZE + 2 + 0.00000001)
        self.exit_cells = frozenset((
            (self.nrows // 2, self.ncols - 1), (self.nrows // 2 + 1, self.ncols - 1),
            (self.nrows - 1, self.ncols//2 + 1), (self.nrows - 1, self.ncols//2),
            (0, self.ncols//2 + 1), (1, self.ncols//2),
            (self.nrows//2 + 1, 0), (self.nrows//2, 0)
        ))
        self.grid = list(it.product(range(1, self.nrows - 1), range(1, self.ncols - 1))) + list(self.exit_cells)
        self.walls = None # will be initialised in init_walls()
        # Simulation parameter
        self.box = args.box # where to distribute peds
        self.nruns = args.nruns
        self.VMAX = 1.2 # in m/s
        self.DT = self.CELL_SIZE / self.VMAX  # TIME step
        self.frame = 0
        self.run_time = 0
        self.clean_dirs = args.clean # clean up directories
        if self.box == [0, 10, 0, 10]:
            self.box = [1, self.nrows - 2, 1, self.ncols - 2]

        self.init_simulation()
        # --------- init variables -------------

    def init_simulation(self):
        self.check_box()
        self.check_N_pedestrians()
        self.init_obstacles()
        self.init_walls()
        self.init_peds()
        self.init_sff()
        self.init_dff()
        if self.drawS:
            setup_dir('sff', self.clean_dirs)
            self.plot_ff(self.sff, "sff")


    def check_box(self):
        """
        exit if box is not well defined
        """
        assert (self.box[0] < self.box[1]), "from_x bigger than to_x"
        assert (self.box[2] < self.box[3]), "from_y bigger than to_y"

    def check_N_pedestrians(self):
        """
        check if <N_pedestrian> is too big.
        if so change it to fit in <box>
        """
        # holding box, where to distribute pedestrians
        _from_x = self.box[0]
        _to_x = self.box[1]
        _from_y = self.box[2]
        _to_y = self.box[3]
        nx = _to_x - _from_x + 1
        ny = _to_y - _from_y + 1
        if self.npeds > nx * ny:
            logging.warning("N_pedestrians (%d) is too large (max. %d). Set to max." % (self.npeds, nx * ny))
            self.npeds = nx * ny

    def init_obstacles(self):
        self.obstacles = np.ones((self.nrows, self.ncols), int)

    @lru_cache(16*1024)
    def get_neighbors(self, cell):
        """
        Default: von Neumann neighborhood
        if flag, moore neighborhood
        """
        neighbors = []
        i, j = cell
        if i < self.nrows - 1 and self.walls[(i + 1, j)] >= 0:
            neighbors.append((i + 1, j))
        if i >= 1 and self.walls[(i - 1, j)] >= 0:
            neighbors.append((i - 1, j))
        if j < self.ncols - 1 and self.walls[(i, j + 1)] >= 0:
            neighbors.append((i, j + 1))
        if j >= 1 and self.walls[(i, j - 1)] >= 0:
            neighbors.append((i, j - 1))

        # moore
        if self.moore:
            if i >= 1 and j >= 1 and self.walls[(i-1, j - 1)] >= 0:
                neighbors.append((i-1, j - 1))
            if i < self.nrows - 1 and  j < self.ncols -1  and self.walls[(i+1, j+1)] >= 0:
                neighbors.append((i+1, j + 1))
            if i < self.nrows - 1 and  j >= 1  and self.walls[(i+1, j-1)] >= 0:
                neighbors.append((i+1, j - 1))
            if i >= 1 and  j < self.ncols -1  and self.walls[(i-1, j+1)] >= 0:
                neighbors.append((i-1, j + 1))

        random.shuffle(neighbors)
        return neighbors


    def init_walls(self):
        """
        define where are the walls. Consider the exits
        """
        walls = np.copy(self.obstacles)
        walls[0, :] = walls[-1, :] = walls[:, -1] = walls[:, 0] = -1
        for e in self.exit_cells:
            walls[e] = 1

        self.walls = walls

    def init_dff(self):
        """
        initialize the dff with zeros.
        self.dff will be updated every step of the simulation
        """
        self.dff = np.zeros((self.nrows, self.ncols))
        self.mean_dff = np.zeros((self.nrows, self.ncols))

    @lru_cache(1)
    def init_sff(self):
        """
        initialise static floor field
        self.sff is initialised
        """
        # start with exit's cells
        SFF = np.empty((self.nrows, self.ncols))  # static floor field
        MAX_sff = np.sqrt(self.nrows ** 2 + self.ncols ** 2)
        SFF[:] = MAX_sff
        cells_initialised = []
        for e in self.exit_cells:
            cells_initialised.append(e)
            SFF[e] = 0

        while cells_initialised:
            cell = cells_initialised.pop(0)
            neighbor_cells = self.get_neighbors(cell)
            for neighbor in neighbor_cells:
                if SFF[cell] + 1 < SFF[neighbor]:
                    SFF[neighbor] = SFF[cell] + 1
                    cells_initialised.append(neighbor)

        sff_second_max = np.amax(SFF[SFF != np.amax(SFF)])
        SFF[SFF == MAX_sff] = sff_second_max
        self.sff = SFF

    def init_peds(self):
        """
        distribute pedestrians in box
        self.peds is initialised
        """
        from_x, to_x = self.box[:2]
        from_y, to_y = self.box[2:]
        nx = to_x - from_x + 1
        ny = to_y - from_y + 1
        PEDS = np.ones(self.npeds, int)  # pedestrians
        # the rest of cells in the box
        empty_cells_in_box = np.zeros(nx * ny - self.npeds, int)
        # put 0s and 1s together
        PEDS = np.hstack((PEDS, empty_cells_in_box))
        np.random.shuffle(PEDS)
        PEDS = PEDS.reshape((nx, ny))
        world = np.zeros((self.nrows, self.ncols), int)
        world[from_x:to_x + 1, from_y:to_y + 1] = PEDS
        logging.info("Init peds in Box: x: [%.2f, %.2f]. y: [%.2f, %.2f]",
                     from_x, to_x, from_y, to_y)

        self.peds = world

    def step(self, frame):
        """
        sequential update of the system
        updates the following
        - self.peds
        - self.dff
        """

        self.frame = frame
        tmp_peds = np.empty_like(self.peds)  # temporary cells
        np.copyto(tmp_peds, self.peds)
        N = np.sum(self.peds)
        print('frame: %4.3d\t |   time:  %4.2f\t |  N: %4.3d ' % (self.frame, self.frame*self.DT, N))
        dff_diff = np.zeros((self.nrows, self.ncols))
        if self.shuffle:  # sequential random update
            random.shuffle(self.grid)
        elif self.reverse:  # reversed sequential update
            self.grid.reverse()

        for (i, j) in self.grid:  # walk through all cells in geometry
            if self.peds[i, j] == 0:
                continue

            if (i, j) in self.exit_cells:
                tmp_peds[i, j] = 0
                dff_diff[i, j] += 1
                continue

            p = 0
            probs = {}
            cell = (i, j)
            for neighbor in self.get_neighbors(cell):  # get the sum of probabilities
                # original code:
                # probability = np.exp(-kappaS * sff[neighbor]) * np.exp(kappaD * dff[neighbor]) * \
                # (1 - tmp_peds[neighbor])
                # the absolute value of the exponents can get very large yielding 0 or
                # inifite probability.
                # to prevent this we multiply every probability with exp(kappaS * sff[cell) and
                # exp(-kappaD * dff[cell]).
                # since the probabilities are normalized this doesn't have any effect on the model

                probability = np.exp(self.kappaS * (self.sff[cell] - self.sff[neighbor])) * \
                              np.exp(self.kappaD * (self.dff[neighbor] - self.dff[cell])) * \
                              (1 - tmp_peds[neighbor])

                p += probability
                probs[neighbor] = probability

            if p == 0:  # pedestrian in cell can not move
                continue

            r = np.random.rand() * p
            # print ("start update")
            for neighbor_ in self.get_neighbors(cell):
                r -= probs[neighbor_]
                if r <= 0:  # move to neighbor cell
                    tmp_peds[neighbor_] = 1
                    tmp_peds[i, j] = 0
                    dff_diff[i, j] += 1
                    break

        self.update_dff(dff_diff)
        self.peds = tmp_peds
        if not self.peds.any():
            raise StopIteration()

        title = 'frame: %3.3d |   time:  %3.2f |  N: %3.3d ' % (self.frame, self.frame*self.DT, N)
        self.title.set_text(title)
        self.image.set_array(self.peds+self.walls)


    def update(self, frame):
        try:
            self.step(frame)
        except StopIteration:
            close(self.fig)
            if self.drawD_avg:
                setup_dir('dff', self.clean_dirs)
                self.plot_ff(self.mean_dff/self.frame, "dff")


        return self.image, #, self.title,



    def update_dff(self, diff):
        self.dff += diff

        for i, j in it.chain(it.product(range(1, self.nrows - 1), range(1, self.ncols - 1)), self.exit_cells):
            for _ in range(int(self.dff[i, j])):
                if np.random.rand() < self.delta: # decay
                    self.dff[i, j] -= 1
                elif np.random.rand() < self.alpha: # diffusion
                    self.dff[i, j] -= 1
                    self.dff[random.choice(self.get_neighbors((i, j)))] += 1
            assert self.walls[i, j] > -1 or self.dff[i, j] == 0, (self.dff, i, j)

        self.mean_dff += self.dff

    def print_logs(self):
        """
        print some infos to the screen
        """
        print("Simulation of %d pedestrians" % self.npeds)
        print("Simulation space (%.2f x %.2f) m^2" % (self.width, self.height))
        print("SFF:  %.2f | DFF: %.2f" % (self.kappaS, self.kappaD))
        print("Diffusion:  %.2f | Decay: %.2f" % (self.diffusion, self.decay))
        print("Mean Evacuation time: %.2f s, runs: %d" % (self.frame * self.dt / self.nruns, self.nruns))
        print("Total Run time: %.2f s" % self.run_time)
        print("Factor: x%.2f" % (self.dt * self.frame / self.run_time))

    def plot_peds(self, fig, ax):
        self.fig = fig
        cmap = plt.get_cmap("gray")
        cmap.set_bad(color='b', alpha=0.8)
        N = np.sum(self.peds)
        im = ax.imshow(self.peds + self.walls, cmap=cmap, interpolation='nearest', vmin=-1, vmax=2, animated=True)
        plt.grid(True, color='k', alpha=0.6)
        plt.yticks(np.arange(1.5, self.peds.shape[0], 1))
        plt.xticks(np.arange(1.5, self.peds.shape[1], 1))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)

        title = 'frame: %3.3d |   time:  %3.2f |  N: %3.3d ' % (self.frame, self.frame*self.DT, N)
        self.title = ax.text(0.5,1.05,  '%8s' % title,
                        transform=ax.transAxes, ha="center")
        return im


    def plot_ff(self, ff, name):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.set_cmap('jet')
        cmap = plt.get_cmap()
        cmap.set_bad(color='k', alpha=0.8)
        vect = ff.copy()
        vect[self.walls < 0] = np.Inf
        max_value = np.max(ff)
        min_value = np.min(ff)
        im = ax.imshow(vect, cmap=cmap, interpolation='nearest', vmin=min_value, vmax=max_value, extent=[0, self.ncols, 0, self.nrows])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        figure_name = os.path.join(name, '%s.png'%name)
        print("plot %s"%figure_name)
        plt.savefig(figure_name, dpi=600)
        plt.close()
        return im

def close(fig):
    plt.close(fig)

def setup_dir(Dir, clean):
    print("make ", Dir)
    if os.path.exists(Dir) and clean:
        os.system('rm -rf %s' % Dir) # TODO: this is OS specific
    os.makedirs(Dir, exist_ok=True)

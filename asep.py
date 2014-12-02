import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time, random 
import logging, types, argparse

logfile='log.dat'
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def getParserArgs():
    parser = argparse.ArgumentParser(description='ASEP - TASEP')
    parser.add_argument("-n", "--np", type=int , default=10, help='number of agents (default 10)')
    parser.add_argument("-N", "--nr", type=int , default=1, help='number of runs (default 1)')
    parser.add_argument("-p", "--plotP", action='store_const', const=1, default=0, help='plot Pedestrians')
    parser.add_argument("-r", "--shuffle", action='store_const', const=1,  default=0, help='random shuffle')
    parser.add_argument("-v", "--reverse", action='store_const', const=1,  default=0, help='reverse sequential update')
    parser.add_argument("-l", "--log" , type=argparse.FileType('w'), default='log.dat', help="log file (default log.dat)")
    args = parser.parse_args()
    return args


def init_peds(N, n_cells):
    """
    distribute N pedestrians in box 
    """
    if N>n_cells: 
        N = n_cells
    PEDS = np.ones(N, int) # pedestrians    
    EMPTY_CELLS_in_BOX = np.zeros( n_cells - N, int) # the rest of cells in the box
    PEDS = np.hstack((PEDS,  EMPTY_CELLS_in_BOX)) # put 0s and 1s together
    np.random.shuffle(PEDS) # shuffle them
    return PEDS


def plot_peds(peds, walls, i):
    """
    plot the cells. we need to make 'bad' walls to better visualize the cells
    """
    walls = walls * np.Inf
    P = np.vstack((walls,peds))
    P = np.vstack((P,walls))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.cla()
    cmap = plt.get_cmap("gray")
    cmap.set_bad(color = 'k', alpha = 0.8)
    im = ax.imshow(P, cmap=cmap, vmin=0, vmax=1,interpolation = 'nearest')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1 )
    plt.colorbar(im, cax=cax, ticks=[0, 1])
    ax.set_axis_off()
    N = sum(peds)
    S = "t: %3.3d | n: %d\n"%(i,N)
    plt.title("%20s"%S,rotation=0,fontsize=10,verticalalignment='bottom')
    plt.savefig("figs/peds%.5d.png"%i,dpi=100,facecolor='lightgray')
    
def print_logs(N_pedestrians, width,t, dt, nruns, Dt, vel, d):
    """
    print some infos to the screen
    """
    print ("Simulation space (%.2f x 1) m^2"%(width))
    print ("Mean Evacuation time: %.2f s, runs: %d"%((t*dt)/nruns, nruns))
    print ("Total Run time: %.2f s"%(Dt))
    print ("Factor: %.2f s"%( dt*t/Dt ) )
    print("--------------------------")
    print ("N %d   mean_velocity  %.2f [m/s]   density  %.2f [1/m]"%(N_pedestrians, vel, d))
    print("--------------------------")
    
def asep_parallel(cells):
    neighbors = np.roll(cells,-1)
    dim = len(cells)
    nmove = 0
    tmp_cells = np.zeros(dim)
    for i,j in  enumerate(cells):
        if j and not neighbors[i]:
            tmp_cells[i], tmp_cells[(i+1)%dim] = 0, 1
            nmove += 1
        elif j:
            tmp_cells[i] = 1
    return tmp_cells, nmove
    
if __name__ == "__main__":
    args = getParserArgs() # get arguments
    # init parameters
    N_pedestrians = args.np
    shuffle = args.shuffle
    reverse = args.reverse
    drawP = args.plotP
    #######################################################
    MAX_STEPS = 20 # simulation time
    nruns = args.nr  # repeate simulations, for TASEP 
    steps = range(MAX_STEPS)
    cellSize = 0.4 # m
    vmax= 1.2 # m/s
    dt = cellSize/vmax  # time step
    width = 10 # m
    n_cells = int(width/cellSize)  # number of cells
    if N_pedestrians >=n_cells:
        N_pedestrians = n_cells - 1
        print("warning: maximum of %d cells are allowed"%N_pedestrians)
    else:
        print("info: n_pedestrians=%d (max_pedestrians=%d)"%(N_pedestrians,n_cells))
    #######################################################
    t1 = time.time()
    tsim = 0
    density = float(N_pedestrians)/width
    walls = np.ones(n_cells)
    velocities = [] # velocities over all runs
    for n in range(nruns):
        peds = init_peds(N_pedestrians, n_cells)
        velocity = 0
        for t in steps: # simulation loop
            if drawP:
                plot_peds(peds, walls, t)
                
            peds, nmove = asep_parallel(peds)
            v = nmove*vmax/float(N_pedestrians)
            velocity += v
        
        velocity /= MAX_STEPS
        velocities.append(velocity)
        tsim += t
        print ("\trun: %3d ----  vel: %.2f |  den: %.2f"%(n, velocity,  density))
    t2 = time.time()
    mean_velocity = np.mean(velocities)
    print_logs(N_pedestrians, width, tsim, dt, nruns, t2-t1, mean_velocity, density)
    
    

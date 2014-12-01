import numpy as np
import matplotlib.pyplot as plt
import itertools  # for cartesian product
import time, random 
#######################################################
MAX_STEPS = 20
steps = range(MAX_STEPS)

cellSize = 0.4 # m
vmax= 1.2
dt = cellSize/vmax  # time step
width = 4 # m
height = .4 # m
dim_y = int(width/cellSize)  # number of columns, add ghost cells
dim_x = int(height/cellSize) # number of rows, add ghost cells

#DFF = np.ones( (dim_x, dim_y) ) # dynamic floor field
#######################################################
import logging, types, argparse

logfile='log.dat'
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def getParserArgs():
    parser = argparse.ArgumentParser(description='Cellular Automaton. Floor Field Model [Burstedde2001] Simulation of pedestrian dynamics using a two-dimensional cellular automaton Physica A, 2001, 295, 507-525')
    parser.add_argument("-s", "--ks", type=int , default=10, help='Sensitivity parameter for the  Static Floor Field (default 10)')
    parser.add_argument("-d", "--kd", type=int , default=0, help='Sensitivity parameter for the  Dynamic Floor Field (default 0)')
    parser.add_argument("-n", "--numPeds", type=int , default=10, help='Number of agents (default 10)')
    parser.add_argument("-p", "--plotS", type=int , default=0, help='Plot Static Floor Field (default 0)')
    parser.add_argument("-P", "--plotP", type=int , default=0, help='Plot Pedestrians (default 0)')
    parser.add_argument("-r", "--shuffle", type=int , default=0, help='1 for random shuffle (default 0)')
    parser.add_argument("-v", "--reverse", type=int , default=0, help='1 for reverse sequential update (default 0)')
    parser.add_argument("-a", "--periodic", type=int , default=0, help='1 periodic (ASEP) (default 0)')
    parser.add_argument("-l", "--log" , type=argparse.FileType('w'), default='log.dat', help="log file (default log.dat)")
    args = parser.parse_args()
    return args


def init_peds(N, n_cells):
    """
    distribute N pedestrians in box 
    """
    if N>dim_y: 
        N=dim_y
    PEDS = np.ones(N, int) # pedestrians    
    EMPTY_CELLS_in_BOX = np.zeros( n_cells - N, int) # the rest of cells in the box
    PEDS = np.hstack((PEDS,  EMPTY_CELLS_in_BOX)) # put 0s and 1s together
    np.random.shuffle(PEDS) # shuffle them
    return PEDS


def plot_peds(peds, i):
    #peds = np.hstack((peds,np.array([np.Inf])))
    #peds = np.hstack((np.array([np.Inf]),peds))
    walls = np.empty_like (peds) 
    walls[:] = np.Inf
    
    P = np.vstack((walls,peds))
    P = np.vstack((P,walls))
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.cla()
    N = np.sum(peds)
#    print peds
    cmap = plt.get_cmap("gray")
    cmap.set_bad(color = 'k', alpha = 0.8)

    ax.imshow(P, cmap=cmap,interpolation = 'nearest') # 1-peds because I want the peds to be black
    S = "t: %3.3d  |  N: %3.3d "%(i,N)
    plt.title("%6s"%S)    
    plt.savefig("figs/peds%.5d.png"%i)
    # print "figure: peds%.5d.png"%i

    
def print_logs(N_pedestrians, width,t, dt, nruns, Dt, vel, d):
    """
    print some infos to the screen
    """
    print ("Simulation of %d pedestrians"%N_pedestrians)
    print ("Simulation space (%.2f x 1) m^2"%(width))
    print ("Mean Evacuation time: %.2f s, runs: %d"%((t*dt)/nruns, nruns))
    print ("Total Run time: %.2f s"%(Dt))
    print ("Factor: %.2f s"%( dt*t/Dt ) )
    print("--------------------------\n")
    print ("velocity  %.2f   density  %.2f"%(vel, d))

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
    
    N_pedestrians = args.numPeds
    shuffle = args.shuffle
    reverse = args.reverse
    drawP = args.plotP
    nruns = 1  # repeate simulations 
    t1 = time.time()
    tsim = 0
    d =float(N_pedestrians)/width # density
    for n in range(nruns):
        peds = init_peds(N_pedestrians, dim_y)
        vel = 0 # velocity
        for t in steps: # simulation loop
            if drawP:
                plot_peds(peds, t)
                print ("\trun: %3d ----  t: %3d |  N: %3d"%(n,t, np.sum(peds)))
                
            peds, nmove = asep_parallel(peds)
            v = nmove*vmax/float(N_pedestrians)
            print nmove, v
            vel+= v
            
            
            if not peds.any(): # is everybody out?
                break
        vel /= len(steps)
        tsim += t
    t2 = time.time()
    print_logs(N_pedestrians, width, tsim, dt, nruns, t2-t1,vel,d)
    
    

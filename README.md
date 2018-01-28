# Purpose

Implementation of some known cellular automata. Use for academic purposes only.
Work in progress ...

# Simulation results

With the following parameter: 
- Width of room: 30 m
- Hight of the room: 30 m 
- Number of runs: 1
- Number of pedestrians 2000
- Diffusion parameter: 2
- Decay parameter: 0.2
- Static FF parameter: 2
- Dynamic FF parameter: 5

Call: 

```
python cellular_automaton.py -W 30 -H 30  -N 1 -n 2000 --diffusion 2 --plotAvgD  --plotD -d 5 -s 2 -P
```
- Video
[simulation](https://youtu.be/fD4l9P24J1k)

- Dynamics floor field (averaged over time)
![](figs/DFF-avg_S2.00_D5.00.png)

# Different neighborhoods

Call the script with the option `-moore` to use the moore neighborhood. Otherwise, von Neumann neighborhood will be used as default. 

The choice of the neighborhood has an influence on the evacuation time, as can seen below.

![evactime](./figs/moore_neumann_tevac.png)

## Moore neighborhood (video)

[![./figs/SFF_moore.png](http://img.youtube.com/vi/DAzu7GkUjHc/0.jpg)](https://youtu.be/DAzu7GkUjHc)

## von Neumann neighborhood (video)

[![./figs/SFF_neumann.png](http://img.youtube.com/vi/tnQegJcclu0/0.jpg)](https://youtu.be/tnQegJcclu0)

# Models

Two models are implemented: 

## Floor field model:
  - Different update schemes: sequential, shuffle sequential, reverse sequential.
  - visualisation of the cell states.
  - ~~todo~~: make a video from the png's
  - *todo*: track cells with _id_ for further trajectory analysis.
  - ~~todo~~: implement the dynamic floor field
  - *todo*: implement the parallel update.
  - todo: implement the conflict friction `mu`
  - *todo*: read geometry from a png file. See [read_png.py](geometry/read_png.py).
## ASEP model
  - the theoretical fundamental diagram can be reproduced, see [figure](figs/asep_fd.png). The size of the system should be reasonably high and the simulation time also.
  - *todo*: implement TASEP
  - *todo*: implement sequential update with all its variants.
  - Remarque: There are two implementations of the asep. One it optimized using vector-operations from `numpy` (`asep_fast.py`) and the other implementation is using explicit loops (`asep_slow.py`). The naming of the two variations is justified when measuring their execution time:
  ```
  python make_fd.py asep_fast.py:         0:56.71 real,   52.12 user,     4.03 sys
  python make_fd.py asep_slow.py:         1:15.42 real,   70.55 user,     4.23 sys
  ```



# How to use

run

```
python one_of_the_scripts -h
``` 

to see the options.

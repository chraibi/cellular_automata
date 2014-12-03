Implementation of some known cellular automata. Use for academic purposes only.
Work in progress ...

* models

1. Floor field model:
  - Different update schemes: sequential, shuffle sequential, reverse sequential.
  - visualisation of the cell states.
  - *todo*: make a video from the png's
  - *todo*: track cells with _id_ for further trajectory analysis.
  - *todo*: implement the dynamic floor field
  - *todo*: implement the parallel update.
  - *todo*: read geometry from a png file. See [read_png.py](geometry/read_png.py).
2. ASEP model
  - the theoretical fundamental diagram can be reproduced, see [figure](figs/asep_fd.png). The size of the system should be reasonably high and the simulation time also.
  - *todo*: implement TASEP
  - *todo*: implement sequential update with all its variants.

* How to use

run

```
python one_of_the_scripts -h
``` 

to see the options.

Simulation of a bottleneck:
\kappa_s=10, sequential update 
[Video](https://www.youtube.com/watch?v=xyU8jfzUxNg&feature=youtu.be)

Implementation of some known cellular automata. Use for academic purposes only.
Work in progress ...

See [Video](https://www.youtube.com/watch?v=xyU8jfzUxNg&feature=youtu.be).

* models

1. Floor field model:
  - Different update schemes: sequential, shuffle sequential, reverse sequential.
  - visualisation of the cell states.
  - ~~done~~: make a video from the png's
  - *todo*: track cells with _id_ for further trajectory analysis.
  - ~~done~~: implement the dynamic floor field
  - *todo*: implement the parallel update.
  - *todo*: read geometry from a png file. See [read_png.py](geometry/read_png.py).
2. ASEP model
  - the theoretical fundamental diagram can be reproduced, see [figure](figs/asep_fd.png). The size of the system should be reasonably high and the simulation time also.
  - *todo*: implement TASEP
  - *todo*: implement sequential update with all its variants.
  - Remarque: There are two implementations of the asep. One it optimized using vector-operations from `numpy` (`asep_fast.py`) and the other implementation is using explicit loops (`asep_slow.py`). The naming of the two variations is justified when measuring their execution time:
  ```
  python make_fd.py asep_fast.py:         0:56.71 real,   52.12 user,     4.03 sys
  python make_fd.py asep_slow.py:         1:15.42 real,   70.55 user,     4.23 sys
  ```



* How to use

run

```
python one_of_the_scripts -h
``` 

to see the options.

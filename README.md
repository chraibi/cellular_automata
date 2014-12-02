Implementation of some known cellular automata. Work in progress ...

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
  - the theoretical fundamental diagram can be reproduced, see [figure](asep_fd.png). The size of the system should be reasonably high and the simulation time also.
  - *todo*: implement TASEP
  - *todo*: implement sequential update with it all variants.

* How to use
run

```
python script -h
```

to see the options.


# Generation of the fundamental diagram by calling some model defined in this repository
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

import subprocess
import matplotlib.pyplot as plt
import os
# ----------------------------------------
num_runs = 10
max_pedestrians = 120
sim_steps = 1000
pedestrians = range(1, max_pedestrians)
filename = open("stdout.txt", "w")
# ----------------------------------------
for n in pedestrians:
    print("run asep.py with -n %3.3d -N %3.3d -m % 3.4d" % (n, num_runs, sim_steps))
    subprocess.call(["python", "asep.py", "-n" "%d" % n, "-N", "%d" % num_runs, "-m", "%d" % sim_steps],
                    stdout=filename)
# ----------------------------------------
filename.close()

velocities = []
densities = []
# the line should be something like this
# N 1   mean_velocity  1.20 [m/s]   density  0.10 [1/m]
filename = open("stdout.txt", "r")
for line in filename:
    if line.startswith("N"):
        line = line.split()
        velocities.append(float(line[3]))
        densities.append(float(line[6]))

filename.close()
# -------- plot FD ----------
# rho vs v
fig = plt.figure()
ax = fig.add_subplot(111)
ax.cla()
plt.subplot(211)
plt.plot(densities, velocities, lw=2)
plt.ylim([0, max(velocities)+0.05])
plt.ylabel(r"$v\, [m/s]$", size=20)
plt.xlabel(r"$\rho\, [m^{-1}]$", size=20)
# rho vs J (J=rho*v)
J = [r * v for (r, v) in zip(densities, velocities)]
plt.subplot(212)
plt.plot(densities, J, lw=2)
plt.xlabel(r"$\rho\, [m^{-1}]$", size=20)
plt.ylabel(r"$J\, [s^{-1}]$", size=20)
fig.tight_layout()
print("\n")

for end in ["pdf", "png", "eps"]:
    figure_name = os.path.join("figs", "asep_fd.%s" % end)
    print("result written in %s" % figure_name)
    plt.savefig(figure_name)
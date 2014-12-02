#! /usr/bin/env python
import subprocess
import matplotlib.pyplot as plt

#----------------------------------------
nruns = 30
maxpeds = 30
npeds = range(1, maxpeds)
stdout = open("stdout.txt","w")
#----------------------------------------
for n in npeds:
    print("run asep.py with -n %3.3d -N %3.3d"%(n,nruns))
    subprocess.call(["python", "asep.py", "-n" "%d"%n, "-N", "%d"%nruns], stdout=stdout)
#----------------------------------------
stdout.close()

velocities = []
densities = []
# the line should be something like this
# N 1   mean_velocity  1.20 [m/s]   density  0.10 [1/m]
stdout = open("stdout.txt","r")
for line in stdout:
    if line.startswith("N"):
        line = line.split()
        velocities.append(float(line[3]))
        densities.append(float(line[6]))

stdout.close()
# -------- plot FD ----------
# rho vs v
plt.subplot(211)
plt.plot(densities, velocities, lw=2)
plt.ylabel(r"$v\, [m/s]$", size=20)
plt.xlabel(r"$\rho\, [m^{-1}]$", size=20)
# rho vs J (J=rho*v)
J = [r*v for (r,v) in zip(densities, velocities)]
plt.subplot(212)
plt.plot(densities, J, lw=2)
plt.ylabel(r"$v\, [m/s]$", size=20)
plt.xlabel(r"$\rho\, [m^{-1}]$", size=20)



plt.savefig("asep_fd.png")

import numpy as np
import scipy
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt
import re

from constants_functions import *

dm2 = -0.001
theta = 0.0001
file_name = f"/home/projects/sterilenuosc/results/results_{theta}_{dm2}.txt"
dw_file_name = f"/home/projects/sterilenuosc/results/dw_{theta}_{dm2}.txt"

stacked = np.loadtxt(file_name)
P0s = stacked[0,:]
Pxs = stacked[1,:]
Pys = stacked[2,:]
Pzs = stacked[3,:]
Bxs = stacked[4,:]
Bys = stacked[5,:]
Bzs = stacked[6,:]
xs = stacked[7,:]
zs = stacked[8,:]

dw_fs = np.loadtxt(dw_file_name)
fs = dw_fs[0, np.arange(0, 100000, 10)]
fa = dw_fs[1, np.arange(0, 100000, 10)]

plt.figure()
plt.xlim(30,1)
plt.semilogx(1e-6*m_e/xs, (P0s-Pzs)/2, label = "QKE")
plt.semilogx(1e-6*m_e/xs, fs, label= "DW")
plt.xlabel("Temperature (MeV)")
plt.ylabel(r"f_s")
plt.title(f"QKE vs DW for dm2={dm2}, theta0={theta}")
plt.legend()
# plt.plot(xs, (P0s+Pzs)/2-1, color = "k")
fig_name = f"/home/projects/sterilenuosc/graph/dw_qke_comparison_{theta}_{dm2}.png"
plt.savefig(fig_name)


# check when cases are finished running
# try qke with dm2=-1e-3
# regraph with both dw and qke
# change x axis to temp
# if different, run qkes at 1eV^2
# multimomentum: y is already in code
# grid of 10 points (0.1-20)
# transition should happen at different times for each bin
# efficiency: write functions which calculate based on y_0, then rescale for each bin with necessary expression
# check that 1 momentum bin ends up being consistent with single momentum code
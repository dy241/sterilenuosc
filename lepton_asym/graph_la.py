import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from constants_functions import *

sstt = 1e-2 
theta0 = 0.5 * np.arcsin(np.sqrt(sstt))
dm2 = 1e-3
xi = 1e-2
init_LA = 1/(12*Apery) * (np.pi ** 2 * xi + xi ** 3)

T_start=4e7
T_end=4e5

file_name1 = "/home/projects/sterilenuosc/results/archive/alt_precession_5.01e-02_1.00e-03_LA6.84e-03_T4.00e+07-4.00e+05.txt"
file_name2 = "/home/projects/sterilenuosc/results/alt_precession_5.01e-02_1.00e-03_LA6.84e-03_T4.00e+05-3.80e+05.txt"
stacked1 = np.loadtxt(file_name1)
stacked2 = np.loadtxt(file_name2)
stacked = np.concatenate([stacked1.T, stacked2.T]).T

nbins = (len(stacked)-3) // 8
y_arr = np.linspace(0.1, 25, nbins)
f0 = f_0(y_arr, 0)

xs = stacked[0,:]
zs = stacked[-1,:]
LAs = stacked[-2, :]

plt.figure(figsize=(12, 6))

cmap = mpl.colormaps['turbo'] #[clr(i/n_temps) for i in range(n_temps)]
colors = cmap(np.linspace(0, 1, nbins))

for i in range(nbins):
    P0s = stacked[8*i+1,:]
    Pxs = stacked[8*i+2,:]
    Pys = stacked[8*i+3,:]
    Pzs = stacked[8*i+4,:]
    P0bars = stacked[8*i+5,:]
    Pxbars = stacked[8*i+6,:]
    Pybars = stacked[8*i+7,:]
    Pzbars = stacked[8*i+8,:]

    plt.xlim(4e-1, 3.8e-1)
    plt.ylim(-0.001, 0.004)
    # plt.semilogx(1e-6*m_e/xs, zs, color='tab:blue', linewidth=2)
    # plt.semilogx(1e-6*m_e/xs, (Pys-Pybars)/2, label=y_arr[i], color=colors[i],lw=1)
    # plt.semilogx(1e-6*m_e/xs, Pxbars, label=y_arr[i], color=colors[i],lw=1)
    # plt.semilogx(1e-6*m_e/xs, (P0s-Pzs)/2, label=y_arr[i], color=colors[i],lw=1)
    # plt.semilogx(1e-6*m_e/xs, (P0bars-Pzbars)/2, label=y_arr[i], color=colors[i],lw=1.5,linestyle='dashed')
    # plt.semilogx(1e-6*m_e/xs, (P0s-P0bars+Pzs-Pzbars)/2, label=y_arr[i], color=colors[i],lw=1)
    plt.semilogx(1e-6*m_e/xs, LAs, color='b', lw=1, linestyle='dashed')
# plt.legend(loc="best")
plt.xlabel("Temperature (MeV)")
plt.ylabel(r"f_s")
plt.title(r"$\Delta m^2 = 0.001 eV^2, sin^2(2\theta_0)=0.01$")

figname="/home/projects/sterilenuosc/la_asym_graph.png"
plt.savefig(figname)
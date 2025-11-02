import numpy as np
import scipy
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt
import re

from constants_functions import *

file_name = "/home/projects/sterilenuosc/results/factor1.0_precession_7.09e-02_2.00e-02.txt"
# file_name2 = "/home/projects/sterilenuosc/results/factor0.1_precession_7.09e-02_2.00e-01.txt"
# file_name3 = "/home/projects/sterilenuosc/results/factor0.01_precession_7.09e-02_2.00e-01.txt"
stacked = np.loadtxt(file_name)
# stacked2 = np.loadtxt(file_name2)
# stacked3 = np.loadtxt(file_name3)

nbins = (len(stacked)-2) // 4
bins = np.linspace(2, 25, nbins)
plt.figure(figsize=(8, 6))
plt.xlim([3e1, 1e0])
xs = stacked[0,:]
zs = stacked[-1,:]

dw_file = f"/home/projects/sterilenuosc/results/dw_mm_1.00e-02_1.00e-02.txt"
dw_file2 = f"/home/projects/sterilenuosc/results/dw_0.01_0.01.txt"
# dw_file3 = f"/home/projects/sterilenuosc/results/dw_rho_nu_error.txt"
dw_arr = np.loadtxt(dw_file)
dw_arr2 = np.loadtxt(dw_file2)
# dw_arr3 = np.loadtxt(dw_file3)
print(dw_arr.shape)
# nbins=(dw_arr.shape[0]-2)//2
# xs = dw_arr[-1,:]
# zs = dw_arr[-2,:]

colors = ['tab:gray', 'tab:brown', 'tab:red', 'tab:orange', 'tab:olive', 'tab:green', 'tab:cyan', 'tab:blue', 'tab:purple', 'tab:pink']

for i in [0]:#range(nbins):
    P0s = stacked[4*i+1,:]
    Pxs = stacked[4*i+2,:]
    Pys = stacked[4*i+3,:]
    Pzs = stacked[4*i+4,:]

    # P0s2 = stacked2[4*i+1,:]
    # Pzs2 = stacked2[4*i+4,:]
    # P0s3 = stacked3[4*i+1,:]
    # Pzs3 = stacked3[4*i+4,:]
    plt.xlim(3e1, 1e0)
    # plt.ylim(1, 1.002)
    # plt.semilogx(1e-6*m_e/xs, zs, color='tab:blue', linewidth=2)
    # plt.semilogx(1e-6*m_e/xs, (P0s-Pzs)/2, label=bins[i], color=colors[i],lw=1)
    # plt.semilogx(1e-6*m_e/xs, (P0s2-Pzs2)/2, label=bins[i], color=colors[i], linestyle='dashed', lw=2)
    # plt.semilogx(1e-6*m_e/xs, (P0s3-Pzs3)/2, label=bins[i], color=colors[i], linestyle='dotted',lw=2.5)
    plt.semilogx(1e-6*m_e/xs, dw_arr[i,:], linestyle='solid', color=colors[i+1], label='multi momentum DW')
    plt.semilogx(1e-6*m_e/xs, dw_arr2[0,:], linestyle='dashed', color=colors[i], label='old DW')
    # plt.semilogx(1e-6*m_e/xs, dw_arr[2,:], linestyle='dotted', color=colors[i], label='multi momentum DW')
    # plt.semilogx(1e-6*m_e/xs, dw_arr2, linestyle='dashed', color=colors[i], label='old DW')
    # plt.semilogx(1e-6*m_e/xs, dw_arr3[i,:], linestyle='solid', color=colors[i])
    # plt.plot(1e-6*m_e/xs, dw_arr[-2,:], color = "tab:red", linestyle='dotted', linewidth=4)

plt.legend(loc="best")
plt.xlabel("Temperature (MeV)")
plt.ylabel(r"f_s")
plt.title(r"$\Delta m^2 = 0.01 eV^2, \theta_0=0.01$")
fig_name = file_name.replace("txt","png")
fig_name = fig_name.replace("results", "graph")
fig_name = f"/home/projects/sterilenuosc/graph/dw_comparison.png"
plt.savefig(fig_name)


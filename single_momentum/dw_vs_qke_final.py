import numpy as np
from scipy.integrate import solve_ivp
from constants_functions import * 
import matplotlib.pyplot as plt

dm2s = [1e-1, 1e-2]
thetas = [7.9e-2, 1e-2, 1e-3, 1e-4]

final_fss = []

for dm2 in dm2s:
    for theta in thetas:
        file_name = "results/results_" + str(round(theta, 5)) + "_" + str(round(dm2, 5)) + ".txt"
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

        fs = (P0s - Pzs)/2
        l = len(fs)

        final_fss.append(np.mean(fs[l * 9 // 10:]))

fig, ax = plt.subplots()

ax.set_yscale('log')
ax.semilogx()
plt.scatter(thetas, final_fss[:4], marker = "+", color = "b")
plt.scatter(thetas, final_fss[4:], marker = "+", color = "k")

plt.savefig("graph/qke_final_graph.png")
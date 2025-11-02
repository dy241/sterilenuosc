import numpy as np
from scipy.integrate import solve_ivp
from constants_functions import * 
import matplotlib.pyplot as plt

dm2s = [-1e-1, -1e-2, -1e-3]
thetas = [7.9e-2, 1e-4]

final_fss = []

for dm2 in dm2s:
    for theta in thetas:
        file_name = f"/home/projects/sterilenuosc/results/results_{theta}_{dm2}.txt"
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

final_fss = np.array(final_fss)
final_fss = np.reshape(final_fss, (3, 2))

fig, ax = plt.subplots()
im = ax.imshow(np.log(final_fss), cmap='spring')

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(thetas)), labels=thetas)
ax.set_yticks(np.arange(len(dm2s)), labels=dm2s)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(dm2s)):
    for j in range(len(thetas)):
        text = ax.text(j, i, '{:.4e}'.format(final_fss[i, j]),
                       ha="center", va="center", color="k")

ax.set_title("Final fs")
fig.tight_layout()
plt.show()

plt.savefig("/home/projects/sterilenuosc/graph/2d_qke_hist.png")




final_fss = []
for dm2 in dm2s:
    for theta in thetas:
        file_name = f"/home/projects/sterilenuosc/results/dw_{theta}_{dm2}.txt"
        stacked = np.loadtxt(file_name)
        fs = stacked[0, :]
        fa = stacked[1, :]
        l=len(fs)
        final_fss.append(np.mean(fs[l * 9 // 10:]))

final_fss = np.array(final_fss)
final_fss = np.reshape(final_fss, (3, 2))

fig, ax = plt.subplots()
im = ax.imshow(np.log(final_fss), cmap='spring')

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(thetas)), labels=thetas)
ax.set_yticks(np.arange(len(dm2s)), labels=dm2s)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(dm2s)):
    for j in range(len(thetas)):
        text = ax.text(j, i, '{:.4e}'.format(final_fss[i, j]),
                       ha="center", va="center", color="k")

ax.set_title("Final fs")
fig.tight_layout()
plt.show()

plt.savefig("/home/projects/sterilenuosc/graph/2d_dw_hist.png")
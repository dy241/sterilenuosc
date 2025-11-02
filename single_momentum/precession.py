import numpy as np
import scipy
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt

from constants_functions import *

# bounds: https://arxiv.org/pdf/1112.3319.pdf
# https://iopscience.iop.org/article/10.1088/1742-6596/481/1/012005/pdf
#

dm2 = -1e-1 # 7.53e-5 # = delta m^2 # either m_21 or m_32 in normal order # m_21 normal
theta0 = 1e-4 # np.arcsin(np.sqrt(0.025)) * 0.5 = 0.079 # np.arcsin(np.sqrt(0.307)) # theta_12



y = 3.15
f0 = f_0(y)
z0 = [1.]

def xtoz(x): # https://stackoverflow.com/questions/16243955/numpy-first-occurrence-of-value-greater-than-existing-value
    i = min(np.searchsorted(xs, x), len(xs)-1)
    return zs[0, i]

def H(x):
    z = xtoz(x)
    rho = (2 * rho_e(x, z) + rho_nu() + rho_gamma(z)) * (m_e/x) ** 4
    a = np.sqrt(rho/3)
    return a / (m_Pl)

def l(x, z, H_x): # nondimensionalize 
    const = 2 * np.sqrt(2) * G_F / (x * H_x) * y * (m_e / x) ** 5 / (m_W ** 2)
    var = 2 * (rho_e(x, z) + P_e(x, z))
    return np.array([0, 0, -var * const])

def b(x): # constant (vacuum term)
    theta = theta0
    const = dm2 / (x * H(x)) * (x / m_e) / (2 * y)
    return const * np.array([np.sin(2*theta), 0, -np.cos(2*theta)])

def Gamma_e(x, y, z):
    return C_e * G_F ** 2 * y * (m_e/x) ** 5 * z ** 5

def deriv_p(x, P):
    z = xtoz(x)
    H_x = H(x)
    B = b(x)
    L = l(x, z, H_x)
    R = Gamma_e(x, y, z) * f0 * (1 - 0.5 * (P[0] + P[3]))
    D = 0.5 * Gamma_e(x, y, z)
    dP0 = np.array([1 / (x * H_x) * R / f0])
    dPvec = (cross(B + L, P[1:]) \
        + np.array([0, 0, 1 / (x * H_x) * R / f0]) \
        - np.array([1 / (x * H_x) * D * P[1], 1 / (x * H_x) * D * P[2], 0])) * factor
    return np.concatenate((dP0, dPvec))


Pzs = np.array([])
P0s = np.array([])

factor = 1
atol1, rtol1 = 1e-6, 1e-6
atol2, rtol2 = 1e-6, 1e-6
T_start = 3e7
T_end = 1e6
nxrange = 10000

start_time = time.time()

start, end = m_e/T_start, m_e/T_end

x_span = np.logspace(np.log10(start) + 1e-10, np.log10(end) - 1e-10, nxrange)

bunch = solve_ivp(dz_dx, [start, end], z0, t_eval = x_span, atol=atol1, rtol=rtol1) # , t_eval = x_span
xs = bunch["t"]
zs = bunch["y"]

#solving diffeq for polarization vectors
theta_i = 0
initial_state = [1, np.sin(theta_i), 0, np.cos(theta_i)]

bunch = scipy.integrate.solve_ivp(deriv_p, [xs[0], xs[-1]], initial_state, atol=atol2, rtol=rtol2, t_eval=xs) # tolerance: rtol, atol put at e-6
Ps = bunch["y"]

Bs = normalize(np.transpose(np.array([(b(xs[i]) + l(xs[i], zs[0,i], H(xs[i]))) for i in range(len(xs))])))
end_time = time.time()
print("factor:", factor, ", nx:", nxrange, ", solve_ivp:", end_time-start_time)

print("final P_z", np.mean(Ps[3, nxrange*9//10:-1]))
Pzs = np.append(Pzs, np.mean(Ps[3, nxrange*9//10:-1]))
print("final P_0", np.mean(Ps[0, nxrange*9//10:-1]))
P0s = np.append(P0s, np.mean(Ps[0, nxrange*9//10:-1]))
# plt.axvline(xs[nxrange*9//10], color = "k")
# plt.semilogx(xs, Ps[3,:])
# plt.show()

print(Ps.shape, Bs.shape, xs.shape, zs.shape)
stacked = np.stack([Ps[0,:], Ps[1,:], Ps[2,:], Ps[3,:], Bs[0,:], Bs[1,:], Bs[2,:], xs, zs[0,:]])

np.savetxt("/home/projects/sterilenuosc/results/fake_results_" + str(round(theta0,5)) + "_" + str(round(dm2,5)) + ".txt", stacked)

# plt.figure()
# plt.plot(factors, 1-Pzs)
# plt.plot(factors, P0s-1, color = "k")
# plt.show()

# plt.figure()
# plt.semilogx(factors, 1-Pzs)
# plt.semilogx(factors, P0s-1, color = "k")
# plt.show()

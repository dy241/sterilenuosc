# This is a multimomentum 1 active/1 sterile oscillation code that solves the quantum kinetic equations
# QKEs/other equations given by https://arxiv.org/pdf/1204.5861

import numpy as np
import scipy
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt

from constants_functions import *

# bounds: https://arxiv.org/pdf/1112.3319.pdf
# https://iopscience.iop.org/article/10.1088/1742-6596/481/1/012005/pdf


# diagnostic print
print("started")


# initialize parameters
Neffs = [] # can alter code to loop over arrays of parameters, track Neff for each case
dm2 = -1e-3
sstt = 1e-5
theta0 = 0.5 * np.arcsin(np.sqrt(sstt))
xi = 1e-2
init_LA = 1/(12*Apery) * (np.pi ** 2 * xi + xi ** 3)
ymin = 0.1
nbins = 20
z0 = [1.]

if nbins == 1: # linspace doesn't work with 1 bin
    y_arr = np.array([3.15])
else: # could also try alternate y binnings (see 3.1 of https://arxiv.org/pdf/1204.5861 or 2.2 of https://arxiv.org/pdf/1302.7279)
    y_arr = np.linspace(ymin, 25, nbins)

# equilibrium fermi-dirac distributions for no asym, then neutrinos and antineutrinos
f0 = f_0(y_arr, 0)
feq = f_0(y_arr, xi)
feqbar = f_0(y_arr, -xi)

# array to directly store dLA values from solve_ivp, could be useful for diagnostic
dLAs = []

# track how many calls to deriv_p are made
counter = 0




def H(x, P0_arr, P0bar_arr, z): # hubble factor
    rho = (2 * rho_e(x, z) + rho_nu(y_arr, P0_arr, P0bar_arr) + rho_gamma(z)) * (m_e/x) ** 4
    a = np.sqrt(rho/3)
    return a / (m_Pl)

# y is replaced by 1! be sure to multiply back at the end

def l(x, z, H_x): # matter term
    const = 2 * np.sqrt(2) * G_F / (x * H_x) * 1 * (m_e / x) ** 5 / (m_W ** 2)
    var = 2 * (rho_e(x, z) + P_e(x, z))
    return np.array([0, 0, -var * const])

def b(x, H_x): # constant (vacuum term)
    theta = theta0
    const = dm2 / (x * H_x) * (x / m_e) / (2 * 1)
    return const * np.array([np.sin(2*theta), 0, -np.cos(2*theta)])

def v_L(x, LA, H_x): # lepton sym term
    return np.array([0, 0, 2 * np.sqrt(2) * Apery / np.pi ** 2 * G_F * (m_e / x) ** 3 * LA / (x * H_x)])

def Gamma_e(x, z):
    return C_e * G_F ** 2 * 1 * (m_e/x) ** 5 * z ** 5

# factor = 0.01 # try [0.1, 0.01, 10]
# also try multiplying collision term by factor (lower factor reduces runtime)

def deriv_p(x, P): # P = [[P0, Pvec, P0bar, Pvecbar] * nbins, LA, z]
    global counter
    counter += 1
    if counter % 1000 == 0:
        print(f"counter: {counter}, T={1e-6*m_e/x}, LA={P[-2]}")

    dP = np.array([])
    # unnormalized in y
    z = P[-1]

    P0_arr = P[np.arange(0, nbins * 8, 8)]
    Px_arr = P[np.arange(1, nbins * 8, 8)]
    Py_arr = P[np.arange(2, nbins * 8, 8)]
    Pz_arr = P[np.arange(3, nbins * 8, 8)]
    P0bar_arr = P[np.arange(4, nbins * 8, 8)]
    Pxbar_arr = P[np.arange(5, nbins * 8, 8)]
    Pybar_arr = P[np.arange(6, nbins * 8, 8)]
    Pzbar_arr = P[np.arange(7, nbins * 8, 8)]

    Pvec_arr = np.stack([Px_arr, Py_arr, Pz_arr]).T
    Pvecbar_arr = np.stack([Pxbar_arr, Pybar_arr, Pzbar_arr]).T

    # following syms should be equivalent: first way is explicit calc, second way is from evolution

    # integrand = y_arr ** 2 * f0 * (P0_arr-P0bar_arr+Pz_arr-Pzbar_arr)
    # LA = inte(y_arr, integrand)
    LA = P[-2]

    # lepton asym should go to zero in normal hierarchy, and has small effect below 1e-4 or 1e-5 (1204.5861)
    # can directly set to 0 to avoid runtime spike
    # if np.abs(LA) < 1e-4:
    #     LA = 0

    H_x = H(x, P0_arr, P0bar_arr, z)

    B = b(x, H_x) * factor
    L = l(x, z, H_x) * factor
    V_L = v_L(x, LA, H_x)
    gamma_e = Gamma_e(x, z)

    R = gamma_e * y_arr * f0 * (feq/f0 - 0.5 * (P0_arr + Pz_arr)) 
    Rbar = gamma_e * y_arr * f0 * (feqbar/f0 - 0.5 * (P0bar_arr + Pzbar_arr)) 
    D = 0.5 * gamma_e * y_arr # * factor

    dP0s = np.array([1 / (x * H_x) * R / f0])
    dPvec = (np.cross(B[None, :]/y_arr[:, None] + L[None, :] * y_arr[:, None] + V_L[None, :] * np.ones_like(y_arr)[:, None], Pvec_arr) \
        + np.squeeze(np.stack([np.zeros_like(dP0s), np.zeros_like(dP0s), dP0s])).T \
        - np.stack([1 / (x * H_x) * D * Px_arr, 1 / (x * H_x) * D * Py_arr, np.zeros(nbins)]).T)
    dP0bars = np.array([1 / (x * H_x) * Rbar / f0])
    dPvecbar = (np.cross(B[None, :]/y_arr[:, None] + L[None, :] * y_arr[:, None] - V_L[None, :] * np.ones_like(y_arr)[:, None], Pvecbar_arr) \
        + np.squeeze(np.stack([np.zeros_like(dP0bars), np.zeros_like(dP0bars), dP0bars])).T \
        - np.stack([1 / (x * H_x) * D * Pxbar_arr, 1 / (x * H_x) * D * Pybar_arr, np.zeros(nbins)]).T)
    dP = np.concatenate((dP0s.T, dPvec, dP0bars.T, dPvecbar), axis=1)
    
    # don't need to calculate dLA if effect negligible (per above)

    # if np.abs(LA) < 1e-4:
    #     dLA = 0
    # else:
    integrand = B[0]/y_arr * y_arr ** 2 * f0 * (Py_arr - Pybar_arr)
    dLA = 1 / (8 * Apery) * inte(y_arr, integrand)

    global dLAs
    dLAs.append(np.array([x, dLA]))

    integrand = np.power(y_arr, 3) * f0 * dP0s.flatten()

    Jprime = J_prime(x/z)
    Kprime = K_prime(x/z)
    Jay = J(x/z)
    Kay = K(x/z)
    dz = (x/z * Jay + G1(x/z, Jprime, Kprime, Jay, Kay) - 1 / (2 *  np.pi ** 2 * z ** 3) * inte(y_arr, integrand)) \
    / (x/z * Jay + Y(x/z) + G2(x/z, Jprime, Kprime, Jay, Kay) + 2 * np.pi ** 2 / 15)
    
    dP = np.concatenate((dP.flatten(), np.array([dLA, dz])))
    return dP


Pzs = np.array([])
P0s = np.array([])

factor = 1e-0
atol, rtol = 1e-6, 1e-6

# can choose to run "multi-stage" by loading the final P of a past run, set load=True
load=False
#load initial state
if load:
    loadstate_name = "/home/projects/sterilenuosc/results/archive/alt_precession_5.01e-02_1.00e-03_LA6.84e-03_T4.00e+07-4.00e+05.txt"
    stacked = np.loadtxt(loadstate_name)
    initial_state = stacked[1:,-1]
    T_start = m_e/stacked[0,-1]
# otherwise initializes a fresh state with only active species in fermi-dirac equilibrium
else:
    initial_state = np.zeros((2*nbins*4+1+1))
    initial_state[0:8*nbins:8] = f_0(y_arr, xi) / f_0(y_arr, 0) # P0
    initial_state[3:8*nbins:8] = f_0(y_arr, xi) / f_0(y_arr, 0) # Pz
    initial_state[4:8*nbins:8] = f_0(y_arr, -xi) / f_0(y_arr, 0) # P0bar
    initial_state[7:8*nbins:8] = f_0(y_arr, -xi) / f_0(y_arr, 0) # Pzbar
    initial_state[-2] = init_LA
    initial_state[-1] = 1.
    T_start = 4e7 # starting temperature in eV


nxrange = int(1e5) # number of datapoints stored, log spaced in x
T_end = 1e5  # final temperature in eV

start, end = m_e/T_start, m_e/T_end

x_span = np.logspace(np.log10(start) + 1e-10, np.log10(end) - 1e-10, nxrange)

#solving diffeq for polarization vectors
print("solving")
start_time = time.time() # to track runtime
bunch = scipy.integrate.solve_ivp(deriv_p, [x_span[0], x_span[-1]], initial_state, atol=atol, rtol=rtol, t_eval=x_span, method="LSODA")
end_time = time.time()
print("factor:", factor, ", nx:", nxrange, ", solve_ivp:", end_time-start_time) # prints total runtime

Ps = bunch["y"]
xs = bunch["t"]
print(Ps.shape, xs[np.newaxis, :].shape)
stacked = np.concatenate([xs[np.newaxis, :], Ps])

np.savetxt(f"/home/projects/sterilenuosc/results/alt_precession_{'{:.2e}'.format(theta0)}_{'{:.2e}'.format(dm2)}_LA{'{:.2e}'.format(init_LA)}_T{'{:.2e}'.format(T_start)}-{'{:.2e}'.format(T_end)}.txt", stacked)
print(stacked.shape)

dLAs = np.array(dLAs)
np.savetxt(f"/home/projects/sterilenuosc/results/alt_dLAs_{'{:.2e}'.format(theta0)}_{'{:.2e}'.format(dm2)}_LA{'{:.2e}'.format(init_LA)}_T{'{:.2e}'.format(T_start)}-{'{:.2e}'.format(T_end)}.txt", dLAs)

# calculating Neff on the final state
PPbarf = stacked[1:-1,-1]
P0_arr = PPbarf[np.arange(0, nbins * 8, 8)]
Px_arr = PPbarf[np.arange(1, nbins * 8, 8)]
Py_arr = PPbarf[np.arange(2, nbins * 8, 8)]
Pz_arr = PPbarf[np.arange(3, nbins * 8, 8)]
P0bar_arr = PPbarf[np.arange(4, nbins * 8, 8)]
Pxbar_arr = PPbarf[np.arange(5, nbins * 8, 8)]
Pybar_arr = PPbarf[np.arange(6, nbins * 8, 8)]
Pzbar_arr = PPbarf[np.arange(7, nbins * 8, 8)]
Pf = np.stack((P0_arr, Px_arr, Py_arr, Pz_arr)).flatten()
Pfbar = np.stack((P0bar_arr, Pxbar_arr, Pybar_arr, Pzbar_arr)).flatten()
N = Neff(y_arr, Pf, Pfbar, f0)
Neffs.append(N)

print(str(dm2) + "_" + str(sstt) + "done")
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt

from constants_functions import *

# long term goals: make sure mm is working correctly, compare with dw, then add neutrino-antineutrino asym

# bounds: https://arxiv.org/pdf/1112.3319.pdf
# https://iopscience.iop.org/article/10.1088/1742-6596/481/1/012005/pdf

print("started")

Neffs = []
dm2 = 1e-0
sstt = 1e-1
xi = 0 #1.5e-2
init_LA = 1/(12*Apery) * (np.pi ** 2 * xi + xi ** 3)
# dm2s = np.logspace(-3, 1, 21) * -1
# sstts = np.logspace(-4, -1, 21)

dLAs = []

counter = 0
theta0 = 0.5 * np.arcsin(np.sqrt(sstt)) 

ymin = 0.1
nbins = 20 # try higher nbins for high momentums

if nbins == 1:
    y_arr = np.array([3.15])
else:
    y_arr = np.linspace(ymin, 25, nbins)
f0 = f_0(y_arr, 0)
feq = f_0(y_arr, xi)
feqbar = f_0(y_arr, -xi)
z0 = [1.]


def H(x, P0_arr, P0bar_arr, z):
    rho = (2 * rho_e(x, z) + rho_nu(y_arr, P0_arr, P0bar_arr) + rho_gamma(z)) * (m_e/x) ** 4
    a = np.sqrt(rho/3)
    return a / (m_Pl)

# y is replaced by 1! be sure to multiply back at the end

def l(x, z, H_x): # nondimensionalize 
    const = 2 * np.sqrt(2) * G_F / (x * H_x) * 1 * (m_e / x) ** 5 / (m_W ** 2)
    var = 2 * (rho_e(x, z) + P_e(x, z))
    return np.array([0, 0, -var * const])

def b(x, H_x): # constant (vacuum term)
    theta = theta0
    const = dm2 / (x * H_x) * (x / m_e) / (2 * 1)
    return const * np.array([np.sin(2*theta), 0, -np.cos(2*theta)])

def v_L(x, LA, H_x):
    return np.array([0, 0, 2 * np.sqrt(2) * Apery / np.pi ** 2 * G_F * (m_e / x) ** 3 * LA / (x * H_x)])

def Gamma_e(x, z):
    return C_e * G_F ** 2 * 1 * (m_e/x) ** 5 * z ** 5

# factor = 0.01 # try [0.1, 0.01, 10]
# also try multiplying collision term by factor

def deriv_p(x, P): # P = [[P0, Pvec, P0bar, Pvecbar] * nbins, LA, z]
    global counter
    counter += 1
    if counter % 10000 == 0:
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

    # integrand = y_arr ** 2 * f0 * (P0_arr-P0bar_arr+Pz_arr-Pzbar_arr)
    # LA = inte(y_arr, integrand)
    LA = P[-2]
    # if np.abs(LA) < 1e-4:
    #     LA = 0

    H_x = H(x, P0_arr, P0bar_arr, z)

    B = b(x, H_x) * factor
    L = l(x, z, H_x) * factor
    V_L = v_L(x, LA, H_x)
    gamma_e = Gamma_e(x, z)

    R = gamma_e * y_arr * f0 * (feq/f0 - 0.5 * (P0_arr + Pz_arr)) # replace 1 with feq/f0
    Rbar = gamma_e * y_arr * f0 * (feqbar/f0 - 0.5 * (P0bar_arr + Pzbar_arr)) # replace 1 with feqbar/f0
    D = 0.5 * gamma_e * y_arr # * factor

    dP0s = np.array([1 / (x * H_x) * R / f0]) # make plot of transition x vs y, compare with expectation analytically
    dPvec = (np.cross(B[None, :]/y_arr[:, None] + L[None, :] * y_arr[:, None] + V_L[None, :] * np.ones_like(y_arr)[:, None], Pvec_arr) \
        + np.squeeze(np.stack([np.zeros_like(dP0s), np.zeros_like(dP0s), dP0s])).T \
        - np.stack([1 / (x * H_x) * D * Px_arr, 1 / (x * H_x) * D * Py_arr, np.zeros(nbins)]).T)
    dP0bars = np.array([1 / (x * H_x) * Rbar / f0])
    dPvecbar = (np.cross(B[None, :]/y_arr[:, None] + L[None, :] * y_arr[:, None] - V_L[None, :] * np.ones_like(y_arr)[:, None], Pvecbar_arr) \
        + np.squeeze(np.stack([np.zeros_like(dP0bars), np.zeros_like(dP0bars), dP0bars])).T \
        - np.stack([1 / (x * H_x) * D * Pxbar_arr, 1 / (x * H_x) * D * Pybar_arr, np.zeros(nbins)]).T)
    dP = np.concatenate((dP0s.T, dPvec, dP0bars.T, dPvecbar), axis=1) # test shapes
    
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

load=False
#load initial state
if load:
    loadstate_name = "/home/projects/sterilenuosc/results/alt_precession_5.01e-02_1.00e-03_LA6.84e-03.txt"
    stacked = np.loadtxt(loadstate_name)
    initial_state = stacked[1:,-1]
    T_start = m_e/stacked[0,-1]

else:
    initial_state = np.zeros((2*nbins*4+1+1))
    initial_state[0:8*nbins:8] = f_0(y_arr, xi) / f_0(y_arr, 0) # P0
    initial_state[3:8*nbins:8] = f_0(y_arr, xi) / f_0(y_arr, 0) # Pz
    initial_state[4:8*nbins:8] = f_0(y_arr, -xi) / f_0(y_arr, 0) # P0bar
    initial_state[7:8*nbins:8] = f_0(y_arr, -xi) / f_0(y_arr, 0) # Pzbar
    initial_state[-2] = init_LA # calculate LA based on xi
    initial_state[-1] = 1.
    T_start = 4e7

def terminate(x, P):
    LA = P[-2]
    return np.abs(LA) - 1e-5
terminate.terminal = True

nxrange = int(1e5)
T_end = 1e5

start_time = time.time()

start, end = m_e/T_start, m_e/T_end

x_span = np.logspace(np.log10(start) + 1e-10, np.log10(end) - 1e-10, nxrange)

#solving diffeq for polarization vectors

print("solving")
bunch = scipy.integrate.solve_ivp(deriv_p, [x_span[0], x_span[-1]], initial_state, atol=atol, rtol=rtol, t_eval=x_span, method="Radau", events=terminate)
Ps = bunch["y"]

# Bs = normalize(np.transpose(np.array([(b(xs[i]) + l(xs[i], zs[0,i], H(xs[i]))) for i in range(len(xs))])))
end_time = time.time()
print("factor:", factor, ", nx:", nxrange, ", solve_ivp:", end_time-start_time)

xs = bunch["t"]
print(Ps.shape, xs[np.newaxis, :].shape)

# convert P to f for dw
fs = np.zeros((4*nbins+2, xs.size))

fs[-1,:] = Ps[-1,:]
fs[-2,:] = Ps[-2,:]

for i in range(nbins):
    fs[i,:] = (Ps[8*i,:] - Ps[8*i+3,:])/2 # fs
    fs[i+nbins,:] = (Ps[8*i+4,:] - Ps[8*i+7,:])/2 # fsbar
    fs[i+2*nbins,:] = (Ps[8*i,:] + Ps[8*i+3,:])/2 # f0
    fs[i+3*nbins,:] = (Ps[8*i+4,:] + Ps[8*i+7,:])/2 # f0bar

fstacked = np.concatenate([xs[np.newaxis, :], fs])
initial_state = fs[:,-1]

# start dw stage

start_time = time.time()

T_mid = m_e/xs[-1]
T_end = 1e5
nxrange = int(1e5)

print(f"cutoff T={'{:.2e}'.format(T_mid/1e6)}MeV")

start, end = m_e/T_mid, m_e/T_end

x_span = np.logspace(np.log10(start) + 1e-10, np.log10(end) - 1e-10, nxrange)

def deriv_f(x, f): # f = [fs] * nbins + [fsbar] * nbins + [fa] * nbins + [fabar] * nbins + [LA] (!) + [z]
    fs, fsbar, fa, fabar = f[:nbins], f[nbins:2*nbins], f[2*nbins:3*nbins], f[3*nbins:4*nbins]
    z = f[-1]
    Gamma_alpha = Gamma_e(x,z)

    P0 = fs+fa
    P0bar = fsbar+fabar
    hubble = H(x, P0, P0bar, z)

    R = Gamma_alpha * (1 - fa) * f0
    dP0 = R / x / hubble / f0

    Rbar = Gamma_alpha * (1 - fabar) * f0
    dP0bar = Rbar / x / hubble / f0
    
    numerator = Gamma_alpha * (x * dm2 / m_e / 2 / y_arr) ** 2 * np.sin(2 * theta0) ** 2 * (fa - fs)
    denominator = 4 * x * hubble * ((x * dm2 / m_e / 2 / y_arr) ** 2 * np.sin(2 * theta0) ** 2 \
    + Gamma_alpha ** 2 / 4 \
    + (x/m_e*dm2/2/y_arr*np.cos(2*theta0) + (m_e/x) ** 5 * 2 * np.sqrt(2) * G_F * y_arr / (m_W ** 2) * 2 * (rho_e(x, z) + P_e(x, z))) ** 2)
    dfs = numerator/denominator

    numerator = Gamma_alpha * (x * dm2 / m_e / 2 / y_arr) ** 2 * np.sin(2 * theta0) ** 2 * (fabar - fsbar)
    denominator = 4 * x * hubble * ((x * dm2 / m_e / 2 / y_arr) ** 2 * np.sin(2 * theta0) ** 2 \
    + Gamma_alpha ** 2 / 4 \
    + (x/m_e*dm2/2/y_arr*np.cos(2*theta0) + (m_e/x) ** 5 * 2 * np.sqrt(2) * G_F * y_arr / (m_W ** 2) * 2 * (rho_e(x, z) + P_e(x, z))) ** 2)
    dfbars = numerator/denominator

    integrand = np.power(y_arr, 3) * f0 * (dP0.flatten() + dP0bar.flatten())

    Jprime = J_prime(x/z)
    Kprime = K_prime(x/z)
    Jay = J(x/z)
    Kay = K(x/z)

    dz = (x/z * Jay + G1(x/z, Jprime, Kprime, Jay, Kay) - 1 / (2 *  np.pi ** 2 * z ** 3) * inte(y_arr, integrand)) \
    / (x/z * Jay + Y(x/z) + G2(x/z, Jprime, Kprime, Jay, Kay) + 2 * np.pi ** 2 / 15)  # check if need factor of 1/2 in dz

    return np.concatenate([dfs, dfbars, 2*dP0-dfs, 2*dP0bar-dfbars, [0], [dz]])

atol, rtol = 1e-6, 1e-6

bunch = scipy.integrate.solve_ivp(deriv_f, t_span=[x_span[0], x_span[-1]], y0=initial_state, atol=atol, rtol=rtol, t_eval=x_span, method="Radau")

fs = bunch["y"]



# concatenate precession and dw results
dwstacked = np.concatenate([bunch['t'][np.newaxis, :], fs])
print(dwstacked.shape)
stacked = np.concatenate([fstacked, dwstacked], axis=1)
filepath = f"/home/projects/sterilenuosc/results/precession+dw_{'{:.2e}'.format(theta0)}_{'{:.2e}'.format(dm2)}_LA{'{:.2e}'.format(init_LA)}_T{'{:.2e}'.format(T_start)}-{'{:.2e}'.format(T_end)}.txt"
np.savetxt(filepath, stacked)
print(f"nbins={nbins}, {time.time() - start_time}")
print(f"{'{:.2e}'.format(theta0)}_{'{:.2e}'.format(dm2)}")
Pf = np.array([[fs[nbins+i,-1]+fs[i,-1], 0, 0, fs[nbins+i,-1]-fs[i,-1]] for i in range(nbins)]).flatten()
# N = Neff(y_arr, Pf, f0)
# Neffs.append(N)


























# np.savetxt(f"/home/projects/sterilenuosc/results/alt_precession_{'{:.2e}'.format(theta0)}_{'{:.2e}'.format(dm2)}_LA{'{:.2e}'.format(init_LA)}_T{'{:.2e}'.format(T_start)}-{'{:.2e}'.format(T_end)}.txt", stacked)
# print(stacked.shape)

# dLAs = np.array(dLAs)
# np.savetxt(f"/home/projects/sterilenuosc/results/alt_dLAs_{'{:.2e}'.format(theta0)}_{'{:.2e}'.format(dm2)}_LA{'{:.2e}'.format(init_LA)}_T{'{:.2e}'.format(T_start)}-{'{:.2e}'.format(T_end)}.txt", dLAs)

# PPbarf = stacked[1:-1,-1]
# P0_arr = PPbarf[np.arange(0, nbins * 8, 8)]
# Px_arr = PPbarf[np.arange(1, nbins * 8, 8)]
# Py_arr = PPbarf[np.arange(2, nbins * 8, 8)]
# Pz_arr = PPbarf[np.arange(3, nbins * 8, 8)]
# P0bar_arr = PPbarf[np.arange(4, nbins * 8, 8)]
# Pxbar_arr = PPbarf[np.arange(5, nbins * 8, 8)]
# Pybar_arr = PPbarf[np.arange(6, nbins * 8, 8)]
# Pzbar_arr = PPbarf[np.arange(7, nbins * 8, 8)]
# Pf = np.stack((P0_arr, Px_arr, Py_arr, Pz_arr)).flatten()
# Pfbar = np.stack((P0bar_arr, Pxbar_arr, Pybar_arr, Pzbar_arr)).flatten()
# N = Neff(y_arr, Pf, Pfbar, f0)
# Neffs.append(N)

# print(str(dm2) + "_" + str(sstt) + "done")
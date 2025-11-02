# Dodelson-Widrow code, not adapted to include lepton asymmetry

import numpy as np
import scipy
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt

from constants_functions import *

# dm2 = 1e-0
# theta0 = 5e-3

fname = f"/home/projects/sterilenuosc/results/posNeffs_dw.txt"
Neffs = []

# dm2s = [1e-0]
# sstts = [1e-2]

dm2s = np.logspace(-3, 1, 21)
sstts = np.logspace(-4, -1, 21)

for dm2 in dm2s:
    for sinsqtwoth in sstts:

        # dm2 = 2e-2 # 7.53e-5 # = delta m^2 # either m_21 or m_32 in normal order # m_21 normal
        # theta0 = 0.5 * np.arcsin(np.sqrt(sinsqtwoth)) # np.arcsin(np.sqrt(0.025)) * 0.5 = 0.079 # np.arcsin(np.sqrt(0.307)) # theta_12

        start_time = time.time()
        theta0 = 0.5 * np.arcsin(np.sqrt(sinsqtwoth))

        # try different values for dm2 and theta0 to see if final f_s goes higher

        nbins = 20
        linear = True

        ymin = 1e-1
        ymax = 25
        yext = 3.1
        k1 = ymin/yext
        k2 = ymax/yext
        e2 = (1 + k1) / (k2 - k1)
        e1 = k1 * (1 + e2)

        def yspace(ymin, ymax, nbins):
            yext = 3.1
            k1 = ymin/yext
            k2 = ymax/yext
            e2 = (1 + k1) / (k2 - k1)
            e1 = k1 * (1 + e2)
            u = np.linspace(0, 1, nbins)
            return yext * (u + e1) / (1 + e2 - u)

        if nbins == 1:
            y_arr = np.array([3.15])
        elif linear:
            y_arr = np.linspace(ymin, ymax, nbins) # linspace is ok, better for integrals since constant width
        else:
            y_arr = yspace(ymin, ymax, nbins)
        z0 = [1]

        T_start = 4e7
        T_end = 1e5
        nxrange = int(1e4)

        start, end = m_e/T_start, m_e/T_end

        x_span = np.logspace(np.log10(start) + 1e-10, np.log10(end) - 1e-10, nxrange)

        # bunch = solve_ivp(dz_dx, [start, end], z0, t_eval = x_span, atol=atol1, rtol=rtol1, method="Radau") # , t_eval = x_span
        # xs = bunch["t"]
        # zs = bunch["y"]
        # zs = zs.flatten()

        def Gamma_e(x, y_arr, z): # recheck using same dz/dx, make sure agrees with y normalization
            # return C_e * G_F ** 2 * 3.15 * (m_e * z / x) ** 5
            return C_e * G_F ** 2 * y_arr * (m_e * z / x) ** 5

        def H(x, f): # rewrite to have one shared function with precession_mm
            fs, fa = f[:nbins], f[nbins:-1]
            # fs, fa = f[:-1:2], f[1::2]
            z = f[-1]
            rho = (2 * rho_e(x, z) + rho_nu(y_arr, (fa+fs)/2) + rho_gamma(z)) * (m_e/x) ** 4 # final result changes from 8e-3 to 8.7e-3
            # try different grids to see which one gets same evolution
            # we have to know how many bins we need for rho_nu to be good
            # rho_nu(y_arr, 1) should give 3 * 7/120 * np.pi ** 2
            # check to see how small differences in rho_nu affects evolution
            # should probably need low momentum bins (including range infeasiable for qke)
            # rho = (2 * rho_e(x, z) + 3 * 7/120 * np.pi ** 2 + rho_gamma(z)) * (m_e/x) ** 4 # incorrect factor of 2 removed
            a = np.sqrt(rho/3)
            return a / (m_Pl)

        f_0_arr = 1 / (np.exp(y_arr) + 1)

        time1 = 0
        time2 = 0
        time3 = 0
        time4 = 0
        time5 = 0

        def deriv_f(x, f): # f = [fs] * nbins + [f0] * nbins + [z]
            global time1
            global time2
            global time3
            global time4
            global time5

            st = time.time()
            fs, fa = f[:nbins], f[nbins:-1]
            z = f[-1]
            Gamma_alpha = Gamma_e(x,y_arr,z)
            time1 += time.time()-st
            st = time.time()
            hubble = H(x, f)
            time2 += time.time()-st
            st = time.time()

            R = Gamma_alpha * f_0_arr * (1 - fa)
            dP0 = R / f_0_arr / x / hubble
            
            numerator = Gamma_alpha * (x * dm2 / m_e / 2 / y_arr) ** 2 * np.sin(2 * theta0) ** 2 * (fa - fs)
            denominator = 4 * x * hubble * ((x * dm2 / m_e / 2 / y_arr) ** 2 * np.sin(2 * theta0) ** 2 \
            + Gamma_alpha ** 2 / 4 \
            + (x/m_e*dm2/2/y_arr*np.cos(2*theta0) + (m_e/x) ** 5 * 2 * np.sqrt(2) * G_F * y_arr / (m_W ** 2) * 2 * (rho_e(x, z) + P_e(x, z))) ** 2)
            dfs = numerator/denominator
            integrand = np.power(y_arr, 3) * f_0_arr * dP0.flatten()
            time3 += time.time()-st
            st = time.time()
            Jprime = J_prime(x/z)
            Kprime = K_prime(x/z)
            Jay = J(x/z)
            Kay = K(x/z)
            time4 += time.time()-st
            st = time.time()
            dz = (x/z * Jay + G1(x/z, Jprime, Kprime, Jay, Kay) - 1 / (2 *  np.pi ** 2 * z ** 3) * inte(y_arr, integrand)) \
            / (x/z * Jay + Y(x/z) + G2(x/z, Jprime, Kprime, Jay, Kay) + 2 * np.pi ** 2 / 15) # replace with old dz/dx, should get identical result as before format change 
            # dz = dz_dx(x, z)
            time5 += time.time()-st
            st = time.time()
            return np.concatenate([dfs, 2*dP0-dfs, [dz]])

        atol, rtol = 1e-8, 1e-8

        initial_state = np.array([0] * nbins + [1] * nbins + [1])
        bunch = scipy.integrate.solve_ivp(deriv_f, t_span=[x_span[0], x_span[-1]], y0=initial_state, atol=atol, rtol=rtol, t_eval=x_span, method="Radau")

        fs = bunch["y"]
        stacked = np.concatenate([fs, x_span[None,:]])
        filepath = f"/home/projects/sterilenuosc/results/dw_Radau_n{nbins}_{ymin}_{ymax}_{'{:.2e}'.format(theta0)}_{'{:.2e}'.format(dm2)}.txt"
        # np.savetxt(filepath, stacked)
        print(f"nbins={nbins}, linear={linear}, {time.time() - start_time}")
        print(f"{'{:.2e}'.format(theta0)}_{'{:.2e}'.format(dm2)}")
        print(time1+time2+time3+time4+time5)
        Pf = np.array([[fs[nbins+i,-1]+fs[i,-1], 0, 0, fs[nbins+i,-1]-fs[i,-1]] for i in range(nbins)]).flatten()
        N = Neff(y_arr, Pf, f_0_arr)
        Neffs.append(N)

Neffs = np.array(Neffs)
np.savetxt(fname, Neffs)
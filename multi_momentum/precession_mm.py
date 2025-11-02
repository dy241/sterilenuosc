import numpy as np
import scipy
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt

from constants_functions import *

# long term goals: make sure mm is working correctly, compare with dw, then add neutrino-antineutrino asym

# bounds: https://arxiv.org/pdf/1112.3319.pdf
# https://iopscience.iop.org/article/10.1088/1742-6596/481/1/012005/pdf

# write function for Neff

fname = "/home/projects/sterilenuosc/results/negNeffs_precession"
print("started")

Neffs = []
dm2s = [1e-3] 
sstts = [1e-2] 
# dm2s = np.logspace(-3, 1, 21) * -1
# sstts = np.logspace(-4, -1, 21)

for dm2 in dm2s:
    for sstt in sstts:

        counter = 0
        # dm2 = 1e-3 # try negative dm2
        # sstt = 1e-2
        # theta0 = 0.5 * np.arcsin(np.sqrt(0.02)) 
        theta0 = 0.5 * np.arcsin(np.sqrt(sstt)) 

        ymin = 0.1
        nbins = 20 # try higher nbins for high momentums
        binning = 0

        if nbins == 1:
            y_arr = np.array([3.15])
        elif binning:
            y_arr = np.concatenate([[ymin, 1], np.linspace(2, 25, 8)])
        else:
            y_arr = np.linspace(ymin, 25, nbins) # start with 0.8
        f0 = f_0(y_arr)
        z0 = [1.]

        # def xtoz(x): # https://stackoverflow.com/questions/16243955/numpy-first-occurrence-of-value-greater-than-existing-value
        #     i = min(np.searchsorted(xs, x), len(xs)-1)
        #     return zs[0, i]

        def H(x, P0_arr, z):
            rho = (2 * rho_e(x, z) + rho_nu(y_arr, P0_arr) + rho_gamma(z)) * (m_e/x) ** 4
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

        def Gamma_e(x, z):
            return C_e * G_F ** 2 * 1 * (m_e/x) ** 5 * z ** 5

        # factor = 0.01 # try [0.1, 0.01, 10]
        # also try multiplying collision term by factor
        def deriv_p_old(x, P):
            dP = np.array([])
            # unnormalized in y
            z = P[-1]

            P0_arr = P[np.arange(0, nbins * 4, 4)]
            H_x = H(x, P0_arr, z)
            B = b(x, H_x) * factor
            L = l(x, z, H_x) * factor
            gamma_e = Gamma_e(x, z)
            # np.reshape P array into (nbins, 4)
            # rewrite cross product to take in array of vectors
            dP0s = []
            for i in range(nbins): # write without for loop
                # time difference between for loop and np array ized version
                y = y_arr[i]
                R = gamma_e*y * f0[i] * (1 - 0.5 * (P[4*i] + P[4*i+3])) # * factor
                D = 0.5 * gamma_e*y # * factor
                dP0 = np.array([1 / (x * H_x) * R / f0[i]]) # make plot of transition x vs y, compare with expectation analytically
                dPvec = (cross(B/y + L*y, P[4*i+1:4*i+4]) \
                    + np.array([0, 0, dP0[0]]) \
                    - np.array([1 / (x * H_x) * D * P[4*i+1], 1 / (x * H_x) * D * P[4*i+2], 0])) 
                # B = B[None,:], y_arr = y_arr[:, None]
                # returns a (20, 3) array
                dPi = np.concatenate((dP0, dPvec))
                dP = np.concatenate((dP, dPi))
                dP0s.append(dP0)
            dP0s = np.array(dP0s)

            integrand = np.power(y_arr, 3) * f0 * dP0s.flatten()
            # add booleans to e.g. choose which version of dzdx to use
            # dz = dz_dx(x, z)
            # dz = dz_dx_QED(x, z)
            Jay = J(x/z)
            Jprime = J_prime(x/z)
            Kay = K(x/z)
            Kprime = K_prime(x/z)
            dz = (x/z * Jay + G1(x/z, Jprime, Kprime, Jay, Kay) - 1 / (2 *  np.pi ** 2 * z ** 3) * inte(y_arr, integrand)) \
            / (x/z * Jay + Y(x/z) + G2(x/z, Jprime, Kprime, Jay, Kay) + 2 * np.pi ** 2 / 15) # replace with old dz/dx, should get identical result as before format change 
            dP = np.append(dP, dz)
            return dP

        # time functions and add throughout loops
        time1 = 0
        time2 = 0
        time3 = 0
        time4 = 0
        time5 = 0
        time6 = 0
        def deriv_p(x, P):
            global counter
            counter += 1
            if counter % 10000 == 0:
                print(f"counter: {counter}, T={1e-6*m_e/x}")
            global time1
            global time2
            global time3
            global time4
            global time5
            global time6
            st = time.time()

            dP = np.array([])
            # unnormalized in y
            z = P[-1]

            P0_arr = P[np.arange(0, nbins * 4, 4)]
            Px_arr = P[np.arange(1, nbins * 4, 4)]
            Py_arr = P[np.arange(2, nbins * 4, 4)]
            Pz_arr = P[np.arange(3, nbins * 4, 4)]
            Pvec_arr = np.stack([Px_arr, Py_arr, Pz_arr]).T
            time1 += time.time()-st
            st = time.time()

            H_x = H(x, P0_arr, z)
            time2 += time.time()-st
            st = time.time()

            B = b(x, H_x) * factor
            L = l(x, z, H_x) * factor
            gamma_e = Gamma_e(x, z)
            time3 += time.time()-st
            st = time.time()

            R = gamma_e * y_arr * f0 * (1 - 0.5 * (P0_arr + Pz_arr)) # * factor
            D = 0.5 * gamma_e * y_arr # * factor
            time4 += time.time()-st
            st = time.time()

            dP0s = np.array([1 / (x * H_x) * R / f0]) # make plot of transition x vs y, compare with expectation analytically
            dPvec = (np.cross(B[None, :]/y_arr[:, None] + L[None, :] * y_arr[:, None], Pvec_arr) \
                + np.squeeze(np.stack([np.zeros_like(dP0s), np.zeros_like(dP0s), dP0s])).T \
                - np.stack([1 / (x * H_x) * D * Px_arr, 1 / (x * H_x) * D * Py_arr, np.zeros(nbins)]).T)    
            dP = np.concatenate((dP0s.T, dPvec), axis=1)
            integrand = np.power(y_arr, 3) * f0 * dP0s.flatten()
            time5 += time.time()-st
            st = time.time()

            Jprime = J_prime(x/z)
            Kprime = K_prime(x/z)
            Jay = J(x/z)
            Kay = K(x/z)
            dz = (x/z * Jay + G1(x/z, Jprime, Kprime, Jay, Kay) - 1 / (2 *  np.pi ** 2 * z ** 3) * inte(y_arr, integrand)) \
            / (x/z * Jay + Y(x/z) + G2(x/z, Jprime, Kprime, Jay, Kay) + 2 * np.pi ** 2 / 15) # replace with old dz/dx, should get identical result as before format change 
            # dz = dz_dx(x, z)
            time6 += time.time()-st
            
            dP = np.append(dP, dz)
            return dP


        Pzs = np.array([])
        P0s = np.array([])

        factor = 1e-0 # plot value of B/y and L*y for y=3.15 and R at different factors
        # when only one time scale, code should be a lot faster
        atol, rtol = 1e-8, 1e-8 # try changing tol values and see how much time changes

        theta_i = 0
        initial_state = [1, np.sin(theta_i), 0, np.cos(theta_i)] * nbins + [1]

        nxrange = int(1e6)
        T_start = 4e7
        T_end = 1e6

        start_time = time.time()

        start, end = m_e/T_start, m_e/T_end

        x_span = np.logspace(np.log10(start) + 1e-10, np.log10(end) - 1e-10, nxrange)

        # bunch = solve_ivp(dz_dx, [start, end], z0, t_eval = x_span, atol=atol1, rtol=rtol1) # , t_eval = x_span
        # xs = bunch["t"]
        # zs = bunch["y"]

        #solving diffeq for polarization vectors

        print("solving")
        bunch = scipy.integrate.solve_ivp(deriv_p, [x_span[0], x_span[-1]], initial_state, atol=atol, rtol=rtol, t_eval=x_span, method="Radau")
        Ps = bunch["y"]

        # Bs = normalize(np.transpose(np.array([(b(xs[i]) + l(xs[i], zs[0,i], H(xs[i]))) for i in range(len(xs))])))
        end_time = time.time()
        print("factor:", factor, ", nx:", nxrange, ", solve_ivp:", end_time-start_time)
        print(f"time1:{time1}, time2:{time2}, time3:{time3}, time4:{time4}, time5:{time5}, time6:{time6}")

        print("final P_z", np.mean(Ps[3, nxrange*9//10:-1]))
        Pzs = np.append(Pzs, np.mean(Ps[3, nxrange*9//10:-1]))
        print("final P_0", np.mean(Ps[0, nxrange*9//10:-1]))
        P0s = np.append(P0s, np.mean(Ps[0, nxrange*9//10:-1]))
        # plt.axvline(xs[nxrange*9//10], color = "k")
        # plt.semilogx(xs, Ps[3,:])
        # plt.show()

        print(Ps.shape, x_span[np.newaxis, :].shape)
        stacked = np.concatenate([x_span[np.newaxis, :], Ps])

        np.savetxt(f"/home/projects/sterilenuosc/results/precession_binning{binning}_{'{:.2e}'.format(theta0)}_{'{:.2e}'.format(dm2)}_stage{'{:.2e}'.format(T_end)}.txt", stacked)
        
        # np.savetxt(f"/home/projects/sterilenuosc/time_for_T={'{:.2e}'.format(T_end)}_{'{:.2e}'.format(theta0)}_{'{:.2e}'.format(dm2)}.txt", np.array([end_time - start_time]))

        Pf = stacked[1:-1,-1]
        N = Neff(y_arr, Pf, f0)
        Neffs.append(N)

        print(str(dm2) + "_" + str(sstt) + "done")

Neffs = np.array(Neffs)
# np.savetxt(fname, Neffs)

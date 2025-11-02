import numpy as np
from scipy.integrate import quad
from numba import jit

# https://pdg.lbl.gov/2023/reviews/rpp2023-rev-phys-constants.pdf
# constants

m_e = 0.51099895e6
m_W = 80.377e9
m_Pl = 1.220890e28/np.sqrt(8 * np.pi)
G_F = 1.1663788e-23
C_e = 1.27
C_mu = 0.92
C_tau = 0.92
Gamma = 0
alpha = 1/137
Apery = 1.202057




# do integrals analytically, then compare to result for different #bins, range

# astrophysical/thermo

# rho_nu = lambda: 3 * 7/120 * np.pi ** 2 # needs to be an integral over multiple momentum bins when those are added

def rho_nu(y_arr, P0_arr, P0bar_arr):
    # energy density of neutrinos and antineutrinos with 2 flavors that do not mix and 1 active-sterile mixing doublet
    f_arr = np.power(y_arr, 3) * P0_arr * f_0(y_arr, 0)
    fbar_arr = np.power(y_arr, 3) * P0bar_arr * f_0(y_arr, 0) # f_0 needs to be different?
    integral = inte(y_arr, f_arr) # should the integral be small compared to non-mixing flavor term?
    integralbar = inte(y_arr, fbar_arr)
    return 2 * (1/2 / (2 * np.pi ** 2) * (integral + integralbar) + 7 / 120 * np.pi ** 2) # add factor of 1/2, integral + integral anti # integral/2pi**2 term should be equal to 7/240 pi ** 2 at start

def rho_gamma(z):
    return np.pi ** 2 / 15 * z ** 4

@jit(nopython=True, cache=True)
def intrhoe(var, x_z):
    a = var ** 2 * np.sqrt(var ** 2 + (x_z) ** 2) / (np.exp(np.sqrt(var ** 2 + (x_z) ** 2)) + 1)
    return a

def rho_e(x, z):
    # energy density of electrons only (no positrons)
    # hence factor of 2 in hubble rate (need electrons and positrons)
    result, err = quad(intrhoe, 0, 100, args=(x/z))
    return 1 / np.pi ** 2 * result * z ** 4

@jit(nopython=True, cache=True)
def intPe(var, x_z):
    a = var ** 4 / (3 * np.sqrt(var ** 2 + (x_z) ** 2) * (np.exp(np.sqrt(var ** 2 + (x_z) ** 2)) + 1))
    return a

def P_e(x, z):
    result, err = quad(intPe, 0, 50, args=(x/z))
    return 1 / np.pi ** 2 * result * z ** 4

def dz_dx(x, z): # must include extra terms for collisions
    num = x/z * J(x/z) # + G1 find final value of z (should be around 1.397...)
    denom = x**2 / z**2 * J(x/z) + Y(x/z) + 2 * np.pi**2 / 15 # + G2
    return num/denom

def dz_dx_QED(x, z): # must include extra terms for collisions
    Jay = J(x/z)
    Kay = K(x/z)
    Jprime = J_prime(x/z)
    Kprime = K_prime(x/z)
    num = x/z * Jay + G1(x/z, Jprime, Kprime, Jay, Kay) # find final value of z (should be around 1.397...)
    denom = x**2 / z**2 * Jay + Y(x/z) + 2 * np.pi**2 / 15 + G2(x/z, Jprime, Kprime, Jay, Kay)
    return num/denom

def Neff(y_arr, P_arr, Pbar_arr, f0): # P_arr is just P for each y
    # for P = [2, 0, 0, 0] should give 1
    # for P = [1, ?, ?, ?] should give 0
    integrand = f0 * np.power(y_arr, 3) * ((P_arr[0::4] + Pbar_arr[0::4])/2 - 1)
    # return 120 / (7 * np.pi ** 4) * inte(y_arr, integrand)
    n = f0 * np.power(y_arr, 3) * (np.ones_like(P_arr[0::4]) * 2 - 1)
    return 1/inte(y_arr, n) * inte(y_arr, integrand)






# math functions

def f_0(y, xi):
    return 1/(np.exp(y-xi) + 1)

# rewrite G1, G2 to reduce number of integrals
# interpolate between known t to avoid integrals
# interpolate as function of log(tau)
# calculate difference between interpolation and true for finer grid

@jit(nopython=True, cache=True)
def intJ(var, t): # var = p/T_gamma for an electron = y/z, this y is integrated over, not same as y for neutrinos outside
    a = 1 / np.pi ** 2 * var ** 2 * np.exp(np.sqrt(t ** 2 + var ** 2)) / ((np.exp(np.sqrt(t ** 2 + var ** 2)) + 1) ** 2)
    return a

def J(t): # t = m_e/T_gamma = x/z
    result, err = quad(intJ, 0, 50, args=(t))
    return result

@jit(nopython=True, cache=True)
def intY(var, t): 
    a = 1 / np.pi ** 2 * var ** 4 * np.exp(np.sqrt(t ** 2 + var ** 2)) / ((np.exp(np.sqrt(t ** 2 + var ** 2)) + 1) ** 2)
    return a

def Y(t):
    result, err = quad(intY, 0, 50, args=(t))
    return result

@jit(nopython=True, cache=True)
def intj(x, t): 
    return (np.exp(np.sqrt(x ** 2 + t ** 2))) / (((np.exp(np.sqrt(x ** 2 + t ** 2))) + 1) ** 2)

def j(t):
    result, err = quad(intj, 0, 50, args=(t))
    return 1 / np.pi ** 2 * result

@jit(nopython=True, cache=True)
def intK(x, t): 
    return x ** 2 / np.sqrt(x ** 2 + t ** 2) / ((np.exp(np.sqrt(x ** 2 + t ** 2))) + 1)

def K(t):
    result, err = quad(intK, 0, 50, args=(t))
    return 1 / np.pi ** 2 * result

@jit(nopython=True, cache=True)
def intk(x, t): 
    return 1 / np.sqrt(x ** 2 + t ** 2) / ((np.exp(np.sqrt(x ** 2 + t ** 2))) + 1)

def k(t):
    result, err = quad(intk, 0, 50, args=(t))
    return 1 / np.pi ** 2 * result

def J_prime(t):
    return -t * j(t)

def K_prime(t):
    return -t * k(t)

def Y_prime(t):
    return -3 * t * J(t)

def G1(t, Jprime, Kprime, Jay, Kay):
    return 2 * np.pi * alpha * (1/3 * Kprime + 1/6 * Jprime + Jprime * Kay + Jay * Kprime)

def G2(t, Jprime, Kprime, Jay, Kay):
    return -8 * np.pi * alpha * (1/6 * Kay + 1/6 * Jay - 1/2 * Kay ** 2 + Kay * Jay) \
    + 2 * np.pi * alpha * t * (1/6 * Kprime - Kay * Kprime + 1/6 * Jprime + Jprime * Kay + Jay * Kprime)



def cross_vectorized(Ps1, Ps2):
    arr1 = Ps1[::]
    arr2 = Ps2[::]
    return

def cross(arr1, arr2): # arr = [np.array(x,y,z)]
    x1, y1, z1 = arr1
    x2, y2, z2 = arr2
    return np.array([y1*z2-z1*y2, z1*x2-x1*z2, x1*y2-y1*x2])

@jit(nopython=True, cache=True)
def inte(xs, fxs):
    sum = 1/2 * np.sum((fxs[:-1] + fxs[1:]) * (xs[1:] - xs[:-1]))
    return sum

from scipy.interpolate import CubicSpline
def interpolate(newspan, span, f):
    interpolator = CubicSpline(span, f)
    return interpolator(newspan)



def normalize(v): # https://stackoverflow.com/questions/21030391/how-to-normalize-a-numpy-array-to-a-unit-vector
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v/norm
    else:
        norm = np.linalg.norm(v, axis = 0)
        return np.divide(v, norm, out=np.zeros_like(v), where=(norm!=0)) #https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero/37977222#37977222
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.integrate import solve_ivp

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






# astrophysical/thermo

rho_nu = lambda: 3 * 7/120 * np.pi ** 2 
# energy density of both neutrinos and antineutrinos for 3 species

# needs to be an integral over multiple momentum bins when those are added
# should equal for equilibrium
# is reduced by factor of Tcm^4
# change this to a integral expression

# can keep constant for now, calculate at end how evolve

def rho_gamma(z):
    return np.pi ** 2 / 15 * z ** 4

def rho_e(x, z):
    def func(y):
        a = y ** 2 * np.sqrt(y ** 2 + (x/z) ** 2) / (np.exp(np.sqrt(y ** 2 + (x/z) ** 2)) + 1)
        return a
    result, err = quad(func, 0, 100)
    return 1 / np.pi ** 2 * result * z ** 4

def P_e(x, z):
    def func(y):
        a = y ** 4 / (3 * np.sqrt(y ** 2 + (x/z) ** 2) * (np.exp(np.sqrt(y ** 2 + (x/z) ** 2)) + 1))
        return a
    result, err = quad(func, 0, 100)
    return 1 / np.pi ** 2 * result * z ** 4

def dz_dx(x, z): # must include extra terms for collisions
    num = x/z * J(x/z)
    denom = x**2 / z**2 * J(x/z) + Y(x/z) + 2 * np.pi**2 / 15 
    return num/denom







# math functions

def f_0(y):
    return 1/(np.exp(y) + 1)

def J(t): # t = m_e/T_gamma = x/z
    def func(var): # var = p/T_gamma for an electron = y/z, this y is integrated over, not same as y for neutrinos outside
        a = 1 / np.pi ** 2 * var ** 2 * np.exp(np.sqrt(t ** 2 + var ** 2)) / ((np.exp(np.sqrt(t ** 2 + var ** 2)) + 1) ** 2)
        return a
    result, err = quad(func, 0, 100)
    return result

def Y(t):
    def func(var): 
        a = 1 / np.pi ** 2 * var ** 4 * np.exp(np.sqrt(t ** 2 + var ** 2)) / ((np.exp(np.sqrt(t ** 2 + var ** 2)) + 1) ** 2)
        return a
    result, err = quad(func, 0, 100)
    return result

def cross(arr1, arr2): #arr = np.array(x,y,z)
    x1, y1, z1 = arr1
    x2, y2, z2 = arr2
    return np.array([y1*z2-z1*y2, z1*x2-x1*z2, x1*y2-y1*x2])

def inte(xs, fxs, method="trapezoid"):
    n = len(xs)
    if not len(fxs) == n:
        raise Exception("unequal length of inputs and outputs")
    sum = 0

    if method == "trapezoid":
        for i in range(1, n):
            sum += 0.5 * (fxs[i] + fxs[i-1]) * (xs[i] - xs[i-1])

    return sum







# no physics involved

def normalize(v): # https://stackoverflow.com/questions/21030391/how-to-normalize-a-numpy-array-to-a-unit-vector
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v/norm
    else:
        norm = np.linalg.norm(v, axis = 0)
        return np.divide(v, norm, out=np.zeros_like(v), where=(norm!=0)) #https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero/37977222#37977222
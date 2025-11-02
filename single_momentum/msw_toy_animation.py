import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# vacuum case
k = 1
t_starttransition = 2
t_endtransition = 2.1
n = 10
steps = 1000

theta0 = 0
thetaf = np.pi/3
initial_state = [np.sin(2 * theta0), 0, -np.cos(2*theta0)]

plt.close()

def cross(arr1, arr2): #arr = np.array(x,y,z)
    x1, y1, z1 = arr1
    x2, y2, z2 = arr2
    return np.array([y1*z2-z1*y2, z1*x2-x1*z2, x1*y2-y1*x2])

def deriv_p(t, p): # change so that B depends on t
    B = b(t)
    return cross(B, p)

def b(t):
    
    if t < t_starttransition:
        theta = theta0
    elif t < t_endtransition:    
        theta = theta0 + (t-t_starttransition)/(t_endtransition-t_starttransition) * (thetaf-theta0)
    else:
        theta = thetaf
    return k * np.array([np.sin(2*theta), 0, -np.cos(2*theta)])

T = np.linspace(0, n * t_endtransition, steps) # force P to step with known T

bunch = scipy.integrate.solve_ivp(deriv_p, [0., n * t_endtransition], initial_state, atol=1e-6, rtol=1e-6, t_eval=T) # tolerance: rtol, atol put at e-6
# T = bunch["t"]
Ps = bunch["y"]

# https://stackoverflow.com/questions/38118598/how-to-create-a-3d-animation

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.view_init(elev=20, azim=45, roll=0) # change angle of projection

Bs = np.transpose(np.array([b(t)/k for t in T]))
data = Ps

print(data.shape, Bs.shape)

trace, = ax.plot(data[0, 0], data[1,0], data[2,0])
trace.set_marker(",")
line, = ax.plot(data[0, 0], data[1,0], data[2,0]) # change to lines after
line.set_marker("o")
# vec, = ax.plot([0, Bs[0, 0]], [0, Bs[1, 0]], [0, Bs[2, 0]])
vec, = ax.plot(Bs[0, 0], Bs[1,0], Bs[2,0])
vec.set_marker("o")

def update(i, data, line):
    line.set_data(data[:2, max(i-2, 0):i])
    line.set_3d_properties(data[2, max(i-2, 0):i])
    trace.set_data(data[:2, :i])
    trace.set_3d_properties(data[2, :i])
    vec.set_data(Bs[:2, i-1])
    vec.set_3d_properties(Bs[2, i-1])

frames = 1000

ax.set_xlim3d([-1.0, 1.0])
ax.set_xlabel('X')

ax.set_ylim3d([-1.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([-1.0, 1.0])
ax.set_zlabel('Z')

ani = FuncAnimation(fig, update, frames, fargs=(data, line), interval=10, blit=False)
plt.show()

# problems: how to draw vec (table for now)
#           how to change interval (solved (?))

# instead of theta varying linear, go from 0 to given finite value (e.g. pi/4)
# 3 stages: constant-variable-constant: simulates MSW transition, can see actual transition probabilities
# probability as function of time stays same (take z component of P), plot the probability
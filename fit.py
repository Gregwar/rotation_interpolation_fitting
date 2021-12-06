import pinocchio as pin
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from transforms3d.quaternions import mat2quat, quat2mat
from scipy.optimize import minimize

# import pybullet as p
import time

p.connect(p.GUI)
frame = p.loadURDF('frame/robot.urdf')

R1 = pin.exp3(np.array([1, -3, 0.3]))
R2 = pin.exp3(np.array([3, 0.5, -0.5]))
w = pin.log3(R2 @ R1.T)
Rs = []

# First, generate a 
for t in np.linspace(0, 1, 50):
    R = pin.exp3(w * t + np.random.normal(0, 0.1, 3)) @ R1
    Rs.append(R)

_w_s = pin.log3(Rs[0])
_w = pin.log3(Rs[-1] @ Rs[0].T)
# w_s = np.array([0., 0., 0.])
# w = np.array([0., 0., 0.])
w_s = _w_s.copy()
w = _w.copy()

def score(params):
    w, w_s = params[:3], params[3:]
    s = 0
    R0 = pin.exp3(w_s)
    t = 0
    for R in Rs:
        error = pin.log3(pin.exp3(w * t) @ R0 @ R.T)
        s += error @ error
        t += 1/len(Rs)

    return s
def grad_fd(params):
    s1 = score(params)
    eps = 1e-6
    G_fd = []
    for k in range(6):
        params[k] += eps
        s2 = score(params)
        G_fd.append((s2 - s1)/eps)
        params[k] -= eps

    return G_fd

def grad(params):
    w, w_s = params[:3], params[3:]
    G = np.zeros(6)
    R0 = pin.exp3(w_s)
    t = 0
    for R in Rs:
        Rw = pin.exp3(w * t)
        error = pin.log3(Rw @ R0 @ R.T)
        Gw = 2 * t * (pin.Jlog3(Rw @ R0 @ R.T) @ pin.Jexp3(R @ R0.T @ w) @ R @ R0.T).T @ error
        Gw_s = 2 * (pin.Jlog3(Rw @ R0 @ R.T) @ R @ pin.Jexp3(w_s)).T @ error
        G[:3] += Gw
        G[3:] += Gw_s
        t += 1/len(Rs)

    return G

params = np.hstack((w, w_s))
sol = minimize(score, params, 
    # jac=grad,
    # method='BFGS'
    )
print(sol)

x = sol.x
w, w_s = x[:3], x[3:]

print(score(np.hstack((w, w_s))))
print(score(np.hstack((_w, _w_s))))

# WS = np.array([pin.log3(R) for R in Rs]).T
# V = np.array([pin.log3(pin.exp3(w*t) @ pin.exp3(w_s)) for t in np.linspace(0,1,len(Rs))]).T
# plt.plot(WS[0])
# plt.plot(WS[1])
# plt.plot(WS[2])
# plt.plot(V[0])
# plt.plot(V[1])
# plt.plot(V[2])
# plt.grid()
# plt.show()

s_slider = p.addUserDebugParameter('s', 0, 1, 0)
t_slider = p.addUserDebugParameter('t', 0, 1, 0)

while True:
    t = p.readUserDebugParameter(t_slider)
    if p.readUserDebugParameter(s_slider) < 0.5:
        R = Rs[min(len(Rs)-1, int(t*50))]
    else:
        R = pin.exp3(w * t) @ pin.exp3(w_s)

    qw, qx, qy, qz = mat2quat(R)
    p.resetBasePositionAndOrientation(frame, [0, 0, 0.1], [qx, qy, qz, qw])
    time.sleep(0.01)


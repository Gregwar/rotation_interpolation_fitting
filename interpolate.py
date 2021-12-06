import pinocchio as pin
import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat
import pybullet as p
import time

p.connect(p.GUI)
frame = p.loadURDF('frame/robot.urdf')

R1 = pin.exp3(np.array([1, -3, 0.3]))
R2 = pin.exp3(np.array([3, 0.5, -0.5]))
w = pin.log3(R2 @ R1.T)
t_slider = p.addUserDebugParameter('t', 0, 1, 0)

while True:
    t = p.readUserDebugParameter(t_slider)
    R = pin.exp3(w * t) @ R1

    qw, qx, qy, qz = mat2quat(R)
    p.resetBasePositionAndOrientation(frame, [0, 0, 0.1], [qx, qy, qz, qw])
    time.sleep(0.01)
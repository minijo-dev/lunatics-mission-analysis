import pandas as pd
import matplotlib.pyplot as plt

state = pd.read_csv("log_state.csv", sep='\s+')
q0 = state["q0"]
q1 = state["q1"]
q2 = state["q2"]
q3 = state["q3"]
bx = state["bx"]
by = state["by"]
bz = state["bz"]

euler = pd.read_csv("log_euler.csv", sep='\s+')
roll = euler["roll"]
pitch = euler["pitch"]
yaw = euler["yaw"]

w = pd.read_csv("log_angv.csv", sep='\s+')
wx = w["wx"]
wy = w["wy"]
wz = w["wz"]

plt.plot(q0)
plt.plot(q1)
plt.plot(q2)
plt.plot(q3)
plt.ylabel("Quaternions")
plt.show()

plt.plot(bx)
plt.plot(by)
plt.plot(bz)
plt.ylabel("Bias")
plt.show()

plt.plot(roll)
plt.plot(pitch)
plt.plot(yaw)
plt.ylabel("Euler Angles")
plt.show()

plt.plot(wx)
plt.plot(wy)
plt.plot(wz)
plt.ylabel("Angular Velocity")
plt.show()




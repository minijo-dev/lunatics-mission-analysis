import pandas as pd
import matplotlib.pyplot as plt

state = pd.read_csv("log.csv", sep='\s+')
q0 = state["q0"]
q1 = state["q1"]
q2 = state["q2"]
q3 = state["q3"]
wx = state["wx"]
wy = state["wy"]
wz = state["wz"]

state_true = pd.read_csv("log_true.csv", sep='\s+')
q0_true = state_true["q0"]
q1_true = state_true["q1"]
q2_true = state_true["q2"]
q3_true = state_true["q3"]
wx_true = state_true["wx"]
wy_true = state_true["wy"]
wz_true = state_true["wz"]

plt.plot(q0)
plt.plot(q0_true)
plt.show()

plt.plot(q1)
plt.plot(q1_true)
plt.show()

plt.plot(q2)
plt.plot(q2_true)
plt.show()

plt.plot(q3)
plt.plot(q3_true)
plt.show()

plt.plot(wx)
plt.plot(wx_true)
plt.show()

plt.plot(wy)
plt.plot(wy_true)
plt.show()

plt.plot(wz)
plt.plot(wz_true)
plt.show()

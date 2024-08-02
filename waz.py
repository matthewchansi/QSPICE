import matplotlib.pyplot as plt
import numpy as np

z = ""

with open ("res.txt", "r") as f:
    z = f.read()

#z = z.split("\n")
z = [i.strip() for i in z.strip().split("\n")]
f = [i.split(" ")[0] for i in z]
z = [float(i) for i in f]
f = [0] * (1500 - len(z))
f += z

print(len(f))
# make data
# plot


fig, ax = plt.subplots()
ax.plot(f, linewidth=2.0)
ax.set_ylabel("FOM = open-loop gain (dB)")
ax.set_xlabel("iterations")
ax.axvline(x=1000, color="r")
ax.tick_params(axis = "both")
ax.set_yticks([i * 5 for i in range(0, 10)])
ax.set_xticks([i * 250 for i in range(0, 7)])
ax.set_title("optimization for open-loop gain in 22n ota-5")
plt.show()
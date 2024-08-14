import matplotlib.pyplot as plt
import numpy as np

z = ""

vals = []

for i in range(2, 6):
    with open (f"res_{i}_mod.txt", "r") as f:
        z = f.read()

    #z = z.split("\n")
    z = [i.strip() for i in z.strip().split("\n")]
    #print(z)
    f = [i.split(" ")[7] for i in z]
    z = [float(i) for i in f]
    vals.append(z)

print(len(vals))

#print(len(f))

val_sobol = []
with open ("res1.txt", "r") as f:
    z = f.read()

#z = z.split("\n")
z = [i.strip() for i in z.strip().split("\n")][:512]
#print(z)
f = [i.split(" ")[7] if len(i) != 1 else 0 for i in z]
z = [float(i) for i in f]
val_sobol = z

sobol_y = range(0, 512)
mace_y = range(512, 512+256)
fig, ax = plt.subplots()
ax.plot(sobol_y, val_sobol, linewidth=2)
for i in vals:
    ax.plot(mace_y, i, linewidth=1.0)
ax.set_ylabel("FOM = open-loop gain (dB)")
ax.set_xlabel("iterations")
#ax.axvline(x=512, color="r")
ax.tick_params(axis = "both")
ax.set_yticks([i * 5 for i in range(0, 10)])
ax.set_xticks([i * 128 for i in range(0, 10)])

ax.set_ylim([0, 45])
ax.set_xlim([0, 768])
ax.set_title("MACE-128 optimization for OL gain in 22n ota-5\n pm > 85 gx > 45M, UCB beta from algo")
plt.savefig("waz2_mod.png")
plt.show()
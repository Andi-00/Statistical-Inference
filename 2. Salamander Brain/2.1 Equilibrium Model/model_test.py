import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['pgf.rcfonts'] = False
plt.rcParams['font.serif'] = []
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['errorbar.capsize'] = 2
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.title_fontsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
plt.rcParams['figure.figsize'] = (10, 7)

# Directory of the current folder
dir = "./2. Salamander Brain/"

# Number of spins
n = 160

# We first read in our infered model weights
h = np.genfromtxt(dir + "2.1 Equilibrium Model/Loss & Weights/H_history_eq.txt", delimiter = " ")
j = np.genfromtxt(dir + "2.1 Equilibrium Model/Loss & Weights/J_history_eq.txt", delimiter = " ")

j = np.reshape(j, (99, n, n))
# h = np.mean(h, axis = 0)
# j = np.mean(j, axis = 0)

h = h[-1]
j = j[-1]

# Train measurements
s_train = np.genfromtxt(dir + "Measurements/Equilibrium/s_train.txt", delimiter = " ")
s2_train = np.genfromtxt(dir + "Measurements/Equilibrium/s2_train.txt", delimiter = " ")
s3_train = np.reshape(np.genfromtxt(dir + "Measurements/Equilibrium/s3_train.txt", delimiter = " "), (n, n, n))

# Train measurements
s_test = np.genfromtxt(dir + "Measurements/Equilibrium/s_test.txt", delimiter = " ")
s2_test = np.genfromtxt(dir + "Measurements/Equilibrium/s2_test.txt", delimiter = " ")
s3_test = np.reshape(np.genfromtxt(dir + "Measurements/Equilibrium/s3_test.txt", delimiter = " "), (n, n, n))

# Loss values for the train and test data
train_loss = -np.einsum("i, i ->", h, s_train) - np.einsum("ij, ij ->", j, s2_train) / 2
test_loss = -np.einsum("i, i ->", h, s_test) - np.einsum("ij, ij ->", j, s2_test) / 2

print(train_loss)
print(test_loss)

# Plots of the train, test correlations
# fig, ax = plt.subplots()

# ax.scatter(s_train, s_test, color = "black", label = r"$s_i$")
# ax.scatter(s2_train, s2_test, color = "crimson", label = r"$s_i s_j$")
# ax.scatter(s3_train, s3_test, color = "royalblue", label = r"$s_i s_j s_k$")

# ax.grid()
# ax.legend()

# plt.show()

# Loss during the training
l = np.genfromtxt(dir + "2.1 Equilibrium Model/Loss & Weights/loss_eq.txt", delimiter = " ")
x = np.arange(len(l))

# Plot of the loss during the training
fig, ax = plt.subplots()

# Only every n_sep-th loss value is shown in the plot
n_sep = 4

# Loss function
ax.scatter(x[::n_sep], l[::n_sep], color = "crimson", zorder = 100, label = "$\mathcal L$")
ax.plot(x[::n_sep], l[::n_sep], color = "black", zorder = 10)

# Line that visualises the loss for the averaged couplings
ax.axhline(train_loss, color = "black", ls = "--", label = "\tilde \mathcal L")

ax.grid()
ax.set_ylabel("Loss Value $\mathcal L$")
ax.set_xlabel("Training Iteration Number")

plt.savefig(dir + "2.1 Equilibrium Model/Figures/loss_plot.png")


# Computation of the 3-spin-correlation. For that we again have to simulate spin flips

# Energy difference between the new and old state (E' - E)
dH = lambda k, s : 2 * s[k] * (h[k] + np.einsum("a, a ->", s, j[k]))

# Spin flip function
def spin_flip(N, s):

    # Spins to flip
    k = rng.integers(0, n, N)

    # List that stores the different spin configurations
    states = [s.copy()]

    # Loop to flip the spins
    for l in k:

        # Computation of the energy difference
        diff = dH(l, s)

        # Conditions for a spin flip
        if diff <= 0 : s[l] *= -1
        elif rng.random() < np.exp(-diff) : s[l] *= -1

        # A copy of the current spin configuration is saved in the states
        states.append(s.copy())
        
    return states

# Set up a seed
rng = np.random.default_rng(3)

# Set up initial spins
s = rng.integers(0, 2, n)
s[s == 0] = -1

n_train = int(1E6)
n_eq = int(2E4)

N = int(n_train + n_eq)

# MCMC algorithm for the spin flips of s_0
# We only store every n-th value for the update sweep
s = spin_flip(N, s)
s = np.array(s[- n_train :: n])

# We will now compute the 3-spin-correlation for the current spins

s3 = np.zeros((n, n, n))
c = 0

for i in range(n):
    print(i)
    for j in range(i + 1):
        for k in range(j + 1):
            s3[i, j, k] = np.mean(s[:, i] * s[:, j] * s[:, k])
            c+= 1

# Difference between the measured correlation and the 
ds3 = (s3 - s3_test).flatten()
ds3 = ds3[ds3 != 0]

bins = np.arange(-1, 1.1, 0.1) - 0.05
weights = np.ones(len(ds3)) / len(ds3)

fig, ax = plt.subplots()
ax.hist(ds3, color = "crimson", edgecolor = "black", zorder  = 10, bins = bins, weights = weights)

ax.grid()
ax.set_xlabel(r"$\langle s_i s_j s_k \rangle_\theta - \langle s_i s_j s_k \rangle_\mathrm{data}$")
ax.set_ylabel(r"Normalised Number of Counts $n/n_\mathrm{total}$")

plt.savefig(dir + "2.1 Equilibrium Model/Figures/s3_histogram.png")

# Hist of the test s3
s3_test = s3_test.flatten()
weights = np.ones(len(s3_test)) / len(s3_test)

fig, ax = plt.subplots()
ax.hist(s3_test.flatten(), color = "crimson", edgecolor = "black", zorder  = 10, bins = bins, weights = weights)

ax.grid()
ax.set_xlabel(r"$\langle s_i s_j s_k \rangle_\mathrm{data}$")
ax.set_ylabel(r"Normalised Number of Counts $n/n_\mathrm{total}$")

plt.savefig(dir + "2.1 Equilibrium Model/Figures/s3_test_histogram.png")

# Correlation Scatter Plots

s2 = np.zeros((n, n))

for i in range(n):
    for j in range(i + 1):
        s2[i, j] = np.mean(s[:, i] * s[:, j])

s = np.mean(s, axis = 0)

print(np.mean((s2 - s2_test) ** 2))
print(np.mean((s - s_test) ** 2))

# Plot of the magnetisation
fig, ax = plt.subplots()

ax.scatter(s_train, s, color = "crimson", zorder = 10)
ax.grid()

ax.set_ylabel(r"$\langle s_i \rangle_\theta$")
ax.set_xlabel(r"$\langle s_i \rangle_\mathrm{data}$")

plt.savefig(dir + "2.1 Equilibrium Model/Figures/scatter_s.png")

# Plot of the correlations
fig, ax = plt.subplots()

ax.scatter(s2_train, s2, color = "crimson", zorder = 10)
ax.grid()

ax.set_ylabel(r"$\langle s_i s_j\rangle_\theta$")
ax.set_xlabel(r"$\langle s_i s_j\rangle_\mathrm{data}$")

plt.savefig(dir + "2.1 Equilibrium Model/Figures/scatter_s2.png")

# Plot of the 3 spin correlations
fig, ax = plt.subplots()

ax.scatter(s3_train, s3, color = "crimson", zorder = 10)
ax.grid()

ax.set_ylabel(r"$\langle s_i s_j s_k\rangle_\theta$")
ax.set_xlabel(r"$\langle s_i s_j s_k\rangle_\mathrm{data}$")

plt.savefig(dir + "2.1 Equilibrium Model/Figures/scatter_s3.png")


plt.show()
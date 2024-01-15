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
h = np.genfromtxt(dir + "Equilibrium Model/Loss & Weights/H_history_eq.txt", delimiter = " ")
j = np.genfromtxt(dir + "Equilibrium Model/Loss & Weights/J_history_eq.txt", delimiter = " ")

h = np.mean(h, axis = 0)

j = np.reshape(j, (99, n, n))
j = np.mean(j, axis = 0)

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

# Loss during the training
l = np.genfromtxt(dir + "Equilibrium Model/Loss & Weights/loss_eq.txt", delimiter = " ")
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

plt.savefig(dir + "Equilibrium Model/Figures/loss_plot.png")


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

n_train = int(1E5)
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

plt.savefig(dir + "Equilibrium Model/Figures/s3_histogram.png")

s3_test = s3_test.flatten()
weights = np.ones(len(s3_test)) / len(s3_test)

fig, ax = plt.subplots()
ax.hist(s3_test.flatten(), color = "crimson", edgecolor = "black", zorder  = 10, bins = bins, weights = weights)

ax.grid()
ax.set_xlabel(r"$\langle s_i s_j s_k \rangle_\mathrm{data}$")
ax.set_ylabel(r"Normalised Number of Counts $n/n_\mathrm{total}$")

plt.savefig(dir + "Equilibrium Model/Figures/s3_test_histogram.png")

plt.show()
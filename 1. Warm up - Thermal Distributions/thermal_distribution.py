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
dir = "./1. Warm Up - Thermal Distributions/"

# Set a random seed
rng = np.random.default_rng()

# Number of spins
n = 5

# Generation of the couplings
h = rng.normal(0, 1, n)
j = rng.normal(0, 1 / n, (n, n))

# We set the diagonal of j to zero and make it symmetric
for i in range(n):
    j[i, i] = 0
    j[:, i] = j[i, :]

# Setup the spins
s = rng.integers(0, 2, n)
s[s == 0] = -1

# Calculation of the energy
H = lambda s, h, j: - np.einsum("i, i ->", h, s) - np.einsum("a, ab, b ->", s, j, s) / 2

# Energy difference between the new and old state (E' - E)
dH = lambda k, s, h, j : 2 * s[k] * (h[k] + np.einsum("a, a ->", s, j[k]))

# Function that performs N spin flips on the spins s with the couplings h and j
def spin_flip(N, s, h, j):

    # Spins to flip
    k = rng.integers(0, n, N)

    # List that stores the different spin configurations
    states = [s.copy()]

    # Loop to flip the spins
    for l in k:

        # Computation of the energy difference
        diff = dH(l, s, h, j)

        # Conditions for a spin flip
        if diff <= 0 : s[l] *= -1
        elif rng.random() < np.exp(-diff) : s[l] *= -1

        # A copy of the current spin configuration is saved in the states
        states.append(s.copy())
        
    return states
    

# Number of spin flips
n_train = int(1E4)
n_eq = int(500)

N = int(n_train + n_eq)

# Perform the spin flips on the spins s
s = spin_flip(N, s, h, j)

# Only the values in the thermal equilibrium are used to compute the averages
# Additionally we only save each n-th value for the update sweep
s = s[- n_train :: n]

# Computation of the correlations
s2 = np.einsum("ab, ac -> abc", s, s)

# Mean Magnetisation and Correlation of the train data
mean_s = np.mean(s, axis = 0)
mean_s2 = np.mean(s2, axis = 0)

# We are now going to infer the parameters h and j
# For that we first initialise some random values for h_0 and j_0
h_0 = rng.normal(0, 1, n)
j_0 = rng.normal(0, 1 / n, (n, n))

for i in range(n):
    j_0[i, i] = 0
    j_0[:, i] = j_0[i, :]

# Number of steps for the gradient descend
m = int(1E3)

# Learning rate a
a = 0.2

# Storage for the loss values
loss = []

# Comparison of the weights
lsq = []

h_mse = []
j_mse = []

# Loop for the gradient descend
for i in range(m):

    # Initialisation of the random spins
    s_0 = rng.integers(0, 2, n)
    s_0[s_0 == 0] = -1

    # MCMC algorithm for the spin flips of s_0
    s_0 = spin_flip(N, s_0, h_0, j_0)
    s_0 = s_0[- n_train :: n]

    # Computation of the correlations
    s2_0 = np.einsum("ab, ac -> abc", s_0, s_0)

    mean_s_0 = np.mean(s_0, axis = 0)
    mean_s2_0 = np.mean(s2_0, axis = 0)

    # Gradient descend update steps
    h_0 += a * (mean_s - mean_s_0)
    j_0 += a * (mean_s2 - mean_s2_0)

    # MSE of the weights
    lsq.append(np.mean(np.append(((h_0 - h) ** 2).flatten(), (((j_0 - j) * n) ** 2).flatten())))
 
    l = np.einsum("i, i ->", h_0, mean_s) + np.einsum("ij, ij ->", j_0, mean_s2) / 2
    loss.append(-l)

    h_mse.append(np.mean((h_0 - h) ** 2))
    j_mse.append(np.mean(((j_0 - j) * n) ** 2))

    # Progress
    if i % 10 == 0 : 
        print("Gradient descend step {}".format(i))
        print("mean squared weights : {:.5f}".format(lsq[-1]))
        print("loss : {:.5f}\n".format(-l))

        
x = np.arange(len(h_mse))

fig, ax = plt.subplots()

ax.plot(x, h_mse, color = "crimson", label = r"MSE of $h$", zorder = 10)
ax.plot(x, j_mse, color = "royalblue", label = r"MSE of $J$", zorder = 10)

ax.grid()
ax.legend()

ax.set_ylabel("Normalised MSE")
ax.set_xlabel("Train Step")

plt.savefig(dir + "./Figures/coupling_mse.png")
plt.show()

# Plot of the loss
fig, ax = plt.subplots()

x = np.arange(len(loss))

ax.plot(x, loss, color = "black")
ax.scatter(x, loss, color = "crimson", zorder = 10)

ax.grid()
ax.set_xlabel("Number of steps")
ax.set_ylabel("Negative log-likelihood $\mathcal L$")
ax.set_title("Loss during the training")

plt.savefig(dir + "Figures/loss_plot_2.png")

fig, ax = plt.subplots()

ax.scatter(lsq, loss, color = "black", zorder = 10)
ax.grid()
ax.set_ylabel("Loss $\mathcal L$")
ax.set_xlabel("MSE")

plt.savefig(dir + "Figures/loss_MSE_2.png")

# Comparison of the true and infered couplings
dh = h_0 - h
dj = (j_0 - j) * n

temp = []

for i in range(n):
    for k in range(i):
        temp.append(dj[i, k])

dj = np.array(temp)


print(h)
print(h_0)
print(j)
print(j_0)

# Plot of the histogram
fig, ax = plt.subplots()

errors = np.append(dh.flatten(), dj.flatten())

b = np.arange(-1, 1.12, 0.1) - 0.05

ax.grid()
ax.hist(errors, bins = b, color = "crimson", edgecolor = "black", zorder  = 10)

ax.set_xlabel(r"Normalised Error $\Delta \theta / \sigma$")
ax.set_ylabel("Number of counts")
ax.set_title("Histogram of the Deviations")

plt.savefig(dir + "Figures/hisogram_2.png")

plt.show()
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

#plt.rcParams['savefig.transparent'] = True
plt.rcParams['figure.figsize'] = (10, 7)

# Set a random seed
np.random.seed(3)

# Number of spins
n = 5

# Generation of the couplings
h = np.random.normal(0, 1, n)
j = np.random.normal(0, 1 / n, (n, n))

# We set the diagonal of j to zero and make it symmetric
for i in range(n):
    j[i, i] = 0
    j[:, i] = j[i, :]

# Setup the spins
s = np.random.randint(0, 2, n)
s[s == 0] = -1

print(s)

# Calculation of the energy
H = lambda s, h, j: - np.einsum("i, i ->", h, s) - np.einsum("a, ab, b ->", s, j, s) / 2

# Energy difference between the new and old state (E' - E)
dH = lambda k, s, h, j : 2 * s[k] * (h[k] + np.einsum("a, a ->", s, j[k]))

# Function that performs N spin flips on the spins s with the couplings h and j
def spin_flip(N, s, h, j):

    # Spins to flip
    k = np.random.randint(0, n, N)

    # List that stores the different spin configurations
    states = [s.copy()]

    # Loop to flip the spins
    for l in k:

        # Computation of the energy difference
        diff = dH(l, s, h, j)

        # Conditions for a spin flip
        if diff <= 0 : s[l] *= -1
        elif np.random.rand() < np.exp(-diff) : s[l] *= -1

        # A copy of the current spin configuration is saved in the states
        states.append(s.copy())
        
    return states
    

# Number of spin flips
N = int(2E4)

# Every n_sep-th configuration of the last n_train values is taken as train data
n_sep = 10
n_train = int(1E4)

# Perform the spin flips on the spins s
s = spin_flip(N, s, h, j)

# Only the values in the thermal equilibrium are used to compute the averages
s = s[- n_train :: n_sep]

# Computation of the correlations
s2 = np.einsum("ab, ac -> abc", s, s)

# Mean Magnetisation and Correlation of the train data
mean_s = np.mean(s, axis = 0)
mean_s2 = np.mean(s2, axis = 0)

print(mean_s)

print(H(mean_s, h, j))

# We are now going to infer the parameters h and j
# For that we first initialise some random values for h_0 and j_0
h_0 = np.random.normal(0, 1, n)
j_0 = np.random.normal(0, 1 / n, (n, n))

for i in range(n):
    j_0[i, i] = 0
    j_0[:, i] = j[i, :]

# Number of steps for the gradient descend
m = int(370)

# Learning rate a
a = 0.3

# Storage for the loss values
loss = []

# Comparison of the weights
lsq = []

# Loop for the gradient descend
for i in range(m):

    # Initialisation of the random spins
    s_0 = np.random.randint(0, 2, n)
    s_0[s_0 == 0] = -1

    # MCMC algorithm for the spin flips of s_0
    s_0 = spin_flip(N, s_0, h_0, j_0)
    s_0 = s_0[- n_train :: n_sep]

    # Computation of the correlations
    s2_0 = np.einsum("ab, ac -> abc", s_0, s_0)

    mean_s_0 = np.mean(s_0, axis = 0)
    mean_s2_0 = np.mean(s2_0, axis = 0)

    # Gradient descend update steps
    h_0 += a * (mean_s - mean_s_0)
    j_0 += a * (mean_s2 - mean_s2_0)

    # MSE of the weights
    lsq.append(np.mean(np.append(((h_0 - h) ** 2).flatten(), (((j_0 - j) * n) ** 2).flatten())))
 
    l = np.einsum("b, ab -> a", h_0, s) + np.einsum("ab, bc, ac -> a", s, j_0, s) / 2
    l = -np.mean(l)
    loss.append(l)

    # Progress
    if i % 10 == 0 : 
        print(i)
        print(lsq[-1])
        
# Plot of the loss
fig, ax = plt.subplots()

x = np.arange(len(loss))

ax.plot(x, loss, color = "black")
ax.scatter(x, loss, color = "crimson", zorder = 10)

ax.grid()
ax.set_xlabel("Number of steps")
ax.set_ylabel("Negative log-likelihood $\mathcal L$")
ax.set_title("Loss during the training")

plt.savefig("./thermal_images/loss_plot.png")

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

plt.savefig("./thermal_images/hisogram.png")

plt.show()
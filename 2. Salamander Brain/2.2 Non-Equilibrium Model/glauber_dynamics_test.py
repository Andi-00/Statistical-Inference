import numpy as np
from matplotlib import pyplot as plt
from time import time

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
dir = "./2. Salamander Brain/2.2 Non-Equilibrium Model/"

# Set a random seed
rng = np.random.default_rng(3)

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

# Function that performs N spin flips on the spins s with the couplings h and j
def spin_flip(N, s, h, j):

    # New spin configurations
    k = rng.integers(0, 2, (N, n))
    k[k == 0] = -1

    # List that stores the different spin configurations
    states = [s.copy()]

    # Loop to flip the spins
    for l in k:

        # Computations of theta and s theta to later compute the probability
        theta = np.einsum("ij, j -> i", j, s) + h
        s_theta = l * theta

        # Probability for the spin flip
        p = np.prod(np.exp(s_theta) / (2 * np.cosh(theta)))

        # Save the new state with the fliped spin with a probability of p
        if rng.random() < p : s = l.copy()

        # A copy of the current spin configuration is saved in the states
        states.append(s)

    return states


# Number of spin flips
N = int(1E4)

# Generation of the time series
s = np.array(spin_flip(N, s, h, j))

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
a = 0.3

# We will here define s_t+1 and s_t as
s0 = s[:-1].copy()
s1 = s[1:].copy()

# Loss computation for the first round
theta = np.einsum("ij, nj -> ni", j_0, s0) + np.reshape(h_0, (1, -1))
loss = [np.mean(np.sum(s1 * theta - np.log(2 * np.cosh(theta)), axis = -1))]

error = np.append((h_0 - h) ** 2, ((j_0 - j) * n) ** 2) 
lsq = [np.mean(error)]

# We will use this variable later to track the time
start = time()

# We will save the last 50 weights and then perform an average to get the final value
j_hist = []
h_hist = []

print(s1.shape)
print(j_0.shape)

print(np.einsum("ij, nj -> ni", j_0, s0).shape)

# Training loop
for i in range(m):

    # Computation of theta for the gradients
    theta = np.einsum("ij, nj -> ni", j_0, s0) + h_0
    
    # Computation of the updates 
    dh = np.mean(s1, axis = 0) - np.mean(np.tanh(theta), axis = 0)
    dj = np.mean(np.einsum("ni, nj -> nij", s1, s0), axis = 0) - np.mean(np.einsum("ni, nj -> nij", np.tanh(theta), s0), axis = 0)

    # Gradient descend update steps
    h_0 += a * dh
    j_0 += a * dj

    # Loss function
    l = np.mean(np.sum(s1 * theta - np.log(2 * np.cosh(theta)), axis = -1))
    loss.append(l)

    # MSE of the weights
    error = np.append((h_0 - h) ** 2, ((j_0 - j) * n) ** 2) 
    lsq.append(np.mean(error))

    # Progress
    if i % 10 == 0 : 

        end = time()

        print("Gradient descend step {}".format(i))
        print("Loss value     : {:.4f}".format(l))
        print("Time / 10 steps: {:6.2f} s \n".format(end - start))

        start = time()

    # We store the last 50 weights and do an average
    if i > m - 50:
        h_hist.append(h_0)
        j_hist.append(j_0)

# Final weight values
h_0 = np.mean(h_hist, axis = 0)
j_0 = np.mean(j_hist, axis = 0)

# Final loss value computation
theta = np.einsum("ij, nj -> ni", j_0, s0) + np.reshape(h_0, (1, -1))
loss.append(np.mean(np.sum(s1 * theta - np.log(2 * np.cosh(theta)), axis = -1)))

# Least squared error of the weights
error = np.append((h_0 - h) ** 2, ((j_0 - j) * n) ** 2) 
lsq.append(np.mean(error))

# log likelihood
loss = np.array(loss)


# Plot of the loss
fig, ax = plt.subplots()

x = np.arange(len(loss))

ax.plot(x, loss, color = "black")
ax.scatter(x, loss, color = "crimson", zorder = 10)

ax.grid()
ax.set_xlabel("Number of steps")
ax.set_ylabel(r"log-likelihood $\mathcal L = \ln P(s(t +1)|s(t))$")
ax.set_title("Loss during the training")

plt.savefig(dir + "Figures/test_loss_plot.png")

# Plot if the MSE vs Loss
fig, ax = plt.subplots()

ax.scatter(lsq, loss, color = "black", zorder = 10)
ax.grid()
ax.set_ylabel("Loss $\mathcal L$")
ax.set_xlabel("MSE")

plt.savefig(dir + "Figures/test_loss_MSE.png")

# Comparison of the true and infered couplings
dh = h_0 - h
dj = (j_0 - j) * n

# temp = []

# for i in range(n):
#     for k in range(i):
#         temp.append(dj[i, k])

# dj = np.array(temp)

print(h_0 - h)
# print(h_0)
print(j_0 - j)
# print(j_0)

# Plot of the histogram
fig, ax = plt.subplots()

# errors = dj.flatten()
errors = np.append(dh.flatten(), dj.flatten())

print(np.mean(errors ** 2))

b = np.arange(-1, 1.12, 0.1) - 0.05

ax.grid()
ax.hist(errors, color = "crimson", edgecolor = "black", zorder  = 10)

ax.set_xlabel(r"Normalised Error $\Delta \theta / \sigma$")
ax.set_ylabel("Number of counts")
ax.set_title("Histogram of the Deviations")

plt.savefig(dir + "Figures/histogram_2.png")

plt.show()
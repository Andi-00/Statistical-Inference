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
dir = "2. Salamander Brain/"

# Data Preperation

# Splitting the data into test and train data and
# Computing the different means and storing them to reduce the computational effort for later
s = np.genfromtxt(dir + "Measurements/bint.txt", delimiter = " ")

# Set a random seed
rng = np.random.default_rng(3)

# Replacing the zeros with -1
s[s == 0] = -1

# Shuffelling the list for the seperation in test and train data
rng.shuffle(s, axis = -1)

# Train / Test = 80% / 20%
t = int(0.8 * s.shape[-1])

# Train and Test Set
s_train = s[:, :t]
s_test = s[:, t:]

# Set a random seed
rng = np.random.default_rng(3)

# We can here choose if we infere symmetric or non symmetric couplings
symmetric = True
txt = "asym"

# Number of spins
n = 160

# Generation of the couplings that we will later infere
h = rng.normal(0, 1, n)
j = rng.normal(0, 1 / n, (n, n))

# We set the diagonal of j to zero and make it symmetric if its needed

if symmetric:
    txt = "sym"
    for i in range(n):
        j[i, i] = 0
        j[:, i] = j[i, :]


s0 = s_train[:-1]
s1 = s_train[1:]

# Loss computation for the first round
theta = np.einsum("ij, nj -> ni", j, s0) + np.reshape(h, (1, -1))
loss = [np.mean(np.sum(s1 * theta - np.log(2 * np.cosh(theta)), axis = -1))]

# We will use this variable later to track the time
start = time()

# We will save the last 50 weights and then perform an average to get the final value
j_hist = []
h_hist = []

# Number of training iterations 
m = int(3E3)

# Learning rate
a = 0.4

# Trinaing loop
for i in range(m):

    # Computation of theta for the gradients
    theta = np.einsum("ij, nj -> ni", j, s0) + h
    
    # Computation of the updates 
    dh = np.mean(s1, axis = 0) - np.mean(np.tanh(theta), axis = 0)
    dj = np.mean(np.einsum("ni, nj -> nij", s1, s0), axis = 0) - np.mean(np.einsum("ni, nj -> nij", np.tanh(theta), s0), axis = 0)

    # Gradient descend update steps
    h += a * dh
    j += a * dj

    # Loss function
    l = np.mean(np.sum(s1 * theta - np.log(2 * np.cosh(theta)), axis = -1))
    loss.append(l)

    # Progress
    if i % 10 == 0 : 

        end = time()

        print("Gradient descend step {}".format(i))
        print("Loss value     : {:.4f}".format(-l))
        print("Time / 10 steps: {:6.2f} s \n".format(end - start))

        start = time()

    # We store the last 50 weights and do an average
    if i > m - 50:
        h_hist.append(h)
        j_hist.append(j)


h = np.mean(h_hist, axis = 0)
j = np.mean(j_hist, axis = 0)

# Saving the weights for later comparison of the results
np.savetxt(dir + "2.2 Non-Equilibrium Model/Loss & Weights/h_" + txt + ".txt", h, delimiter = " ")
np.savetxt(dir + "2.2 Non-Equilibrium Model/Loss & Weights/j_" + txt + ".txt", j, delimiter = " ")
np.savetxt(dir + "2.2 Non-Equilibrium Model/Loss & Weights/loss_" + txt + ".txt", loss, delimiter = " ")

# Plot of the loss
fig, ax = plt.subplots()

x = np.arange(len(loss))

ax.plot(x, loss, color = "black")
ax.scatter(x, loss, color = "crimson", zorder = 10)

ax.grid()
ax.set_xlabel("Number of steps")
ax.set_ylabel("Negative log-likelihood $\mathcal L$")
ax.set_title("Loss during the training")

plt.savefig(dir + "2.2 Non-Equilibrium Model/Figures/train_loss_curve_" + txt + ".png")
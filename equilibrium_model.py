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



# Splitting the data into test and train data and
# Computing the different means and storing them in "./Measurements" to reduce the program time for later
s = np.genfromtxt("./Measurements/bint.txt", delimiter = " ")

# Set a random seed
rng = np.random.default_rng(3)

print(s.shape)

s[s == 0] = -1

rng.shuffle(s, axis = -1)

# Train / Test = 80% / 20%
t = int(0.8 * s.shape[-1])

# Train and Test Set
s_train = s[:t]
s_test = s[t:]

s2_train = np.zeros((160, 160))
s2_test = np.zeros((160, 160))

s3_train = np.zeros((160, 160, 160))
s3_test = np.zeros((160, 160, 160))

for i in range(160):
    print(i)
    for j in range(i + 1):
        s2_train[i, j] = np.mean(s[i, :] * s[j, :])
        s2_train[j, i] = np.mean(s[i, :] * s[j, :])

        s2_test[i, j] = np.mean(s[i, :] * s[j, :])
        s2_test[j, i] = np.mean(s[i, :] * s[j, :])

        for k in range(j + 1):
            s3_train[i, j, k] = np.mean(s[i, :] * s[j, :] * s[k, :])
            s3_test[i, j, k] = np.mean(s[i, :] * s[j, :] * s[k, :])


s_train = np.mean(s_train, axis = -1)
s_test = np.mean(s_test, axis = -1)

s3_train = s3_train.reshape((160, -1))
s3_test = s3_test.reshape((160, -1))

np.savetxt("./Measurements/Equilibrium/s_train.txt", s_train, delimiter = " ")
np.savetxt("./Measurements/Equilibrium/s2_train.txt", s2_train, delimiter = " ")
np.savetxt("./Measurements/Equilibrium/s3_train.txt", s3_train, delimiter = " ")

np.savetxt("./Measurements/Equilibrium/s_test.txt", s_test, delimiter = " ")
np.savetxt("./Measurements/Equilibrium/s2_test.txt", s2_test, delimiter = " ")
np.savetxt("./Measurements/Equilibrium/s3_test.txt", s3_test, delimiter = " ")

# # Set a random seed
# rng = np.random.default_rng(3)

# # Number of spins
# n = 160

# # Generation of the couplings that we will later infere
# h = rng.normal(0, 1, n)
# j = rng.normal(0, 1 / n, (n, n))

# # We set the diagonal of j to zero and make it symmetric
# for i in range(n):
#     j[i, i] = 0
#     j[:, i] = j[i, :]

# # Energy difference between the new and old state (E' - E)
# dH = lambda k, s : 2 * s[k] * (h[k] + np.einsum("a, a ->", s, j[k]))

# # Function that performs N spin flips on the spins s with the couplings h and j
# def spin_flip(N, s):

#     # Spins to flip
#     k = rng.integers(0, n, N)

#     # List that stores the different spin configurations
#     states = [s.copy()]

#     # Loop to flip the spins
#     for l in k:

#         # Computation of the energy difference
#         diff = dH(l, s)

#         # Conditions for a spin flip
#         if diff <= 0 : s[l] *= -1
#         elif rng.random() < np.exp(-diff) : s[l] *= -1

#         # A copy of the current spin configuration is saved in the states
#         states.append(s.copy())
        
#     return states

# # Values of the measured magnetisation and correlation
# mean_s = np.genfromtxt("./Measurements/magnetisation.txt", delimiter = " ")
# mean_s2 = np.genfromtxt("./Measurements/correlation.txt", delimiter = " ")

# # Number of spin flips
# n_train = int(1E5)
# n_eq = int(2E4)

# N = int(n_train + n_eq)

# # Every n_sep-th configuration of the last n_train values is taken as train data
# n_sep = 5

# # Number of steps for the gradient descend
# m = int(2E3)

# # Learning rate a
# a = 0.4

# # Storage for the loss values with the initial value
# loss = [-np.einsum("i, i ->", h, mean_s) - np.einsum("ij, ij ->", j, mean_s2) / 2]

# # We will use this variable later to track the time
# start = time()

# # Loop for the gradient descend
# for i in range(m):

#     # Initialisation of the random spins
#     s_0 = rng.integers(0, 2, n)
#     s_0[s_0 == 0] = -1

#     # MCMC algorithm for the spin flips of s_0
#     s_0 = spin_flip(N, s_0)
#     s_0 = s_0[- n_train :: n_sep]

#     # Computation of the correlations
#     s2_0 = np.einsum("ab, ac -> abc", s_0, s_0)

#     mean_s_0 = np.mean(s_0, axis = 0)
#     mean_s2_0 = np.mean(s2_0, axis = 0)

#     # Gradient descend update steps
#     h += a * (mean_s - mean_s_0)
#     j += a * (mean_s2 - mean_s2_0)
 
#     l = np.einsum("i, i ->", h, mean_s) + np.einsum("ij, ij ->", j, mean_s2) / 2
#     loss.append(-l)

#     # Progress
#     if i % 10 == 0 : 

#         end = time()

#         print("Gradient descend step {}".format(i))
#         print("Loss value     : {:.3f}".format(-l))
#         print("Time / 10 steps: {:6.2f} s \n".format(end - start))

#         start = time()
    

# # Saving the weights for later comparison of the results
# np.savetxt("./Loss & Weights/h_eq.txt", h, delimiter = " ")
# np.savetxt("./Loss & Weights/j_eq.txt", j, delimiter = " ")
# np.savetxt("./Loss & Weights/loss_eq.txt", loss, delimiter = " ")

# # Plot of the loss
# fig, ax = plt.subplots()

# x = np.arange(len(loss))

# ax.plot(x, loss, color = "black")
# ax.scatter(x, loss, color = "crimson", zorder = 10)

# ax.grid()
# ax.set_xlabel("Number of steps")
# ax.set_ylabel("Negative log-likelihood $\mathcal L$")
# ax.set_title("Loss during the training")

# plt.savefig("./Equilibrium_Images/loss_curve_test.png")
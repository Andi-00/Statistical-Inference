import numpy as np

# Set a random seed
np.random.seed(0)

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

# Calculation of the energy
H = lambda s, h, j: - np.einsum("i, i ->", h, s) - np.einsum("a, ab, b ->", s, j, s) / 2

# Energy difference between the new and old state (E' - E)
dH = lambda k, s, h, j : 2 * (h[k] * s[k] + np.einsum("a, a ->", s, j[k]) * s[k])

# Function that performs N spin flips on the spins s with the couplings h and j
def spin_flip(N, s, h, j, save = False):

    # Spins to flip and look at the enegy difference
    k = np.random.randint(0, n, N)

    # Generation of random number between 0 and 1
    # If dE is negative but the number p is lower that the probability, then the spin flip is still done
    p = np.random.rand(N)

    states = []

    # Loop to flip the spins
    for count, l in enumerate(k):
        # Computation of the energy difference
        diff = dH(l, s, h, j)

        # Condition for a spin flip, when its energetically favourable
        if diff <= 0 : s[l] *= -1
        # Otherwise, thermal fluctuations can induce a spin flip with a certain probability
        elif np.exp(-diff) < p[l] : s[l] *= -1
        
        # Values that are stred in an array
        if save and count % 10 == 0: states.append(s)
    
    if save : return states
    

# Number of spin flips
N = int(2E4)

# Perform the spin flips on the spins s
s = spin_flip(N, s, h, j, save = True)

# Only the values in the thermal equilibrium are used to comute the averages
s = s[int(len(s) / 2): ]

# Computation of the correlations
s2 = np.einsum("ab, ac -> abc", s, s)

# Magnetisation and Correlation
mean_s = np.mean(s, axis = 0)
mean_s2 = np.mean(s2, axis = 0)

# We are now going to infer the parameters h and j
# For that we initialise some random chosen numbers h_0 and j_0
h_0 = np.random.normal(0, 1, n)
j_0 = np.random.normal(0, 1 / n, (n, n))

for i in range(n):
    j_0[i, i] = 0
    j_0[:, i] = j[i, :]

# Number of steps for the gradient descend
m = int(10)

# Learning rate a
a = 0.3

# Loop for the gradient descend
for i in range(m):

    # Initialisation of the random spins
    s_0 = np.random.randint(0, 2, n)

    # MCMC algorithm for the spin flips of s_0
    s_0 = spin_flip(N, s_0, h_0, j_0, save = True)
    s_0 = s_0[int(len(s_0) / 2): ]

    # Computation of the correlations
    s2_0 = np.einsum("ab, ac -> abc", s_0, s_0)

    mean_s_0 = np.mean(s_0, axis = 0)
    mean_s2_0 = np.mean(s2_0, axis = 0)

    print(mean_s_0)

    # Gradient descend update steps
    h_0 += a * (mean_s - mean_s_0)
    j_0 += a * (mean_s2 - mean_s2_0)

print(h)
print(h_0)
print()
print(j)
print(j_0)



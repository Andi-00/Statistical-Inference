import numpy as np

# Set a random seed
np.random.seed(10)

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
def spin_flip(N, s, h, j):

    # Spins to flip and look at the enegy difference
    k = np.random.randint(0, n, N)

    # Generation of random number between 0 and 1
    # If dE is negative but the number p is lower that the probability, then the spin flip is still done
    p = np.random.rand(N)

    # Loop to flip the spins
    for l in k:
        diff = dH(l, s, h, j)

        if diff <= 0 : s[l] *= -1
        elif np.exp(-diff) < p[l] : s[l] *= -1

# Number of spin flips
N = int(1E5)

# Perform the spin flips on the spins s
spin_flip(N, s, h, j)

# Energy of the system
print(H(s, h, j))

# Matrix containing all products of si * sj
s2 = np.einsum("i, j -> ij", s, s)

# Magnetisation and Correlation
mean_s = np.mean(s)
mean_s2 = np.mean(s2)

print(mean_s)
print(mean_s2)

# We are now going to infer the parameters h and j
# For that we initialise some random chosen numbers h0 and j0
h0 = np.random.normal(0, 1, n)
j0 = np.random.normal(0, 1 / n, (n, n))

for i in range(n):
    j0[i, i] = 0
    j0[:, i] = j[i, :]

# Number of steps for the gradient descend


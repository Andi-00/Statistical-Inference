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
H = lambda : - np.einsum("i, i ->", h, s) - np.einsum("a, ab, b ->", s, j, s) / 2

# Energy difference between the new and old state (E' - E)
dH = lambda k : 2 * (h[k] * s[k] + np.einsum("a, a ->", s, j[k]) * s[k])

# number of spin flips until convergence
N = int(1E5)

# Spins to flip and look at the enegy difference
k = np.random.randint(0, n, N)

# Generation of random number between 0 and 1
# If dE is negative but the number p is lower that the probability, then the spin flip is still done
p = np.random.rand(N)

# Loop to flip the spins
for l in k:
    diff = dH(l)

    if diff <= 0 : s[l] *= -1
    elif np.exp(-diff) < p[l] : s[l] *= -1

# Energy of the system
print(H())

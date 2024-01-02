import numpy as np

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
E = - np.einsum("i, i ->", h, s) - np.einsum("a, ab, b ->", s, j, s) / 2

dE = lambda k : -h[k] * s[k] - np.einsum("a, a ->", s, j[k]) * s[k]


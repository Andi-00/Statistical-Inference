import numpy as np

# Number of spins
n = 5

# Generation of the couplings
j = np.random.normal(0, 1, (n, n))

# We use the diagonal to assign the values of h and replace the values in j by zero
h = np.zeros(n)

for i in range(n):
    h[i] = j[i, i]
    j[i, i] = 0



import numpy as np

# Creating the mean of the spins and correlations to save them as a txt file
# This will reduce the time it takes to load the data
s = np.genfromtxt("./bint.txt", delimiter = " ")

s[s == 0] = -1
s2 = np.einsum("ac, bc -> abc", s, s)

mean_s = np.mean(s, axis = 1)
mean_s2 = np.mean(s2, axis = 2)

np.savetxt("./mean_spin.txt", s, delimiter = " ")
np.savetxt("./mean_spin_correlation.txt", s2, delimiter = " ")


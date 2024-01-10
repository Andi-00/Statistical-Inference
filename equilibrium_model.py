import numpy as np

# # Creating the mean of the spins and correlations to save them as a txt file
# # This will reduce the time it takes to load the data
# s = np.genfromtxt("./bint.txt", delimiter = " ")

# s[s == 0] = -1

# s2 = np.zeros((160, 160))

# for i in range(160):
#     print(i)
#     for j in range(i + 1):
#         s2[i, j] = np.mean(s[i, :] * s[j, :])
#         s2[j, i] = np.mean(s[i, :] * s[j, :])

# mean_s = np.mean(s, axis = -1)

# np.savetxt("./mean_spin.txt", mean_s, delimiter = " ")
# np.savetxt("./mean_spin_correlation.txt", s2, delimiter = " ")



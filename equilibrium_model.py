import numpy as np

s = np.genfromtxt("./bint.txt", delimiter = ",")
s = np.reshape(s, (160, -1))

s.shape


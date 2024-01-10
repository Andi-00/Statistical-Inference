import numpy as np

print(283041 - 19 * 297 / 20E-3)
s = np.genfromtxt("./bint.txt", delimiter = " ")

s[s == 0] = -1

print(s[:, 100])
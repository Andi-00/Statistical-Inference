import numpy as np
from matplotlib import pyplot as plt

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

# Directory of the current folder
dir = "2. Salamander Brain/Equilibrium Model/"

# Number of spins
n = 160

# We first read in our infered model weights
h = np.genfromtxt(dir + "Equilibrium Model/Loss & Weights/H_history_eq.txt", delimiter = " ")
j = np.genfromtxt(dir + "Equilibrium Model/Loss & Weights/H_history_eq.txt", delimiter = " ")

h = np.mean(h, axis = 0)

j = np.reshape(j, (len(j), n, n))
j = np.mean(j, axis = 0)

# Train measurements
s_train = np.genfromtxt(dir + "Measurements/Equilibrium/s_train.txt", delimiter = " ")
s2_train = np.genfromtxt(dir + "Measurements/Equilibrium/s2_train.txt", delimiter = " ")
s3_train = np.genfromtxt(dir + "Measurements/Equilibrium/s3_train.txt", delimiter = " ")

# Train measurements
s_test = np.genfromtxt(dir + "Measurements/Equilibrium/s_test.txt", delimiter = " ")
s2_test = np.genfromtxt(dir + "Measurements/Equilibrium/s2_test.txt", delimiter = " ")
s3_test = np.genfromtxt(dir + "Measurements/Equilibrium/s3_test.txt", delimiter = " ")

# Loss values for the train and test data
train_loss = -np.einsum("i, i ->", h, s_train) - np.einsum("ij, ij ->", j, s_train) / 2
test_loss = -np.einsum("i, i ->", h, s_test) - np.einsum("ij, ij ->", j, s_test) / 2


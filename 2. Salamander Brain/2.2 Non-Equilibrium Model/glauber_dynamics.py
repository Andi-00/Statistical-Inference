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
dir = "./2. Salamander Brain/"

# Set a random seed
rng = np.random.default_rng(3)

# Number of spins
n = 5

# Generation of the couplings
h = rng.normal(0, 1, n)
j = rng.normal(0, 1 / n, (n, n))

# We set the diagonal of j to zero
for i in range(n):
    j[i, i] = 0

# Setup the spins
s = rng.integers(0, 2, n)
s[s == 0] = -1


import os
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import numpy as np
from lib.utils import read_csv, find_cargo_root
from lib.blocking import block

data_folder = os.path.join(find_cargo_root(), "data")
save_folder = os.path.join(os.path.dirname(find_cargo_root()), "report", "assets")
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)

N = 10
true_val = 15

bruteforce = read_csv(os.path.join(data_folder, "E_vs_MCs_BruteForceMetropolis.csv"))
importance = read_csv(os.path.join(data_folder, "E_vs_MCs_ImportanceMetropolis.csv"))
x = [100, 1000, 3000, 5000, 7000, 10000]
#bruteforce_std = [np.sqrt(block(np.array(vals))[1]) for vals in [bruteforce["energy[au]"][1:up_to] for up_to in x]]
#importance_std = [np.sqrt(block(np.array(vals))[1]) for vals in [importance["energy[au]"][1:up_to] for up_to in x]]

#plt.plot(x, bruteforce_std, "-o", label="Brute-force")
#plt.plot(x, importance_std, "-o", label="Importance")
plt.plot(range(len(bruteforce["energy[au]"][1:])), bruteforce["energy[au]"][1:], "-o", label="Brute-force")
plt.plot(range(len(importance["energy[au]"][1:])), importance["energy[au]"][1:], "-o", label="Importance")
plt.xlabel("Monte Carlo cycles")
plt.ylabel(r"Energy")
plt.legend()
plt.savefig(os.path.join(save_folder, "E_vs_MCs_all.png"))
plt.show()
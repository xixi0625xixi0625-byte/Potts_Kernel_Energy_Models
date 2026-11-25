import numpy as np
import matplotlib.pyplot as plt 

a = "ACGT"
L = 6
q = len(a)

folder = "C:/Users/Lenovo/Desktop/CSHL/VScode/"

energy_true = np.load(folder + "energy_true.npy")
energy_fit = np.load(folder + "energy_fit.npy")
theta_true = np.load(folder + "theta_true.npy")
theta_fit = np.load(folder + "theta_fit.npy")

plt.scatter(energy_true, energy_fit, s = 5)
plt.xlabel("True Energy")
plt.ylabel("Fitted Energy")
plt.show()
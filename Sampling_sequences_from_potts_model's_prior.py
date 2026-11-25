import numpy as np
from scipy.special import softmax
from itertools import product
from seqs_prob.potts_model import potts_energy

a = "ACGT"
q = len(a)
L = 6

all_sequences = np.array(list(product(range(q), repeat=L)))
pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]

sigma_h_squared = 1
sigma_J_squared = 1
np.random.seed(0)

# h
h = np.random.normal(0, sigma_h_squared, size=(L, q))
# J
J = np.zeros((L, L, q, q))
for i, j in pairs:
    J[i, j] = np.random.normal(0, sigma_J_squared, size=(q, q))
        
theta = np.concatenate([h.flatten(), J.flatten()])
np.set_printoptions(threshold=np.inf)
print(theta)

# Energy for every possible sequences
E_all = potts_energy(all_sequences, theta, L, q)
print(E_all)
P = softmax(E_all)

folder = "C:/Users/Lenovo/Desktop/CSHL/VScode/"
np.save(folder + "theta_true.npy", theta)
np.save(folder + "energy_true.npy", E_all)

folder = "C:/Users/Lenovo/Desktop/CSHL/VScode/"
np.savetxt(folder + "theta_true.txt", theta)
np.savetxt(folder + "energy_true.txt", E_all)

index = np.random.choice(len(all_sequences), size=1000, p=P)
samples = all_sequences[index]

#for s in samples[:10]:
    #rint("".join(a[k] for k in s))

print(["".join(a[k] for k in s) for s in samples])


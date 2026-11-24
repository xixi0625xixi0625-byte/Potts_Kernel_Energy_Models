import numpy as np
from scipy.special import softmax
from itertools import product
from seqs_prob.potts_model import potts_energy

a = "ACGT"
q = len(a)
L = 4

all_sequences = np.array(list(product(range(q), repeat=L)))
pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]

sigma_h_squared = 1
sigma_J_squared = 1
np.random.seed(0)

#h
h = np.random.normal(0, sigma_h_squared, size=(L, q))
#J
J = np.zeros((L, L, q, q))
for i, j in pairs:
    J[i, j] = np.random.normal(0, sigma_J_squared, size=(q, q))
        
theta = np.concatenate([h.flatten(), J.flatten()])
print(theta)

E_all = potts_energy(all_sequences, theta)
P = softmax(E_all)

index = np.random.choice(len(all_sequences), size=100, p=P)
samples = all_sequences[index]

for s in samples[:10]:
    print("".join(a[k] for k in s))

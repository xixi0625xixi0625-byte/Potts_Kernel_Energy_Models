import numpy as np

#dE/dh_i(a) =1 if x_i = a, 0 otherwise
#dE/dJ_ij(a,b) = 1 if x_i = a and x_j = b, 0 otherwise

def potts_energy_gradient(seqs, L, q):
    pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]
    M = len(seqs)
    rows = np.arange(M)
    gradient_h = np.zeros((M, L, q))
    gradient_J = np.zeros((M, L, L, q, q))
    for i in range(L):
        gradient_h[rows, i, seqs[:, i]] = 1
    for i, j in pairs:
        gradient_J[rows, i, j, seqs[:, i], seqs[:, j]] = 1
    return np.concatenate([gradient_h.reshape(M, -1), gradient_J.reshape(M, -1)], axis=1)


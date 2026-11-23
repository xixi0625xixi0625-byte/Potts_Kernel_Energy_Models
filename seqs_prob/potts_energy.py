import numpy as np

def potts_energy(seqs, theta, L, q):
    pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]
    h = theta[:L*q].reshape(L,q)
    J = theta[L*q:].reshape(L,L,q,q)
    E = h[np.arange(L), seqs].sum(axis=1)
    for i, j in pairs:
        E += J[i, j, seqs[:, i], seqs[:, j]]
    return E